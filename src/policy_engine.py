"""Inference wrapper for the ChessGPT transformer policy."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import chess
import torch
import torch.nn as nn


def encode_board(board: chess.Board) -> List[int]:
    """Encode a chess.Board into 65 token ids (64 squares + side-to-move)."""
    piece_ids = [0] * 64
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        code = piece.piece_type  # 1..6
        piece_ids[square] = code if piece.color == chess.WHITE else code + 6
    piece_ids.append(0 if board.turn == chess.WHITE else 1)
    return piece_ids


class ChessPolicyModel(nn.Module):
    """Same architecture as in train_model.py, reused for inference."""

    def __init__(
        self,
        num_moves: int,
        history_length: int,
        embed_dim: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.board_len = 65
        self.history_len = max(0, history_length)
        self.use_history = self.history_len > 0
        self.seq_len = self.board_len + self.history_len
        self.piece_embed = nn.Embedding(13, embed_dim)
        if self.use_history:
            self.history_embed = nn.Embedding(num_moves + 1, embed_dim, padding_idx=num_moves)
        else:
            self.history_embed = None
        self.pos_embed = nn.Embedding(self.seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_moves)

    def forward(
        self,
        board_tokens: torch.Tensor,
        history_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = board_tokens.size(0)
        embeddings = self.piece_embed(board_tokens)
        if self.use_history:
            if history_tokens is None:
                raise ValueError("History tokens required when history length > 0")
            history_emb = self.history_embed(history_tokens)
            embeddings = torch.cat([embeddings, history_emb], dim=1)
        seq_len = embeddings.size(1)
        pos_ids = torch.arange(seq_len, device=board_tokens.device).unsqueeze(0).expand(
            batch_size, seq_len
        )
        embeddings = embeddings + self.pos_embed(pos_ids)
        encoded = self.transformer(embeddings)
        pooled = encoded.mean(dim=1)
        return self.fc(pooled)


def _detect_device() -> torch.device:
    preferred = os.getenv("CHESSGPT_DEVICE")
    if preferred:
        preferred = preferred.lower()
        if preferred == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("CHESSGPT_DEVICE=mps but MPS backend is unavailable")
        if preferred == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CHESSGPT_DEVICE=cuda but CUDA backend is unavailable")
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PolicyEngine:
    """Loads the trained transformer and produces moves for a python-chess board."""

    def __init__(self) -> None:
        self._setup_paths()
        self.history_len = int(os.getenv("CHESSGPT_HISTORY_LEN", "0"))
        self.use_history = self.history_len > 0
        self.device = _detect_device()
        self.move_to_idx = self._load_move_mapping(self.mapping_path)
        self.model = ChessPolicyModel(
            num_moves=len(self.move_to_idx),
            history_length=self.history_len,
        )
        state_dict = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.pad_idx = len(self.move_to_idx)
        self.search_depth = max(0, int(os.getenv("CHESSGPT_SEARCH_DEPTH", "0")))
        self.search_width = max(1, int(os.getenv("CHESSGPT_SEARCH_WIDTH", "5")))

    def _setup_paths(self) -> None:
        src_dir = Path(__file__).resolve().parent
        repo_root = src_dir.parent
        self.model_path = Path(
            os.getenv("CHESSGPT_MODEL_PATH", repo_root / "models" / "chess_policy_modal_hist0.pt")
        )
        self.mapping_path = Path(
            os.getenv("CHESSGPT_MAPPING_PATH", repo_root / "data" / "move_mapping.json")
        )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self.model_path}. "
                "Set CHESSGPT_MODEL_PATH if the file lives elsewhere."
            )
        if not self.mapping_path.exists():
            raise FileNotFoundError(
                f"Move mapping not found at {self.mapping_path}. "
                "Set CHESSGPT_MAPPING_PATH if the file lives elsewhere."
            )

    @staticmethod
    def _load_move_mapping(path: Path) -> Dict[str, int]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(move): int(idx) for move, idx in data.items()}

    def select_move(self, board: chess.Board) -> Tuple[chess.Move, Dict[chess.Move, float]]:
        """Return the selected move and a probability distribution over legal moves."""
        scored_moves = self._policy_scores(board)
        if not scored_moves:
            legal_moves = list(board.generate_legal_moves())
            fallback_move = random.choice(legal_moves)
            return fallback_move, {fallback_move: 1.0}
        move_probabilities = dict(scored_moves)
        if self.search_depth <= 0:
            best_move = max(move_probabilities.items(), key=lambda item: item[1])[0]
            return best_move, move_probabilities
        best_move, updated_probs = self._choose_with_search(board, move_probabilities)
        return best_move, updated_probs

    def _history_indices(self, board: chess.Board) -> List[int]:
        if not self.use_history:
            return []
        stack = list(board.move_stack)
        recent = stack[-self.history_len :]
        idxs = [self.move_to_idx.get(move.uci(), self.pad_idx) for move in recent]
        if len(idxs) < self.history_len:
            idxs.extend([self.pad_idx] * (self.history_len - len(idxs)))
        return idxs

    def _policy_scores(self, board: chess.Board) -> List[Tuple[chess.Move, float]]:
        board_tensor = torch.tensor(
            encode_board(board), dtype=torch.long, device=self.device
        ).unsqueeze(0)
        history_tensor = None
        if self.use_history:
            history_tensor = torch.tensor(
                self._history_indices(board), dtype=torch.long, device=self.device
            ).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(board_tensor, history_tensor).squeeze(0).cpu()

        legal_moves = list(board.generate_legal_moves())
        scored_moves: List[Tuple[chess.Move, float]] = []
        for move in legal_moves:
            idx = self.move_to_idx.get(move.uci())
            if idx is None:
                continue
            scored_moves.append((move, float(logits[idx])))

        if not scored_moves:
            return []

        scores_tensor = torch.tensor([score for _, score in scored_moves], dtype=torch.float32)
        probs = torch.softmax(scores_tensor, dim=0).tolist()
        return sorted(
            [(move, prob) for (move, _), prob in zip(scored_moves, probs, strict=True)],
            key=lambda item: item[1],
            reverse=True,
        )

    def _search_value(self, board: chess.Board, depth: int, maximizing: bool) -> float:
        scores = self._policy_scores(board)
        if not scores:
            return -1.0 if maximizing else 1.0
        if depth == 0:
            return scores[0][1]

        candidates = scores[: self.search_width]
        if maximizing:
            best = -float("inf")
            for move, prob in candidates:
                board.push(move)
                response = self._search_value(board, depth - 1, False)
                board.pop()
                score = prob - response
                if score > best:
                    best = score
            return best
        else:
            worst = float("inf")
            for move, prob in candidates:
                board.push(move)
                response = self._search_value(board, depth - 1, True)
                board.pop()
                score = prob - response
                if score < worst:
                    worst = score
            return worst

    def _choose_with_search(
        self, board: chess.Board, move_probs: Dict[chess.Move, float]
    ) -> Tuple[chess.Move, Dict[chess.Move, float]]:
        evaluated: List[Tuple[chess.Move, float]] = []
        candidates = sorted(
            move_probs.items(), key=lambda item: item[1], reverse=True
        )[: self.search_width]
        for move, prob in candidates:
            board.push(move)
            response = self._search_value(board, self.search_depth - 1, False)
            board.pop()
            evaluated.append((move, prob - response))
        best_move = max(evaluated, key=lambda item: item[1])[0]
        return best_move, move_probs


def load_policy_engine() -> PolicyEngine:
    """Factory with lazy instantiation for hot-reload friendliness."""
    return PolicyEngine()
