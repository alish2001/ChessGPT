"""
Simple training script for a transformer‑based chess policy model.

This script loads a preprocessed dataset containing chess positions and
corresponding moves (represented by integer indices), defines a small
transformer model, and trains it using PyTorch.  The goal is to
produce a lightweight model capable of selecting a good move given a
board state without any search.  The architecture is inspired by
recent small transformer designs for chess【364005705419582†L105-L135】, but
adapted for efficiency and simplicity.

To prepare the dataset, first run ``data_ingestion.py`` to convert
raw data sources (puzzle CSVs, PGNs) into a JSON lines file with
entries of the form ``{"fen": fen, "move_idx": idx}`` and a JSON
mapping from move strings to indices.  The fields in the Lichess
puzzles dataset are documented by Lichess: ``FEN`` is the position
before the opponent's blunder, while the ``Moves`` column contains
the blunder followed by the solution sequence【901679508138493†L94-L102】.

Usage example:

```
python train_model.py \
    --dataset data/train_dataset.jsonl \
    --move_mapping data/move_mapping.json \
    --epochs 5 \
    --batch_size 256 \
    --model_path models/chess_policy.pt
```

This will train the model for a few epochs and save the learned
weights to ``models/chess_policy.pt``.  Adjust the hyperparameters
for your hardware and dataset size.  Note that training on millions
of positions may take many hours; you can start with a smaller
subset to validate the pipeline.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm

try:
    import chess
except ImportError as exc:
    raise ImportError(
        "python-chess is required for board parsing. Install it via `pip install chess`."
    ) from exc


def encode_board(board: chess.Board) -> List[int]:
    """Encode a chess.Board into a sequence of piece identifiers.

    The encoding assigns an integer to each square on the board based on
    the occupying piece:

    - ``0`` for an empty square.
    - ``1``–``6`` for White's pawn/knight/bishop/rook/queen/king.
    - ``7``–``12`` for Black's pawn/knight/bishop/rook/queen/king.

    The board is scanned by square index (0 to 63) in the default
    ``python-chess`` order.  An additional element is appended to
    represent the side to move (``0`` for White, ``1`` for Black).

    Args:
        board: A ``chess.Board`` instance representing the position.

    Returns:
        A list of integers of length 65 (64 squares + side to move).
    """
    piece_ids = [0] * 64
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        code = piece.piece_type  # 1..6
        if piece.color == chess.WHITE:
            piece_ids[square] = code
        else:
            piece_ids[square] = code + 6
    # Append side to move: 0 for white, 1 for black
    piece_ids.append(0 if board.turn == chess.WHITE else 1)
    return piece_ids


class ChessDataset(Dataset):
    """PyTorch Dataset for chess positions, histories, and moves."""

    def __init__(self, dataset_path: str, history_length: int, pad_idx: int) -> None:
        self.history_length = history_length
        self.pad_idx = pad_idx
        self.samples: List[Tuple[str, int, List[int]]] = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                fen = record['fen']
                move_idx = int(record['move_idx'])
                history = [int(idx) for idx in record.get('history', [])]
                self.samples.append((fen, move_idx, history))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fen, move_idx, history = self.samples[idx]
        board = chess.Board(fen)
        encoded_board = encode_board(board)
        history_tokens = self._prepare_history(history)
        return (
            torch.tensor(encoded_board, dtype=torch.long),
            torch.tensor(history_tokens, dtype=torch.long),
            torch.tensor(move_idx, dtype=torch.long),
        )

    def _prepare_history(self, history: List[int]) -> List[int]:
        if self.history_length <= 0:
            return []
        # ensure ids are valid (some datasets may lack mapping entries)
        filtered = [idx for idx in history if 0 <= idx < self.pad_idx]
        if len(filtered) >= self.history_length:
            return filtered[-self.history_length :]
        padded = filtered + [self.pad_idx] * (self.history_length - len(filtered))
        return padded


class ChessPolicyModel(nn.Module):
    """A transformer policy network that incorporates recent move history."""

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
            activation='gelu',
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
        board_emb = self.piece_embed(board_tokens)
        embeddings = board_emb
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


def select_device(args: argparse.Namespace) -> torch.device:
    """Select the best available device honoring CLI overrides."""
    if args.device:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        if args.device == "mps":
            if not torch.backends.mps.is_available():
                raise ValueError("MPS requested but not available")
            if args.no_mps:
                raise ValueError("MPS requested but --no_mps flag is set")
        return torch.device(args.device)

    if torch.backends.mps.is_available() and not args.no_mps:
        return torch.device("mps")
    if torch.cuda.is_available() and not args.no_cuda:
        return torch.device("cuda")
    return torch.device("cpu")


def train(args: argparse.Namespace) -> None:
    # Load move mapping to determine the output dimension
    with open(args.move_mapping, 'r', encoding='utf-8') as f:
        move_to_idx: Dict[str, int] = json.load(f)
    num_moves = len(move_to_idx)
    print(f"Number of unique moves: {num_moves}")

    # Create dataset and dataloader
    pad_idx = num_moves
    dataset = ChessDataset(args.dataset, history_length=args.history_length, pad_idx=pad_idx)
    print(f"Loaded {len(dataset)} samples")
    if args.val_split > 0.0:
        val_size = max(1, int(len(dataset) * args.val_split))
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        print(f"Training samples: {train_size}, Validation samples: {val_size}")
    else:
        train_dataset = dataset
        val_dataset = None
        print("Validation split disabled")

    device = select_device(args)
    pin_memory = device.type == 'cuda'
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        if val_dataset is not None
        else None
    )

    # Initialize model
    model = ChessPolicyModel(
        num_moves=num_moves,
        history_length=args.history_length,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    print(f"Using device: {device}")
    model.to(device)

    # Optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    amp_enabled = (device.type == 'cuda') and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    model.train()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            unit="batch",
            dynamic_ncols=True,
            disable=not args.progress_bar,
        )
        examples_seen = 0
        for batch_idx, (board_tokens, history_tokens, targets) in enumerate(progress, start=1):
            board_tokens = board_tokens.to(device)
            history_tokens = history_tokens.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            if amp_enabled:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model(board_tokens, history_tokens if model.use_history else None)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(board_tokens, history_tokens if model.use_history else None)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            batch_size = board_tokens.size(0)
            running_loss += loss.item() * batch_size
            examples_seen += batch_size

            if args.progress_bar:
                current_lr = optimizer.param_groups[0]['lr']
                avg_loss = running_loss / examples_seen
                progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}")
            elif (batch_idx % args.log_interval) == 0:
                avg_loss = running_loss / examples_seen
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {avg_loss:.4f} LR: {current_lr:.2e}"
                )
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch} completed - mean loss: {epoch_loss:.4f}")

        if val_loader is not None and (epoch % args.val_interval == 0 or epoch == args.epochs):
            val_loss = evaluate(model, val_loader, criterion, device, amp_enabled)
            print(f"Validation loss: {val_loss:.4f}")

    # Save trained model
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")


def evaluate(
    model: ChessPolicyModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for board_tokens, history_tokens, targets in dataloader:
            board_tokens = board_tokens.to(device)
            history_tokens = history_tokens.to(device)
            targets = targets.to(device)
            if amp_enabled:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model(board_tokens, history_tokens if model.use_history else None)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(board_tokens, history_tokens if model.use_history else None)
                loss = criterion(outputs, targets)
            running_loss += loss.item() * board_tokens.size(0)
    return running_loss / len(dataloader.dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transformer-based chess policy model")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the JSON lines dataset file')
    parser.add_argument('--move_mapping', type=str, required=True, help='Path to the move-to-index JSON file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for AdamW')
    parser.add_argument('--embed_dim', type=int, default=128, help='Dimension of token embeddings')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads in the transformer')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer encoder layers')
    parser.add_argument('--ff_dim', type=int, default=512, help='Dimension of the feed-forward network in the transformer')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate in the transformer')
    parser.add_argument('--log_interval', type=int, default=50, help='How many batches to wait before logging training status')
    parser.add_argument('--model_path', type=str, default='chess_policy.pt', help='File path to save the trained model')
    parser.add_argument('--history_length', type=int, default=0, help='Number of previous moves (ply) to provide as model context')
    parser.add_argument('--val_split', type=float, default=0.05, help='Fraction of data reserved for validation (0 disables validation)')
    parser.add_argument('--val_interval', type=int, default=1, help='Evaluate on the validation set every N epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for dataset splits')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader worker processes')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'], help='Manually select the compute device')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available')
    parser.add_argument('--no_mps', action='store_true', help='Disable Apple Metal Performance Shaders even if available')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable tqdm progress bars (falls back to interval logging)')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision training even if supported')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.progress_bar = not args.no_progress_bar
    args.val_interval = max(1, args.val_interval)
    train(args)
