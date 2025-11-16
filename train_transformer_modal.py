"""
Modal training script for the bespoke ChessGPT transformer with optional multi-GPU (DDP) support.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import modal
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, random_split
from tqdm import tqdm

try:
    import chess
except ImportError as exc:
    raise ImportError("python-chess is required") from exc


@dataclass
class TrainConfig:
    dataset_path: str
    mapping_path: str
    epochs: int
    batch_size: int
    embed_dim: int
    nhead: int
    num_layers: int
    ff_dim: int
    lr: float
    history_length: int
    val_split: float
    model_path: str
    num_workers: int
    val_interval: int
    resume_from: Optional[str]


# --- Model + dataset ---------------------------------------------------------


def encode_board(board: chess.Board) -> List[int]:
    ids = [0] * 64
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        code = piece.piece_type
        ids[square] = code if piece.color == chess.WHITE else code + 6
    ids.append(0 if board.turn == chess.WHITE else 1)
    return ids


class ChessDataset(Dataset):
    def __init__(self, dataset_path: str, history_length: int, pad_idx: int) -> None:
        self.history_length = history_length
        self.pad_idx = pad_idx
        self.samples: List[Tuple[str, int, List[int]]] = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                history = [int(idx) for idx in record.get("history", [])]
                self.samples.append((record["fen"], int(record["move_idx"]), history))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fen, move_idx, history = self.samples[idx]
        board = chess.Board(fen)
        encoded = encode_board(board)
        history_tokens = self._prepare_history(history)
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(history_tokens, dtype=torch.long),
            torch.tensor(move_idx, dtype=torch.long),
        )

    def _prepare_history(self, history: List[int]) -> List[int]:
        if self.history_length <= 0:
            return []
        filtered = [idx for idx in history if 0 <= idx < self.pad_idx]
        if len(filtered) >= self.history_length:
            return filtered[-self.history_length :]
        return filtered + [self.pad_idx] * (self.history_length - len(filtered))


class ChessPolicyModel(nn.Module):
    def __init__(
        self,
        num_moves: int,
        history_length: int,
        embed_dim: int,
        nhead: int,
        num_layers: int,
        ff_dim: int,
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
        embeddings = self.piece_embed(board_tokens)
        if self.use_history:
            if history_tokens is None:
                raise ValueError("History tokens required when history is enabled.")
            history_emb = self.history_embed(history_tokens)
            embeddings = torch.cat([embeddings, history_emb], dim=1)
        seq_len = embeddings.size(1)
        pos_ids = torch.arange(seq_len, device=board_tokens.device).unsqueeze(0).expand(
            embeddings.size(0), seq_len
        )
        embeddings = embeddings + self.pos_embed(pos_ids)
        encoded = self.transformer(embeddings)
        pooled = encoded.mean(dim=1)
        return self.fc(pooled)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    eval_model = model.module if isinstance(model, DDP) else model
    eval_model.eval()
    total = 0.0
    with torch.no_grad():
        for board_tokens, history_tokens, targets in dataloader:
            board_tokens = board_tokens.to(device)
            history_tokens = history_tokens.to(device)
            targets = targets.to(device)
            with torch.amp.autocast("cuda"):
                outputs = eval_model(board_tokens, history_tokens if eval_model.use_history else None)
                loss = criterion(outputs, targets)
            total += loss.item() * board_tokens.size(0)
    eval_model.train()
    return total / len(dataloader.dataset)


def train_model(config: TrainConfig, rank: int, world_size: int) -> None:
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(rank)
    is_master = rank == 0

    def log(*args, **kwargs):
        if is_master:
            print(*args, **kwargs)

    with open(config.mapping_path, "r", encoding="utf-8") as f:
        move_to_idx: Dict[str, int] = json.load(f)
    num_moves = len(move_to_idx)

    dataset = ChessDataset(config.dataset_path, config.history_length, pad_idx=num_moves)
    if config.val_split > 0:
        val_size = max(1, int(len(dataset) * config.val_split))
        train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
    else:
        train_dataset, val_dataset = dataset, None

    pin_memory = device.type == "cuda"
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        )
        if val_dataset
        else None
    )

    model = ChessPolicyModel(
        num_moves=num_moves,
        history_length=config.history_length,
        embed_dim=config.embed_dim,
        nhead=config.nhead,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
    ).to(device)

    if config.resume_from:
        ckpt = Path(config.resume_from)
        if ckpt.exists():
            log(f"Loading checkpoint from {ckpt}")
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, config.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_master:
            progress = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", dynamic_ncols=True)
        else:
            progress = train_loader

        running_loss = 0.0
        seen = 0
        for board_tokens, history_tokens, targets in progress:
            board_tokens = board_tokens.to(device)
            history_tokens = history_tokens.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            base_model = model.module if isinstance(model, DDP) else model
            history_arg = history_tokens if base_model.use_history else None
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(board_tokens, history_arg)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size_now = board_tokens.size(0)
            running_loss += loss.item() * batch_size_now
            seen += batch_size_now
            if is_master:
                progress.set_postfix(loss=f"{running_loss/max(seen,1):.4f}")

        totals = torch.tensor([running_loss, seen], device=device)
        if world_size > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss = totals[0].item()
        total_seen = max(1, int(totals[1].item()))
        epoch_loss = total_loss / total_seen
        log(f"Epoch {epoch} train loss: {epoch_loss:.4f}")

        if (
            val_loader is not None
            and (epoch % config.val_interval == 0 or epoch == config.epochs)
            and is_master
        ):
            val_loss = evaluate(model, val_loader, criterion, device)
            log(f"Validation loss: {val_loss:.4f}")

    base_model = model.module if isinstance(model, DDP) else model
    if is_master:
        torch.save(base_model.state_dict(), config.model_path)
        log(f"Saved model to {config.model_path}")


def train_worker(rank: int, world_size: int, config: TrainConfig) -> None:
    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        train_model(config, rank, world_size)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


# --- Modal configuration ------------------------------------------------------

IMAGE = (
    modal.Image.debian_slim()
    .apt_install("git", "curl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": f"/root/.local/bin:$PATH"})
    .run_commands("/root/.local/bin/uv venv /root/.uv-env")
    .env({"PATH": f"/root/.uv-env/bin:$PATH"})
    .run_commands(
        "/root/.local/bin/uv pip install --python /root/.uv-env/bin/python "
        "numpy==2.3.4 torch==2.9.1 tqdm==4.67.1 python-chess==1.999"
    )
)

DATA_VOLUME = modal.Volume.from_name("chessgpt-datasets", create_if_missing=True)
OUTPUT_VOLUME = modal.Volume.from_name("chessgpt-models", create_if_missing=True)
GPU_CONFIG = "A100-40GB:2"

app = modal.App("chessgpt-transformer")


@app.function(
    image=IMAGE,
    gpu=GPU_CONFIG,
    timeout=60 * 60 * 12,
    volumes={"/data": DATA_VOLUME, "/outputs": OUTPUT_VOLUME},
)
def train_remote(
    dataset_file: str,
    mapping_file: str,
    epochs: int,
    batch_size: int,
    embed_dim: int,
    nhead: int,
    num_layers: int,
    ff_dim: int,
    lr: float,
    history_length: int,
    val_split: float,
    model_name: str,
    num_workers: int,
    val_interval: int,
    resume_from: Optional[str],
    num_gpus: int,
) -> None:
    dataset_path = str(Path("/data") / dataset_file)
    mapping_path = str(Path("/data") / mapping_file)
    output_path = str(Path("/outputs") / f"{model_name}.pt")
    resume_path = str(Path("/outputs") / resume_from) if resume_from else None
    config = TrainConfig(
        dataset_path=dataset_path,
        mapping_path=mapping_path,
        epochs=epochs,
        batch_size=batch_size,
        embed_dim=embed_dim,
        nhead=nhead,
        num_layers=num_layers,
        ff_dim=ff_dim,
        lr=lr,
        history_length=history_length,
        val_split=val_split,
        model_path=output_path,
        num_workers=num_workers,
        val_interval=max(1, val_interval),
        resume_from=resume_path,
    )

    world_size = max(1, num_gpus)
    if world_size > 1:
        mp.spawn(train_worker, args=(world_size, config), nprocs=world_size, join=True)
    else:
        train_worker(0, 1, config)
    OUTPUT_VOLUME.commit()


@app.local_entrypoint()
def main(
    dataset: str = "train_dataset_hist0_full.jsonl",
    mapping: str = "move_mapping_hist0_full.json",
    epochs: int = 12,
    batch_size: int = 1536,
    embed_dim: int = 320,
    nhead: int = 8,
    num_layers: int = 8,
    ff_dim: int = 1280,
    lr: float = 6e-4,
    history_length: int = 0,
    val_split: float = 0.05,
    model_name: str = "chess_policy_modal_hist0_full",
    num_workers: int = 8,
    val_interval: int = 2,
    resume_from: str = "",
    num_gpus: int = 1,
) -> None:
    train_remote.remote(
        dataset,
        mapping,
        epochs,
        batch_size,
        embed_dim,
        nhead,
        num_layers,
        ff_dim,
        lr,
        history_length,
        val_split,
        model_name,
        num_workers,
        val_interval,
        resume_from if resume_from else None,
        num_gpus,
    )