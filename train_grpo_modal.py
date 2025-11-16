"""
Modal-powered GRPO training loop for ChessGPT.

This script mirrors the structure from Modal's GRPO+TRL example:
https://modal.com/docs/examples/grpo_trl

Usage:
    1. Make sure you have the Modal CLI logged in locally.
    2. Create a Modal volume that contains the ingested JSONL dataset(s):
         modal volume create chessgpt-datasets
         modal volume put chessgpt-datasets data/train_dataset_hist0_games8k.jsonl
         modal volume put chessgpt-datasets data/move_mapping_hist0_games8k.json
    3. Adjust the defaults below (dataset filename, base LLM, steps, etc.).
    4. Kick off training:
         modal run train_grpo_modal.py --dataset train_dataset_hist0_games8k.jsonl \
             --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --episodes 400

The GRPO Trainer expects a reward function; here we provide a simple reward of +1
if the generated move exactly matches the dataset label, and -0.25 otherwise.
For real reinforcement learning you can replace `simple_reward_fn` with logic
that runs a chess engine, rollout environment, or tournament feedback.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import modal

IMAGE = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch==2.9.1",
        "accelerate==1.1.1",
        "transformers==4.45.1",
        "trl==0.12.0",
        "datasets==2.21.0",
        "python-chess==1.999",
    )
)

DATA_VOLUME = modal.Volume.from_name("chessgpt-datasets", create_if_missing=True)


app = modal.App("chessgpt-grpo")


def load_dataset(
    dataset_path: Path,
    mapping_path: Path,
    limit: int | None = None,
) -> Dict[str, List[str]]:
    with mapping_path.open("r", encoding="utf-8") as f:
        move_to_idx = json.load(f)
    idx_to_move = {int(idx): move for move, idx in move_to_idx.items()}

    prompts: List[str] = []
    ids: List[str] = []
    records = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            record = json.loads(line)
            fen = record["fen"]
            move_idx = int(record["move_idx"])
            move_san = idx_to_move.get(move_idx)
            if move_san is None:
                continue
            history = record.get("history", [])
            record_id = f"{idx}"
            prompt = (
                f"[ID={record_id}] Given the chess position in FEN format:\n"
                f"{fen}\n"
                "Respond with the best move in simple UCI notation (e.g. e2e4) on a single line.\n"
                "Move:"
            )
            prompts.append(prompt)
            ids.append(record_id)
            records.append({"id": record_id, "fen": fen, "move": move_san, "history": history})
            if limit is not None and len(prompts) >= limit:
                break
    return {"prompts": prompts, "ids": ids, "records": records}


def make_reward_fn(answer_lookup: Dict[str, str]):
    from typing import List as _List

    def reward_fn(samples: _List[str], prompts: _List[str], **_) -> _List[float]:
        rewards: _List[float] = []
        for sample, prompt in zip(samples, prompts):
            sample_move = sample.strip().split()[0]
            identifier = prompt.split("[ID=")[1].split("]")[0]
            target_move = answer_lookup.get(identifier)
            if target_move is None:
                rewards.append(0.0)
                continue
            rewards.append(1.0 if sample_move == target_move else -0.25)
        return rewards

    return reward_fn


@app.function(
    image=IMAGE,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    timeout=60 * 60 * 6,
    volumes={"/data": DATA_VOLUME},
)
def train_remote(
    dataset_file: str,
    mapping_file: str,
    model_name: str,
    total_episodes: int,
    limit_samples: int | None = None,
) -> None:
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import AutoModelForCausalLMWithValueHead, GRPOConfig, GRPOTrainer

    dataset_path = Path("/data") / dataset_file
    mapping_path = Path("/data") / mapping_file
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} not found inside the Modal volume")
    if not mapping_path.exists():
        raise FileNotFoundError(f"{mapping_path} not found inside the Modal volume")

    print(f"Loading dataset from {dataset_path}")
    ds_payload = load_dataset(dataset_path, mapping_path, limit_samples)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_ds = Dataset.from_dict({"prompt": ds_payload["prompts"]})
    answers = {record["id"]: record["move"] for record in ds_payload["records"]}
    reward_fn = make_reward_fn(answers)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )

    config = GRPOConfig(
        total_episodes=total_episodes,
        num_generations=4,
        max_prompt_length=256,
        max_completion_length=16,
        learning_rate=5e-6,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        save_steps=total_episodes // 4 or 1,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[reward_fn],
        args=config,
        train_dataset=train_ds,
    )

    trainer.train()

    output_dir = Path("/data/grpo_runs") / f"{model_name.replace('/', '_')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_pretrained(str(output_dir))
    DATA_VOLUME.commit()
    print(f"Saved GRPO checkpoint to {output_dir}")


@app.local_entrypoint()
def main(
    dataset: str = "train_dataset_hist0_games8k.jsonl",
    mapping: str = "move_mapping_hist0_games8k.json",
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    episodes: int = 400,
    limit_samples: int | None = 5000,
) -> None:
    train_remote.remote(dataset, mapping, model_name, episodes, limit_samples)
