import os
import shutil
import subprocess
from pathlib import Path

import modal


IMAGE = (
    modal.Image.debian_slim()
    .apt_install("curl", "pkg-config", "libssl-dev", "build-essential")
    .run_commands("curl https://sh.rustup.rs -sSf | sh -s -- -y")
    .env({"PATH": "/root/.cargo/bin:$PATH"})
)

DATA_VOLUME = modal.Volume.from_name("chessgpt-datasets", create_if_missing=True)
CODE_VOLUME = modal.Volume.from_name("chessgpt-rust-ingester", create_if_missing=True)

app = modal.App("rust-ingester")


@app.function(
    image=IMAGE,
    cpu=16,
    timeout=60 * 60 * 12,
    volumes={"/data": DATA_VOLUME, "/src": CODE_VOLUME},
)
def run_ingest(
    puzzle_csv: str = "/data/lichess_puzzle_transformed.csv",
    pgn_dir: str = "/data/Lichess Elite Database",
    output_dataset: str = "/data/train_dataset_hist0_full.jsonl",
    output_mapping: str = "/data/move_mapping_hist0_full.json",
    history_length: int = 0,
    exclude_history: bool = True,
    workers: int = 16,
    cache_dir: str = "/data/cache",
):
    """Run the Rust ingester inside Modal."""
    source_dir = Path("/src/rust-ingester")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} not found. Did you upload to the chessgpt-rust-ingester volume?")
    work_dir = Path("/tmp/rust-ingester")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(source_dir, work_dir)
    os.chdir(work_dir)
    args = [
        "cargo",
        "run",
        "--release",
        "--",
        "--puzzle-csv",
        puzzle_csv,
        "--pgn-dir",
        pgn_dir,
        "--output-dataset",
        output_dataset,
        "--output-mapping",
        output_mapping,
        "--history-length",
        str(history_length),
        "--workers",
        str(workers),
        "--cache-dir",
        cache_dir,
    ]
    if exclude_history:
        args.append("--exclude-history")
    print("Running:", " ".join(args))
    subprocess.run(args, check=True)
    DATA_VOLUME.commit()


@app.local_entrypoint()
def main():
    run_ingest.remote()
