"""
Data ingestion script for preparing chess datasets.

This script consolidates multiple chess data sources into a unified format
for supervised training.  It processes

1. The Lichess puzzle CSV dataset.  Each row contains a FEN string and a
   sequence of moves in UCI notation.  According to the official
   documentation, the FEN represents the position **before** the blunder,
   the first move in the `Moves` column is the blunder, and the second
   move is the first move of the solution【901679508138493†L94-L102】.  For
   training, we apply the blunder move to the FEN and treat the
   following move as the correct answer.

2. PGN game files (either a single file or an entire directory tree).  For
   each game we iterate through the mainline moves and record the board
   position before each move along with the move itself.  This yields
   (FEN, move) pairs representing moves played in strong games.

After aggregating the data, the script builds a move-to-index mapping
over all unique moves and produces two output files:

* A JSON lines file where each line contains a dictionary with two keys:
  ``fen`` (the board position after any preliminary moves) and
  ``move_idx`` (the integer index of the correct move).

* A JSON file containing the ``move_to_idx`` mapping so that the same
  ordering can be reused during training and inference.

Usage example (from the command line):

```
python data_ingestion.py \
    --puzzle_csv data/lichess_db_puzzle.csv \
    --pgn_dir data/lichess_elite_games \
    --output_dataset data/train_dataset.jsonl \
    --output_mapping data/move_mapping.json \
    --limit_puzzles 500000 \
    --limit_games 200000
```

This will parse up to 500k puzzle examples and 200k game examples,
construct a move mapping, and save the results into the specified files.
Adjust the limits or omit them entirely to process all available data.

Dependencies:
    - python-chess
    - tqdm (optional, for progress bars)

"""

import argparse
import csv
import json
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import chain
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple
from tqdm import tqdm

try:
    import chess
    import chess.pgn
except ImportError as exc:
    raise ImportError(
        "The python-chess library is required for this script. Install it via `pip install chess`."
    ) from exc

def parse_puzzles(
    csv_file: str,
    history_length: int,
    include_history: bool,
    limit: Optional[int] = None,
    show_progress: bool = True,
) -> List[Tuple[str, str, List[str]]]:
    """Parse the Lichess puzzles CSV file.

    Each row of the CSV has the following columns (based on the
    official Lichess documentation【364005705419582†L105-L135】):

        PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags

    ``FEN`` is the position before the opponent's mistake.  The
    ``Moves`` column contains the blunder followed by the correct
    sequence to solve the puzzle【901679508138493†L94-L102】.  To obtain a
    training example we apply the blunder (first move) to the FEN and
    then treat the next move as the correct answer.

    Args:
        csv_file: Path to the CSV file.
        limit: If provided, stop after this many examples.

    Returns:
        A list of (fen_after, correct_move, history_moves) tuples.
    """
    dataset: List[Tuple[str, str, List[str]]] = []
    # estimate total rows for better ETA (skip header)
    try:
        with open(csv_file, 'r', encoding='utf-8') as counter:
            total_rows = sum(1 for _ in counter) - 1
    except OSError:
        total_rows = None

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        iterator = reader
        progress = None
        if show_progress:
            if total_rows and total_rows > 0:
                progress = tqdm(reader, desc="Puzzles", unit="rows", total=total_rows, dynamic_ncols=True)
            else:
                progress = tqdm(reader, desc="Puzzles", unit="rows", dynamic_ncols=True)
            iterator = progress
        for row in iterator:
            if not row:
                continue
            fen = (row.get("FEN") or "").strip()
            moves_str = (row.get("Moves") or "").strip()
            if not fen or not moves_str:
                continue
            moves = moves_str.split()
            # we need at least two moves: the blunder and the solution
            if len(moves) < 2:
                continue
            try:
                board = chess.Board(fen)
            except Exception:
                # skip malformed FENs
                continue
            try:
                blunder = chess.Move.from_uci(moves[0])
                board.push(blunder)
            except Exception:
                # skip if we cannot apply the blunder
                continue
            correct_move = moves[1]
            history: List[str] = []
            if include_history and history_length > 0:
                history = [blunder.uci()][-history_length:]
            dataset.append((board.fen(), correct_move, history))
            if limit is not None and len(dataset) >= limit:
                break
        if progress is not None:
            progress.close()
    return dataset

def parse_pgn_file(
    pgn_path: str,
    history_length: int,
    include_history: bool,
    limit_games: Optional[int] = None,
    show_progress: bool = True,
) -> Tuple[List[Tuple[str, str, List[str]]], int]:
    """Parse a single PGN file and extract (FEN, move) pairs.

    Args:
        pgn_path: Path to the PGN file.
        limit_games: If provided, stop after this many games.

    Returns:
        A tuple of (examples, games_parsed).
    """
    dataset: List[Tuple[str, str, List[str]]] = []
    game_count = 0
    total = limit_games
    with open(pgn_path, encoding='utf-8', errors='ignore') as f:
        pbar = None
        if show_progress:
            pbar = tqdm(
                desc=f"Games {os.path.basename(pgn_path)}",
                unit="games",
                total=total,
                dynamic_ncols=True,
            )
        while True:
            try:
                game = chess.pgn.read_game(f)
            except Exception:
                # sometimes PGN files have malformed games; continue
                continue
            if game is None:
                break
            game_count += 1
            if pbar is not None:
                pbar.update(1)
            board = game.board()
            history_buffer: Optional[Deque[str]] = None
            if include_history and history_length > 0:
                history_buffer = deque(maxlen=history_length)
            for move in game.mainline_moves():
                history_snapshot = list(history_buffer) if history_buffer is not None else []
                dataset.append((board.fen(), move.uci(), history_snapshot))
                board.push(move)
                if history_buffer is not None:
                    history_buffer.append(move.uci())
            if limit_games is not None and game_count >= limit_games:
                break
        if pbar is not None:
            pbar.close()
    return dataset, game_count

def parse_pgn_directory(
    directory: str,
    history_length: int,
    include_history: bool,
    limit_games: Optional[int] = None,
    workers: int = 1,
    executor: Optional[ProcessPoolExecutor] = None,
) -> List[Tuple[str, str, List[str]]]:
    """Parse all PGN files within a directory tree.

    Args:
        directory: Directory containing PGN files (recursively).
        limit_games: If provided, stop after this many games in total.

    Returns:
        A list of (fen_before_move, move_uci, history_moves) tuples.
    """
    dataset: List[Tuple[str, str, List[str]]] = []
    total_games = 0
    pgn_files = [
        os.path.join(root, filename)
        for root, _, files in os.walk(directory)
        for filename in files
        if filename.lower().endswith('.pgn')
    ]
    if workers <= 1 or executor is None or len(pgn_files) <= 1:
        progress = tqdm(
            pgn_files,
            desc="PGN files",
            unit="file",
            total=len(pgn_files),
            dynamic_ncols=True,
        )
        for file_path in progress:
            remaining = None
            if limit_games is not None:
                remaining = max(0, limit_games - total_games)
                if remaining <= 0:
                    break
            examples, parsed_games = parse_pgn_file(
                file_path,
                history_length=history_length,
                include_history=include_history,
                limit_games=remaining,
            )
            dataset.extend(examples)
            if limit_games is not None:
                total_games += parsed_games
                if total_games >= limit_games:
                    break
        progress.close()
        return dataset

    print(f"Parsing PGN files in parallel with {workers} workers ...")
    progress = tqdm(
        desc="PGN files",
        unit="file",
        total=len(pgn_files),
        dynamic_ncols=True,
    )
    futures: set = set()
    file_iter = iter(pgn_files)

    def submit_next() -> bool:
        try:
            file_path = next(file_iter)
        except StopIteration:
            return False
        remaining = None
        if limit_games is not None:
            remaining = max(0, limit_games - total_games)
            if remaining <= 0:
                return False
        future = executor.submit(
            parse_pgn_file,
            file_path,
            history_length,
            include_history,
            remaining,
            False,
        )
        future.file_path = file_path  # type: ignore[attr-defined]
        futures.add(future)
        return True

    for _ in range(min(workers, len(pgn_files))):
        if not submit_next():
            break

    while futures:
        future = next(as_completed(futures))
        futures.remove(future)
        file_path = getattr(future, "file_path", None)
        examples, parsed_games = future.result()
        dataset.extend(examples)
        total_games += parsed_games
        print(f"Parsed {parsed_games} games from {os.path.basename(file_path or '')}")
        if limit_games is not None and total_games >= limit_games:
            print("Reached limit; stopping additional PGN parsing")
            break
        submit_next()

        progress.update(1)

    progress.close()
    return dataset

def build_move_mapping(moves: Iterable[str]) -> Dict[str, int]:
    """Assign a unique integer index to each move string.

    Args:
        moves: An iterable of move strings (in UCI notation).

    Returns:
        A dictionary mapping each unique move to an integer index.
    """
    unique_moves = sorted(set(moves))
    move_to_idx: Dict[str, int] = {move: idx for idx, move in enumerate(unique_moves)}
    return move_to_idx

def save_dataset(
    dataset: List[Tuple[str, int, List[int]]], output_path: str, include_history: bool
) -> None:
    """Save the processed dataset to a JSON lines file.

    Each line of the output will be a JSON object containing a FEN
    string and the integer index of the correct move.  Using JSON
    lines makes the file easy to stream from disk during training.

    Args:
        dataset: List of (fen, move_idx) tuples.
        output_path: Path to write the JSON lines file.
    """
    with open(output_path, 'w', encoding='utf-8') as out:
        for fen, move_idx, history in dataset:
            record = {
                "fen": fen,
                "move_idx": move_idx,
            }
            if include_history:
                record["history"] = history
            out.write(json.dumps(record) + "\n")


def write_examples_cache(examples: List[Tuple[str, str, List[str]]], cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as cache_file:
        for fen, move, history in examples:
            cache_file.write(json.dumps({"fen": fen, "move": move, "history": history}) + "\n")


def load_examples_cache(cache_path: str) -> List[Tuple[str, str, List[str]]]:
    dataset: List[Tuple[str, str, List[str]]] = []
    with open(cache_path, 'r', encoding='utf-8') as cache_file:
        for line in cache_file:
            record = json.loads(line)
            dataset.append((record["fen"], record["move"], record.get("history", [])))
    return dataset

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare chess datasets for training")
    parser.add_argument('--puzzle_csv', type=str, default=None, help='Path to the Lichess puzzle CSV file')
    parser.add_argument('--pgn_file', type=str, default=None, help='Path to a PGN file to parse')
    parser.add_argument('--pgn_dir', type=str, default=None, help='Path to a directory containing PGN files')
    parser.add_argument('--output_dataset', type=str, required=True, help='Output path for the processed dataset (JSON lines)')
    parser.add_argument('--output_mapping', type=str, required=True, help='Output path for the move-to-index mapping (JSON)')
    parser.add_argument('--limit_puzzles', type=int, default=None, help='Maximum number of puzzle examples to parse')
    parser.add_argument('--limit_games', type=int, default=None, help='Maximum number of games to parse from PGN files')
    parser.add_argument('--history_length', type=int, default=8, help='Number of previous moves to store for context')
    parser.add_argument('--exclude_history', action='store_true', help='Disable history capture and only store board positions')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1, help='Number of parallel workers for PGN parsing')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory used to cache parsed puzzle datasets')
    args = parser.parse_args()

    include_history = (not args.exclude_history) and args.history_length > 0
    cache_dir = args.cache_dir or os.path.join(os.path.dirname(args.output_dataset), "cache")
    workers = max(1, args.workers or 1)

    combined: List[Tuple[str, str, List[str]]] = []
    executor: Optional[ProcessPoolExecutor] = None
    puzzle_future = None
    puzzle_cache_path: Optional[str] = None
    try:
        if workers > 1:
            executor = ProcessPoolExecutor(max_workers=workers)
        if args.puzzle_csv:
            cache_suffix = "hist" if include_history else "nohist"
            limit_tag = args.limit_puzzles if args.limit_puzzles is not None else "all"
            csv_tag = Path(args.puzzle_csv).stem
            puzzle_cache_path = os.path.join(
                cache_dir,
                f"puzzles_{csv_tag}_{cache_suffix}_limit{limit_tag}.jsonl",
            )
            if puzzle_cache_path and os.path.exists(puzzle_cache_path):
                print(f"Loading puzzles from cache {puzzle_cache_path}")
                puzzle_examples = load_examples_cache(puzzle_cache_path)
                combined.extend(puzzle_examples)
            else:
                print(f"Parsing puzzles from {args.puzzle_csv} ...")
                if executor and workers > 1:
                    puzzle_future = executor.submit(
                        parse_puzzles,
                        args.puzzle_csv,
                        args.history_length,
                        include_history,
                        args.limit_puzzles,
                        False,
                    )
                else:
                    puzzle_examples = parse_puzzles(
                        args.puzzle_csv,
                        args.history_length,
                        include_history=include_history,
                        limit=args.limit_puzzles,
                    )
                    combined.extend(puzzle_examples)
                    if puzzle_cache_path:
                        write_examples_cache(puzzle_examples, puzzle_cache_path)
                    print(f"Parsed {len(puzzle_examples)} puzzle examples")
        if args.pgn_file:
            print(f"Parsing PGN file {args.pgn_file} ...")
            pgn_examples, parsed_games = parse_pgn_file(
                args.pgn_file,
                history_length=args.history_length,
                include_history=include_history,
                limit_games=args.limit_games,
            )
            combined.extend(pgn_examples)
            print(f"Parsed {len(pgn_examples)} examples from PGN file ({parsed_games} games)")
        if args.pgn_dir:
            print(f"Parsing PGN directory {args.pgn_dir} ...")
            pgn_dir_examples = parse_pgn_directory(
                args.pgn_dir,
                history_length=args.history_length,
                include_history=include_history,
                limit_games=args.limit_games,
                workers=workers,
                executor=executor,
            )
            combined.extend(pgn_dir_examples)
            print(f"Parsed {len(pgn_dir_examples)} examples from PGN directory")

        if puzzle_future:
            puzzle_examples = puzzle_future.result()
            combined.extend(puzzle_examples)
            if puzzle_cache_path:
                write_examples_cache(puzzle_examples, puzzle_cache_path)
            print(f"Parsed {len(puzzle_examples)} puzzle examples")
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=False)

    if not combined:
        raise ValueError("No data was parsed. Specify at least one input source.")

    # build mapping
    moves: List[str] = [move for _, move, _ in combined]
    if include_history:
        moves.extend(chain.from_iterable(history for _, _, history in combined))
    move_to_idx = build_move_mapping(moves)
    print(f"Unique moves: {len(move_to_idx)}")

    # convert dataset to index form
    indexed_dataset: List[Tuple[str, int, List[int]]] = []
    for fen, move, history in combined:
        history_indices = [move_to_idx[h] for h in history if h in move_to_idx] if include_history else []
        indexed_dataset.append((fen, move_to_idx[move], history_indices))
    # save dataset and mapping
    print(f"Saving dataset to {args.output_dataset} ...")
    save_dataset(indexed_dataset, args.output_dataset, include_history=include_history)
    print(f"Saving move mapping to {args.output_mapping} ...")
    with open(args.output_mapping, 'w', encoding='utf-8') as f:
        json.dump(move_to_idx, f)
    print("Data preparation completed.")

if __name__ == '__main__':
    main()
