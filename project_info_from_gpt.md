# Objectives
This project aims to build a powerful transformer‑based chess engine that can play complete games
autonomously without search. We will enter this model into the 36‑hour ChessHacks hackathon. Although
the event includes a prize category for models under 10 MB, the primary goal is to maximise playing
strength. Where possible we will strive for a compact model – squeezing knowledge into a tiny neural network is both challenging and rewarding – but if a slightly larger architecture delivers significantly better
chess performance, scaling up is acceptable. The core idea remains to distil high‑quality chess knowledge
by training on engine‑labelled positions and curated games so that the model can evaluate positions “at a
glance” and select strong moves directly

# Data Sources
To teach the model how to play chess, we consolidate several publicly available datasets. The first tier of
data (zero‑engine cost) includes:
1.
Lichess puzzle database – a CSV file containing millions of tactical puzzles. Each row includes a FEN
string and a list of moves in UCI notation. According to the official Lichess documentation, the FEN is
the position before an opponent’s blunder. The first move in the Moves column is the blunder, and
the second move is the start of the solution 
1. For training we apply the blunder to the FEN and
treat the next move as the correct answer. The dataset fields are: PuzzleId , FEN , Moves ,
Rating, RatingDeviation, Popularity, NbPlays, Themes , GameUrl and
OpeningTags 2
. The moves are in UCI format and the solution moves are “only moves” (i.e. any
3
other move would significantly worsen the position) .
2. Lichess elite games – PGN files of games played by titled or high‑rated players. These provide
strong human positional play and opening knowledge.
3. CCRL engine games (optional) – PGN files containing games between top chess engines at long
time controls, providing high‑quality engine moves without running Stockfish ourselves.
We plan to augment this data with our own Stockfish self‑play positions later on for fine‑tuning, but initial
pretraining will use the Tier 1 datasets to maximise coverage quickly.


What has been implemented
Data ingestion script ( data_ingestion.py )
•
•
•
•
•
•
•
Parses the Lichess puzzle CSV, applying the blunder (first move) to the FEN and using the
subsequent move as the correct answer. This follows the official guidance that the position shown to
1
the player is after applying the first move and the solution begins with the second move .
Parses PGN files or entire directories of PGN files, extracting the FEN before each move and the
move itself.
Aggregates examples from puzzles and games into a single list of (FEN, move) pairs.
Builds a move‑to‑index mapping over all unique moves encountered.
Writes two output files:
A JSONL dataset where each line contains a fen and a move_idx referring to the index in the
move mapping.
A JSON file containing the move_to_idx mapping, so the same ordering can be reused during
training and inference.
The script is invoked from the command line. For example:
python data_ingestion.py \
--puzzle_csv data/lichess_db_puzzle.csv \
--pgn_dir data/lichess_elite_games \
--output_dataset data/train_dataset.jsonl \
--output_mapping data/move_mapping.json \
--limit_puzzles 500000 \
--limit_games 200000
Refer to the source code for details: .
Training script ( train_model.py )
•
•
•
•
•
Example usage:
Loads the processed JSONL dataset and the move mapping.
Encodes each board position into a sequence of 65 tokens: 64 tokens for the squares (piece IDs) and
one token for the side to move. Piece IDs are integers: 0 = empty, 1–6 = white pawn/knight/bishop/
rook/queen/king, 7–12 = black pawn/knight/bishop/rook/queen/king.
Defines a lightweight transformer encoder. Each token is embedded via an embedding layer and
summed with a positional embedding. The transformer stack produces contextualised
representations that are pooled (mean pooling) and passed to a final linear layer that outputs logits
over the move classes.
Uses cross‑entropy loss and the AdamW optimizer to train the network. Hyperparameters such as
embedding dimension, number of layers, number of attention heads, feed‑forward dimension,
batch size and learning rate can be configured via command‑line arguments.
Saves the trained model weights to the specified path. The script currently does not implement
validation or learning‑rate schedulers but is a solid starting point for experimentation.
2
python train_model.py \
--dataset data/train_dataset.jsonl \
--move_mapping data/move_mapping.json \
--epochs 5 \
--batch_size 256 \
--embed_dim 128 \
--nhead 8 \
--num_layers 4 \
--ff_dim 512 \
--lr 1e-3 \
--model_path models/chess_policy.pt
See the source code for implementation details: .
Remaining tasks and next steps
1.
2.
3.
4.
5.
Dataset preparation: run the ingestion script on the downloaded Tier 1 datasets to generate a
combined dataset and move mapping. The --limit_puzzles and --limit_games arguments
can be adjusted to fit within memory constraints on an M2 Pro MacBook.
Model training: train the transformer on the processed dataset using the training script. Start with a
small number of epochs and observe the loss curve. Because the dataset may be large, consider
training on a subset initially and then progressively adding more data.
Validation and evaluation: implement a validation split or cross‑validation to monitor overfitting.
Evaluate the trained model by playing games against a baseline engine or the provided hackathon
bot. Monitor for illegal moves and integrate move legality masking at inference time.
Model compression: if we decide to target the sub‑10 MB prize category, experiment with weight
quantisation (e.g. 16‑bit or 8‑bit) or reducing the model size (fewer layers, smaller embedding
dimension) to shrink the footprint. If a larger model yields substantially better results, we can scale
up accordingly; the size limit is a goal rather than a hard requirement.
Self‑play fine‑tuning (optional): once the model is reasonably strong, generate additional
high‑quality examples using Stockfish self‑play. Fine‑tune the model on these positions to improve
accuracy in seldom‑seen situations. This step should be weighed against the 36‑hour time
constraint.
Conclusion
This repository lays the groundwork for training a compact transformer‑based chess engine. It consolidates
high‑quality puzzle and game data into a unified format, builds a comprehensive move mapping, and
provides a flexible training script. Future work will involve training the model, validating its performance,
and iterating on architecture and data to maximise playing strength while remaining mindful of model size
– striking a balance between compression and chess performance to suit the goals of the ChessHacks
competition.
3
1
Am I missing something in the puzzles database, or are solutions just not included? • page 1/1 • Lichess
Feedback • lichess.org
https://lichess.org/forum/lichess-feedback/am-i-missing-something-in-the-puzzles-database-or-are-solutions-just-not-included
2 3
lichess.org open database
https://database.lichess.org/