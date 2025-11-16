use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    fs::{self, File},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use pgn_reader::{RawTag, Reader, SanPlus, Skip, Visitor};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use pgn_reader::shakmaty::{
    fen::Fen,
    CastlingMode,
    Chess,
    EnPassantMode,
    Position,
    uci::UciMove,
};
use std::ops::ControlFlow;

#[derive(Debug, Parser)]
#[command(author, version, about = "Rust fast data ingester for ChessGPT")]
struct Args {
    #[arg(long)]
    puzzle_csv: Option<PathBuf>,
    #[arg(long)]
    pgn_file: Option<PathBuf>,
    #[arg(long)]
    pgn_dir: Option<PathBuf>,
    #[arg(long)]
    output_dataset: PathBuf,
    #[arg(long)]
    output_mapping: PathBuf,
    #[arg(long)]
    limit_puzzles: Option<usize>,
    #[arg(long)]
    limit_games: Option<usize>,
    #[arg(long, default_value_t = 8)]
    history_length: usize,
    #[arg(long)]
    exclude_history: bool,
    #[arg(long, default_value_t = num_cpus::get())]
    workers: usize,
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct Example {
    fen: String,
    mv: String,
    history: Vec<String>,
}

#[derive(Debug)]
struct PgnResult {
    examples: Vec<Example>,
    games: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let include_history = !args.exclude_history && args.history_length > 0;
    let cache_dir = args
        .cache_dir
        .clone()
        .unwrap_or_else(|| args.output_dataset.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."))
            .join("cache"));

    let mut combined: Vec<Example> = Vec::new();

    if let Some(csv_path) = &args.puzzle_csv {
        let cache_path = cache_dir.join(format!(
            "puzzles_{}_{}_limit{}.jsonl",
            csv_path.file_stem().and_then(|s| s.to_str()).unwrap_or("puzzles"),
            if include_history { "hist" } else { "nohist" },
            args.limit_puzzles.map(|v| v.to_string()).unwrap_or_else(|| "all".into()),
        ));
        if cache_path.exists() {
            println!("Loading puzzles from cache {:?}", cache_path);
            combined.extend(load_examples_cache(&cache_path)?);
        } else {
            println!("Parsing puzzles from {:?}", csv_path);
            let puzzles = parse_puzzles(csv_path, args.history_length, include_history, args.limit_puzzles)?;
            if !puzzles.is_empty() {
                write_examples_cache(&puzzles, &cache_path)?;
            }
            combined.extend(puzzles);
        }
    }

    if let Some(pgn_file) = &args.pgn_file {
        println!("Parsing PGN file {:?}", pgn_file);
        let result = parse_pgn_file(pgn_file, args.history_length, include_history, args.limit_games)?;
        combined.extend(result.examples);
        println!("Parsed {} games from file", result.games);
    }

    if let Some(pgn_dir) = &args.pgn_dir {
        let mut files = collect_pgn_files(pgn_dir)?;
        files.sort();
        if files.is_empty() {
            println!("No PGN files found in {:?}", pgn_dir);
        } else {
            let limit = args.limit_games;
            let use_parallel = limit.is_none() && args.workers > 1;
            if use_parallel {
                let pb = progress_bar_with_total(files.len() as u64, "PGN files");
                let history_len = args.history_length;
                let include = include_history;
                let results: Result<Vec<_>> = files
                    .par_iter()
                    .map(|file| {
                        let parsed = parse_pgn_file(file, history_len, include, None);
                        pb.inc(1);
                        parsed
                    })
                    .collect();
                pb.finish_and_clear();
                combined.extend(results?.into_iter().flat_map(|r| r.examples));
            } else {
                let mut remaining = limit.unwrap_or(usize::MAX);
                let pb = progress_bar_with_total(files.len() as u64, "PGN files");
                for file in files {
                    if remaining == 0 {
                        break;
                    }
                    let per_file_limit = if limit.is_some() { Some(remaining) } else { None };
                    let result = parse_pgn_file(&file, args.history_length, include_history, per_file_limit)?;
                    remaining = remaining.saturating_sub(result.games);
                    combined.extend(result.examples);
                    pb.inc(1);
                    if remaining == 0 {
                        break;
                    }
                }
                pb.finish_and_clear();
            }
        }
    }

    if combined.is_empty() {
        return Err(anyhow!("No data parsed. Provide at least one input source."));
    }

    println!("Building move mapping ...");
    let mut move_set: HashSet<String> = HashSet::with_capacity(combined.len() * (if include_history { 2 } else { 1 }));
    let pb_moves = progress_bar_with_total(combined.len() as u64, "Moves");
    for example in &combined {
        move_set.insert(example.mv.clone());
        if include_history {
            for h in &example.history {
                move_set.insert(h.clone());
            }
        }
        pb_moves.inc(1);
    }
    pb_moves.finish_and_clear();
    let mut moves: Vec<_> = move_set.into_iter().collect();
    moves.sort();
    let move_to_idx: BTreeMap<_, _> = moves.into_iter().enumerate().map(|(idx, mv)| (mv, idx)).collect();
    println!("Unique moves: {}", move_to_idx.len());

    println!("Writing dataset to {:?}", args.output_dataset);
    write_dataset(&combined, &move_to_idx, include_history, &args.output_dataset)?;
    println!("Writing move mapping to {:?}", args.output_mapping);
    write_mapping(&move_to_idx, &args.output_mapping)?;
    println!("Done.");
    Ok(())
}

fn parse_puzzles(
    path: &Path,
    history_length: usize,
    include_history: bool,
    limit: Option<usize>,
) -> Result<Vec<Example>> {
    let mut dataset = Vec::new();
    let total_rows = count_rows(path).unwrap_or(0);
    let pb = progress_bar_with_total(total_rows.max(1), "Puzzles");

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("Failed to open {:?}", path))?;

    for record in reader.records() {
        pb.inc(1);
        let record = match record {
            Ok(rec) => rec,
            Err(_) => continue,
        };
        let fen = record.get(1).unwrap_or("").trim();
        let moves_str = record.get(2).unwrap_or("").trim();
        if fen.is_empty() || moves_str.is_empty() {
            continue;
        }
        let moves: Vec<&str> = moves_str.split_whitespace().collect();
        if moves.len() < 2 {
            continue;
        }
        let position: Chess = match Fen::from_ascii(fen.as_bytes())
            .ok()
            .and_then(|fen| fen.into_position(CastlingMode::Standard).ok())
        {
            Some(pos) => pos,
            None => continue,
        };
        let mut pos = position;
        let blunder = match moves[0].parse::<UciMove>()
            .ok()
            .and_then(|uci| uci.to_move(&pos).ok())
        {
            Some(mv) => mv,
            None => continue,
        };
        pos.play_unchecked(blunder);
        let fen_after = Fen::from_position(&pos, EnPassantMode::Legal).to_string();
        let correct_move = moves[1].to_string();
        let history = if include_history && history_length > 0 {
            vec![moves[0].to_string()]
        } else {
            Vec::new()
        };
        dataset.push(Example {
            fen: fen_after,
            mv: correct_move,
            history,
        });
        if let Some(limit) = limit {
            if dataset.len() >= limit {
                break;
            }
        }
    }

    pb.finish_and_clear();
    Ok(dataset)
}

fn parse_pgn_file(
    path: &Path,
    history_length: usize,
    include_history: bool,
    limit_games: Option<usize>,
) -> Result<PgnResult> {
    let file = File::open(path).with_context(|| format!("Failed to open {:?}", path))?;
    let mut reader = Reader::new(BufReader::new(file));
    let mut visitor = ExampleVisitor::new(include_history, history_length, limit_games);
    while reader.read_game(&mut visitor)?.is_some() {}
    Ok(PgnResult {
        examples: visitor.examples,
        games: visitor.games,
    })
}

fn collect_pgn_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if dir.is_dir() {
        for entry in fs::read_dir(dir).with_context(|| format!("Failed to read {:?}", dir))? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                files.extend(collect_pgn_files(&path)?);
            } else if path.extension().map(|ext| ext.eq_ignore_ascii_case("pgn")).unwrap_or(false) {
                files.push(path);
            }
        }
    }
    Ok(files)
}

fn write_dataset(
    examples: &[Example],
    mapping: &BTreeMap<String, usize>,
    include_history: bool,
    output_path: &Path,
) -> Result<()> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let file = File::create(output_path).with_context(|| format!("Failed to create {:?}", output_path))?;
    let mut writer = std::io::BufWriter::new(file);
    let pb = progress_bar_with_total(examples.len() as u64, "Write DS");
    for example in examples {
        if let Some(&move_idx) = mapping.get(&example.mv) {
            let history_indices: Vec<usize> = if include_history {
                example
                    .history
                    .iter()
                    .filter_map(|h| mapping.get(h))
                    .copied()
                    .collect()
            } else {
                Vec::new()
            };
            let record = OutputRecord {
                fen: &example.fen,
                move_idx,
                history: if include_history {
                    Some(history_indices.as_slice())
                } else {
                    None
                },
            };
            serde_json::to_writer(&mut writer, &record)?;
            writer.write_all(b"\n")?;
        }
        pb.inc(1);
    }
    pb.finish_and_clear();
    writer.flush()?;
    Ok(())
}

fn write_mapping(mapping: &BTreeMap<String, usize>, output_path: &Path) -> Result<()> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let file = File::create(output_path).with_context(|| format!("Failed to create {:?}", output_path))?;
    serde_json::to_writer(std::io::BufWriter::new(file), mapping)?;
    Ok(())
}

fn load_examples_cache(path: &Path) -> Result<Vec<Example>> {
    let file = File::open(path).with_context(|| format!("Failed to open cache {:?}", path))?;
    let reader = BufReader::new(file);
    let mut dataset = Vec::new();
    for line in reader.lines().flatten() {
        if let Ok(record) = serde_json::from_str::<CacheRecord>(&line) {
            dataset.push(Example {
                fen: record.fen,
                mv: record.mv,
                history: record.history.unwrap_or_default(),
            });
        }
    }
    Ok(dataset)
}

fn write_examples_cache(examples: &[Example], path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let file = File::create(path).with_context(|| format!("Failed to create cache {:?}", path))?;
    let mut writer = std::io::BufWriter::new(file);
    for example in examples {
        let record = CacheRecord {
            fen: example.fen.clone(),
            mv: example.mv.clone(),
            history: if example.history.is_empty() {
                None
            } else {
                Some(example.history.clone())
            },
        };
        serde_json::to_writer(&mut writer, &record)?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

fn count_rows(path: &Path) -> Result<u64> {
    let file = File::open(path).with_context(|| format!("Failed to open {:?}", path))?;
    let reader = BufReader::new(file);
    Ok(reader.lines().count().saturating_sub(1) as u64)
}

fn progress_bar_with_total(total: u64, label: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            &format!("{{msg:<12}} {{bar:40.cyan/blue}} {{pos}}/{{len}} (ETA {{eta}})")
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb.set_message(label.to_string());
    pb
}

#[derive(Serialize)]
struct OutputRecord<'a> {
    fen: &'a str,
    move_idx: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    history: Option<&'a [usize]>,
}

#[derive(Serialize, Deserialize)]
struct CacheRecord {
    fen: String,
    #[serde(rename = "move")]
    mv: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    history: Option<Vec<String>>,
}

struct ExampleVisitor {
    include_history: bool,
    history_length: usize,
    examples: Vec<Example>,
    games: usize,
    remaining_games: Option<usize>,
}

struct MovetextState {
    pos: Chess,
    history: VecDeque<String>,
}

impl ExampleVisitor {
    fn new(include_history: bool, history_length: usize, limit: Option<usize>) -> Self {
        Self {
            include_history,
            history_length,
            examples: Vec::new(),
            games: 0,
            remaining_games: limit,
        }
    }
}

impl Visitor for ExampleVisitor {
    type Tags = Option<Chess>;
    type Movetext = MovetextState;
    type Output = ();

    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> {
        if let Some(remaining) = self.remaining_games {
            if remaining == 0 {
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(None)
    }

    fn tag(&mut self, tags: &mut Self::Tags, name: &[u8], value: RawTag<'_>) -> ControlFlow<Self::Output> {
        if name == b"FEN" {
            if let Ok(fen) = Fen::from_ascii(value.as_bytes()) {
                if let Ok(pos) = fen.into_position(CastlingMode::Standard) {
                    tags.replace(pos);
                }
            }
        }
        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, tags: Self::Tags) -> ControlFlow<Self::Output, Self::Movetext> {
        self.games += 1;
        if let Some(remaining) = self.remaining_games.as_mut() {
            if *remaining == 0 {
                return ControlFlow::Break(());
            }
            *remaining -= 1;
        }
        let history = VecDeque::with_capacity(self.history_length.max(1));
        ControlFlow::Continue(MovetextState {
            pos: tags.unwrap_or_default(),
            history,
        })
    }

    fn begin_variation(&mut self, _movetext: &mut Self::Movetext) -> ControlFlow<Self::Output, Skip> {
        ControlFlow::Continue(Skip(true))
    }

    fn san(&mut self, movetext: &mut Self::Movetext, san_plus: SanPlus) -> ControlFlow<Self::Output> {
        if let Ok(mv) = san_plus.san.to_move(&movetext.pos) {
            let fen_str = Fen::from_position(&movetext.pos, EnPassantMode::Legal).to_string();
            let uci = UciMove::from_standard(mv.clone()).to_string();
            let history = if self.include_history {
                movetext.history.iter().cloned().collect()
            } else {
                Vec::new()
            };
            self.examples.push(Example {
                fen: fen_str,
                mv: uci.clone(),
                history,
            });
            movetext.pos.play_unchecked(mv);
            if self.include_history && self.history_length > 0 {
                if movetext.history.len() >= self.history_length {
                    movetext.history.pop_front();
                }
                movetext.history.push_back(uci);
            }
        }
        ControlFlow::Continue(())
    }

    fn end_game(&mut self, _movetext: Self::Movetext) -> Self::Output {}
}
