use smaul_native::tokenizer_trainer::WordTokenizerTrainer;
use std::env;
use std::fs;
use std::path::PathBuf;

fn value(args: &[String], name: &str, default: &str) -> String {
    args.windows(2).find(|pair| pair[0] == name).map(|pair| pair[1].clone()).unwrap_or_else(|| default.into())
}

fn number<T: std::str::FromStr>(args: &[String], name: &str, default: &str) -> Result<T, String> {
    let raw = value(args, name, default);
    raw.parse().map_err(|_| format!("{name} expects a number, got '{raw}'"))
}

fn has(args: &[String], name: &str) -> bool { args.iter().any(|arg| arg == name) }

fn print_help() {
    println!("SmaulNative whole-word tokenizer trainer");
    println!();
    println!("Usage:");
    println!("  smaul-native tokenizer-train --dataset PATH [options]");
    println!();
    println!("Options:");
    println!("  --dataset PATH       Dataset file or directory");
    println!("  --vocab-size N       Final vocabulary size (default: 65536)");
    println!("  --output PATH        Tokenizer JSON path (default: ./SmaulNative/tokenizer.json)");
    println!("  --max-records N      Stop after N text records");
    println!("  --help               Show this help");
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if has(&args, "--help") || has(&args, "-h") { print_help(); return Ok(()); }
    let dataset = value(&args, "--dataset", "./datasets");
    let vocab_size: usize = number(&args, "--vocab-size", "65536")?;
    let output: PathBuf = value(&args, "--output", "./SmaulNative/tokenizer.json").into();
    let max_records = if args.iter().any(|arg| arg == "--max-records") { Some(number(&args, "--max-records", "0")?) } else { None };
    if vocab_size < 8 { return Err("--vocab-size must be at least 8".into()); }
    println!("[TOKENIZER] scanning {}", dataset);
    let tokenizer = WordTokenizerTrainer::train_from_path(&dataset, vocab_size, max_records)?;
    if let Some(parent) = output.parent() { fs::create_dir_all(parent).map_err(|e| format!("failed to create {}: {e}", parent.display()))?; }
    tokenizer.save_json(&output)?;
    println!("[TOKENIZER] trained whole-word vocabulary: {} tokens", tokenizer.vocab_size());
    println!("[TOKENIZER] output: {}", output.display());
    Ok(())
}

fn main() {
    if let Err(error) = run() { eprintln!("[TOKENIZER ERROR] {error}"); std::process::exit(1); }
}
