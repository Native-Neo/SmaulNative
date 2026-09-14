use smaul_native::inference::Inference;
use smaul_native::rwkv_model::{RwkvModel, RwkvModelConfig};
use smaul_native::tokenizer::Tokenizer;
use std::env;
use std::path::PathBuf;

fn usage() -> ! {
    eprintln!("usage: smaul generate --model MODEL.safetensors --tokenizer TOKENIZER.json --embd N --layers N --head-size N --prompt TEXT [--max-new N] [--temperature F] [--top-k N] [--top-p F] [--repeat F] [--seed N]");
    std::process::exit(2);
}

fn value(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

fn required(args: &[String], name: &str) -> String { value(args, name).unwrap_or_else(|| { eprintln!("missing {name}"); usage() }) }

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.get(1).map(String::as_str) != Some("generate") { usage(); }
    let model_path = PathBuf::from(required(&args, "--model"));
    let tokenizer_path = PathBuf::from(required(&args, "--tokenizer"));
    let tokenizer = Tokenizer::from_json_file(tokenizer_path)?;
    let embd = required(&args, "--embd").parse::<usize>().map_err(|e| e.to_string())?;
    let layers = required(&args, "--layers").parse::<usize>().map_err(|e| e.to_string())?;
    let head_size = required(&args, "--head-size").parse::<usize>().map_err(|e| e.to_string())?;
    let prompt = required(&args, "--prompt");
    let mut model = RwkvModel::new(RwkvModelConfig::new(tokenizer.vocab_size(), embd, layers, head_size), 0);
    model.load_safetensors(model_path)?;
    let engine = Inference { model: &model, tokenizer: &tokenizer, eos_id: tokenizer.eos_id() };
    let max_new = value(&args, "--max-new").map(|v| v.parse()).transpose().map_err(|e| e.to_string())?.unwrap_or(256);
    let temperature = value(&args, "--temperature").map(|v| v.parse()).transpose().map_err(|e| e.to_string())?.unwrap_or(0.7);
    let top_k = value(&args, "--top-k").map(|v| v.parse()).transpose().map_err(|e| e.to_string())?.unwrap_or(50);
    let top_p = value(&args, "--top-p").map(|v| v.parse()).transpose().map_err(|e| e.to_string())?.unwrap_or(0.95);
    let repeat = value(&args, "--repeat").map(|v| v.parse()).transpose().map_err(|e| e.to_string())?.unwrap_or(1.05);
    let seed = value(&args, "--seed").map(|v| v.parse()).transpose().map_err(|e| e.to_string())?.unwrap_or(0);
    for chunk in engine.stream(&prompt, max_new, temperature, top_k, top_p, repeat, &[], seed) { print!("{chunk}"); }
    println!();
    Ok(())
}
