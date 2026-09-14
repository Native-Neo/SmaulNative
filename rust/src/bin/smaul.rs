use smaul_native::config::RwkvXConfig;
use smaul_native::inference::Inference;
use smaul_native::rwkv_model::RwkvModel;
use smaul_native::tokenizer::Tokenizer;
use std::env;
use std::path::PathBuf;

fn usage() -> ! {
    eprintln!("usage: smaul generate --model-dir DIR --prompt TEXT [--max-new N] [--temperature F] [--top-k N] [--top-p F] [--repeat F] [--seed N]");
    std::process::exit(2);
}

fn value(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

fn required(args: &[String], name: &str) -> String {
    value(args, name).unwrap_or_else(|| {
        eprintln!("missing {name}");
        usage()
    })
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.get(1).map(String::as_str) != Some("generate") {
        usage();
    }

    let dir = PathBuf::from(required(&args, "--model-dir"));
    let config = RwkvXConfig::load(dir.join("config.json"))?;
    let tokenizer = Tokenizer::from_json_file(dir.join("tokenizer.json"))?;

    if tokenizer.vocab_size() != config.vocab_size {
        return Err(format!(
            "tokenizer vocab size {} does not match config {}",
            tokenizer.vocab_size(), config.vocab_size
        ));
    }

    let mut model = RwkvModel::new(config.to_model_config(), 0);
    model.load_safetensors(dir.join("model.safetensors"))?;

    let engine = Inference {
        model: &model,
        tokenizer: &tokenizer,
        eos_id: tokenizer.eos_id(),
    };

    let prompt = required(&args, "--prompt");
    let max_new = value(&args, "--max-new")
        .map(|v| v.parse())
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or(256);
    let temperature = value(&args, "--temperature")
        .map(|v| v.parse())
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or(0.7);
    let top_k = value(&args, "--top-k")
        .map(|v| v.parse())
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or(50);
    let top_p = value(&args, "--top-p")
        .map(|v| v.parse())
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or(0.95);
    let repeat = value(&args, "--repeat")
        .map(|v| v.parse())
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or(1.05);
    let seed = value(&args, "--seed")
        .map(|v| v.parse())
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or(0);

    for chunk in engine.stream(
        &prompt,
        max_new,
        temperature,
        top_k,
        top_p,
        repeat,
        &[],
        seed,
    ) {
        print!("{chunk}");
    }
    println!();
    Ok(())
}
