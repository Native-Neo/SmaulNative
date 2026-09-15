use smaul_native::inference::Inference;
use smaul_native::model_io::PretrainedModel;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

fn main() {
    if let Err(error) = run() {
        eprintln!("[WORKFLOW ERROR] {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut model_dir = "./SmaulNative".to_owned();
    let mut max_tokens = 256usize;
    let mut temperature = 0.7f32;
    let mut top_k = 50usize;
    let mut top_p = 0.95f32;
    let mut repeat_penalty = 1.05f32;
    let mut seed = 42u64;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_dir = args.next().ok_or("--model requires a value")?,
            "--device" => {
                let value = args.next().ok_or("--device requires a value")?;
                if !matches!(value.as_str(), "auto" | "cpu" | "cuda") {
                    return Err("--device must be auto, cpu, or cuda".into());
                }
            }
            "--dtype" => {
                let value = args.next().ok_or("--dtype requires a value")?;
                if !matches!(value.as_str(), "auto" | "fp32" | "fp16" | "bf16") {
                    return Err("--dtype must be auto, fp32, fp16, or bf16".into());
                }
            }
            "--max-tokens" => {
                max_tokens = args
                    .next()
                    .ok_or("--max-tokens requires a value")?
                    .parse()
                    .map_err(|_| "invalid --max-tokens")?;
            }
            "--temperature" => {
                temperature = args
                    .next()
                    .ok_or("--temperature requires a value")?
                    .parse()
                    .map_err(|_| "invalid --temperature")?;
            }
            "--top-k" => {
                top_k = args
                    .next()
                    .ok_or("--top-k requires a value")?
                    .parse()
                    .map_err(|_| "invalid --top-k")?;
            }
            "--top-p" => {
                top_p = args
                    .next()
                    .ok_or("--top-p requires a value")?
                    .parse()
                    .map_err(|_| "invalid --top-p")?;
            }
            "--repeat-penalty" => {
                repeat_penalty = args
                    .next()
                    .ok_or("--repeat-penalty requires a value")?
                    .parse()
                    .map_err(|_| "invalid --repeat-penalty")?;
            }
            "--seed" => {
                seed = args
                    .next()
                    .ok_or("--seed requires a value")?
                    .parse()
                    .map_err(|_| "invalid --seed")?;
            }
            "--help" | "-h" => {
                println!(
                    "Usage: smaul-infer --model DIR [--device auto|cpu|cuda] [--dtype auto|fp32|fp16|bf16] [--max-tokens N] [--temperature F] [--top-k N] [--top-p F] [--repeat-penalty F] [--seed N]"
                );
                return Ok(());
            }
            other if other.starts_with("--") => return Err(format!("unknown argument '{other}'")),
            other if model_dir == "./SmaulNative" => model_dir = other.to_owned(),
            other => return Err(format!("unexpected argument '{other}'")),
        }
    }

    if !Path::new(&model_dir).exists() {
        return Err(format!(
            "model path does not exist: {model_dir}\nPass a valid model directory with --model DIR."
        ));
    }

    let loaded = PretrainedModel::load(&model_dir)
        .map_err(|e| format!("failed to load model from {model_dir}: {e}"))?;
    let engine = Inference {
        model: &loaded.model,
        tokenizer: &loaded.tokenizer,
        eos_id: loaded.tokenizer.eos_id(),
    };

    println!(
        "SmaulNative RWKV-X | {} vocab | {} layers",
        loaded.config.vocab_size, loaded.config.n_layer
    );
    println!("Commands: /clear, /system <text>, /exit");

    let mut messages: Vec<(String, String)> = Vec::new();
    let mut system =
        "You are a helpful local AI assistant. Be concise, accurate, and practical.".to_owned();

    loop {
        print!("\nYou > ");
        io::stdout()
            .flush()
            .map_err(|e| format!("failed to flush prompt: {e}"))?;

        let mut user = String::new();
        if io::stdin()
            .read_line(&mut user)
            .map_err(|e| format!("failed to read input: {e}"))?
            == 0
        {
            break;
        }

        let user = user.trim().to_owned();
        if user.is_empty() {
            continue;
        }
        if user == "/exit" {
            break;
        }
        if user == "/clear" {
            messages.clear();
            println!("[conversation cleared]");
            continue;
        }
        if let Some(value) = user.strip_prefix("/system ") {
            system = value.trim().to_owned();
            println!("[system prompt updated]");
            continue;
        }

        messages.push(("user".into(), user));
        let refs = messages
            .iter()
            .map(|(role, content)| (role.as_str(), content.as_str()))
            .collect::<Vec<_>>();
        let prompt = engine.chat_prompt(&refs, Some(&system));

        print!("\nAssistant > ");
        io::stdout()
            .flush()
            .map_err(|e| format!("failed to flush response: {e}"))?;

        let started = Instant::now();
        let mut answer = String::new();
        for chunk in engine.stream(
            &prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            repeat_penalty,
            &[],
            seed,
        ) {
            print!("{chunk}");
            io::stdout()
                .flush()
                .map_err(|e| format!("failed to flush response: {e}"))?;
            answer.push_str(&chunk);
        }

        let elapsed = started.elapsed().as_secs_f64();
        let tokens = loaded.tokenizer.encode(&answer).len();
        println!(
            "\n[{tokens} tokens | {elapsed:.2}s | {:.2} tok/s]",
            tokens as f64 / elapsed.max(1e-6)
        );
        messages.push(("assistant".into(), answer));
        seed = seed.wrapping_add(1);
    }

    Ok(())
}
