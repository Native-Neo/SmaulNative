use smaul_native::inference::Inference;
use smaul_native::model_io::PretrainedModel;
use std::io::{self, Write};
use std::time::Instant;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model_dir = args.next().unwrap_or_else(|| "./SmaulNative".into());
    let mut max_tokens = 256usize;
    let mut temperature = 0.7f32;
    let mut top_k = 50usize;
    let mut top_p = 0.95f32;
    let mut repeat_penalty = 1.05f32;
    let mut seed = 42u64;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--max-tokens" => max_tokens = parse_arg(&mut args, "--max-tokens")?,
            "--temperature" => temperature = parse_arg(&mut args, "--temperature")?,
            "--top-k" => top_k = parse_arg(&mut args, "--top-k")?,
            "--top-p" => top_p = parse_arg(&mut args, "--top-p")?,
            "--repeat-penalty" => repeat_penalty = parse_arg(&mut args, "--repeat-penalty")?,
            "--seed" => seed = parse_arg(&mut args, "--seed")?,
            "--help" | "-h" => {
                println!("Usage: smaul-infer [MODEL_DIR] [--max-tokens N] [--temperature F] [--top-k N] [--top-p F] [--repeat-penalty F] [--seed N]");
                return Ok(());
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
    }

    let loaded = PretrainedModel::load(&model_dir)?;
    let engine = Inference { model: &loaded.model, tokenizer: &loaded.tokenizer, eos_id: loaded.tokenizer.eos_id() };
    println!("SmaulNative RWKV-X | {} vocab | {} layers", loaded.config.vocab_size, loaded.config.n_layer);
    println!("Commands: /clear, /system <text>, /exit");
    let mut messages: Vec<(String, String)> = Vec::new();
    let mut system = "You are a helpful local AI assistant. Be concise, accurate, and practical.".to_owned();

    loop {
        print!("\nYou > ");
        io::stdout().flush().map_err(|e| e.to_string())?;
        let mut user = String::new();
        if io::stdin().read_line(&mut user).map_err(|e| e.to_string())? == 0 { break; }
        let user = user.trim().to_owned();
        if user.is_empty() { continue; }
        if user == "/exit" { break; }
        if user == "/clear" { messages.clear(); println!("[conversation cleared]"); continue; }
        if let Some(value) = user.strip_prefix("/system ") {
            system = value.trim().to_owned();
            println!("[system prompt updated]");
            continue;
        }

        messages.push(("user".into(), user));
        let refs = messages.iter().map(|(role, content)| (role.as_str(), content.as_str())).collect::<Vec<_>>();
        let prompt = engine.chat_prompt(&refs, Some(&system));
        print!("\nAssistant > ");
        io::stdout().flush().map_err(|e| e.to_string())?;
        let started = Instant::now();
        let mut answer = String::new();
        for chunk in engine.stream(&prompt, max_tokens, temperature, top_k, top_p, repeat_penalty, &[], seed) {
            print!("{chunk}");
            io::stdout().flush().map_err(|e| e.to_string())?;
            answer.push_str(&chunk);
        }
        let elapsed = started.elapsed().as_secs_f64();
        let tokens = loaded.tokenizer.encode(&answer).len();
        println!("\n[{tokens} tokens | {elapsed:.2}s | {:.2} tok/s]", tokens as f64 / elapsed.max(1e-6));
        messages.push(("assistant".into(), answer));
        seed = seed.wrapping_add(1);
    }
    Ok(())
}

fn parse_arg<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>, name: &str) -> Result<T, String> {
    args.next().ok_or_else(|| format!("{name} requires a value"))?.parse().map_err(|_| format!("invalid value for {name}"))
}
