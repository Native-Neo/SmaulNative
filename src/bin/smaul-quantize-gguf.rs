use smaul_native::gguf_k_quantize;
use smaul_native::gguf_quantize::quantize_cli;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kind = args
        .windows(2)
        .find(|pair| pair[0] == "--type")
        .map(|pair| pair[1].as_str());

    let result = match kind {
        Some("q3_k") | Some("q6_k") => {
            if args.len() < 4 {
                Err("usage: smaul-quantize-gguf <input.gguf> <output.gguf> --type q3_k|q6_k".into())
            } else {
                gguf_k_quantize::quantize(&args[1], &args[2], kind.unwrap())
            }
        }
        _ => quantize_cli(&args),
    };

    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
