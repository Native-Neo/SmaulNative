use smaul_native::gguf_quantize::quantize_cli;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Err(error) = quantize_cli(&args) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
