use smaul_native::onnx_export::export_cli;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Err(error) = export_cli(&args) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
