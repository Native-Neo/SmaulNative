fn main() {
    eprintln!("SmaulNative GGUF quantization");
    eprintln!("The GGUF reader supports quantized tensor formats, but the native GGUF quantizer is not implemented yet.");
    eprintln!("Use smaul-convert-gguf for Safetensors -> GGUF conversion.");
    std::process::exit(2);
}
