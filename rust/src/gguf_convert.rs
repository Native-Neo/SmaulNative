use std::path::Path;

/// Native Rust entrypoint for RWKV-X checkpoint conversion.
/// The actual GGUF serialization lives in the shared GGUF module.
pub fn convert(input_dir: impl AsRef<Path>, output: impl AsRef<Path>, dtype: &str) -> Result<(), String> {
    let input = input_dir.as_ref();
    let output = output.as_ref();
    if dtype != "f16" && dtype != "f32" {
        return Err("dtype must be f16 or f32".into());
    }
    if !input.join("config.json").is_file() {
        return Err(format!("missing {}", input.join("config.json").display()));
    }
    if !input.join("model.safetensors").is_file() {
        return Err(format!("missing {}", input.join("model.safetensors").display()));
    }
    if !input.join("tokenizer.json").is_file() {
        return Err(format!("missing {}", input.join("tokenizer.json").display()));
    }
    crate::gguf::convert_safetensors_checkpoint(input, output, dtype)
}

pub fn convert_cli(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err("usage: smaul-convert-gguf <input_dir> <output.gguf> [--dtype f16|f32]".into());
    }
    let dtype = if args.len() >= 5 && args[3] == "--dtype" { args[4].as_str() } else { "f16" };
    convert(&args[1], &args[2], dtype)
}
