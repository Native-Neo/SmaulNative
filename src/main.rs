use libloading::{Library, Symbol};
use std::env;
use std::path::PathBuf;
use std::process::Command;

#[derive(Clone, Copy)]
enum Workflow { FineTune, Sft, Pretrain, QuantizeGguf, SafetensorsToGguf, ExportOnnx, Inference }
impl Workflow {
    fn from_name(name: &str) -> Option<Self> { match name { "finetune" | "fine-tune" => Some(Self::FineTune), "sft" => Some(Self::Sft), "pretrain" => Some(Self::Pretrain), "quantize-gguf" | "quantize" => Some(Self::QuantizeGguf), "safetensors-gguf" | "convert-gguf" => Some(Self::SafetensorsToGguf), "export-onnx" | "onnx" => Some(Self::ExportOnnx), "inference" | "infer" => Some(Self::Inference), _ => None } }
    fn name(self) -> &'static str { match self { Self::FineTune => "Fine Tune", Self::Sft => "SFT", Self::Pretrain => "Pretrain", Self::QuantizeGguf => "Quantize GGUF", Self::SafetensorsToGguf => "Safetensors -> GGUF", Self::ExportOnnx => "Export ONNX", Self::Inference => "Inference" } }
    fn library_name(self) -> &'static str { match self { Self::FineTune => "libsmaul_finetune.so", Self::Sft => "libsmaul_sft.so", Self::Pretrain => "libsmaul_pretrain.so", Self::QuantizeGguf => "libsmaul_quantize-gguf.so", Self::SafetensorsToGguf => "libsmaul_safetensors-gguf.so", Self::ExportOnnx => "libsmaul_export-onnx.so", Self::Inference => "libsmaul_inference.so" } }
}

fn launcher_dir() -> Result<PathBuf, String> { let exe = env::current_exe().map_err(|e| format!("failed to locate launcher: {e}"))?; exe.parent().map(PathBuf::from).ok_or_else(|| "launcher has no parent directory".into()) }
fn workflow_library_path(workflow: Workflow) -> Result<PathBuf, String> { let name = workflow.library_name(); let local = launcher_dir()?.join(name); if local.is_file() { return Ok(local); } let build_output = PathBuf::from(env!("SMAUL_WORKFLOW_LIB_DIR")).join(name); if build_output.is_file() { return Ok(build_output); } Err(format!("workflow library is missing: {name}\nRun `cargo build` to build the workflow libraries.")) }
fn run_tokenizer_train(args: &[String]) -> Result<(), String> { let app = launcher_dir()?.join("smaul-tokenizer-train"); if !app.is_file() { return Err("tokenizer trainer is missing: smaul-tokenizer-train\nRun `cargo build --bin smaul-tokenizer-train` to build it.".into()); } let status = Command::new(app).args(args).status().map_err(|e| format!("failed to start tokenizer trainer: {e}"))?; if status.success() { Ok(()) } else { Err(format!("tokenizer trainer exited with status {}", status.code().unwrap_or(1))) } }
fn run_workflow(workflow: Workflow) -> Result<(), String> { let library_path = workflow_library_path(workflow)?; eprintln!("Starting {}...", workflow.name()); let library = unsafe { Library::new(&library_path) }.map_err(|e| format!("failed to load {}: {e}", library_path.display()))?; let run: Symbol<unsafe extern "C" fn() -> i32> = unsafe { library.get(b"smaul_workflow_run").map_err(|e| format!("workflow library has no entrypoint: {e}"))? }; let code = unsafe { run() }; if code == 0 { Ok(()) } else { Err(format!("{} exited with status {code}", workflow.name())) } }

fn print_help() { println!("SmaulNative Rust CLI"); println!(); println!("Usage:"); println!("  smaul-native <command> [options]"); println!(); println!("Commands:"); println!("  tokenizer-train      Train a whole-word tokenizer from a dataset"); println!("  finetune             Run fine-tuning workflow"); println!("  sft                  Run supervised fine-tuning workflow"); println!("  pretrain             Run pretraining workflow"); println!("  quantize-gguf        Quantize a GGUF model"); println!("  safetensors-gguf     Convert Safetensors to GGUF"); println!("  export-onnx          Export a model to ONNX"); println!("  inference            Run model inference"); println!(); println!("Options after the command are passed to the selected workflow."); }

fn main() {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else { print_help(); std::process::exit(2); };
    if matches!(command.as_str(), "help" | "--help" | "-h") { print_help(); return; }
    if matches!(command.as_str(), "version" | "--version" | "-V") { println!("smaul-native {}", env!("CARGO_PKG_VERSION")); return; }
    let workflow_args: Vec<String> = args.collect();
    if command == "tokenizer-train" { if let Err(error) = run_tokenizer_train(&workflow_args) { eprintln!("[WORKFLOW ERROR] {error}"); std::process::exit(1); } return; }
    let Some(workflow) = Workflow::from_name(&command) else { eprintln!("unknown command: {command}"); eprintln!("Run `smaul-native --help` for usage."); std::process::exit(2); };
    if let Err(error) = run_workflow(workflow) { eprintln!("[WORKFLOW ERROR] {error}"); std::process::exit(1); }
}
