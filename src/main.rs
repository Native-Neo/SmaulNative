use libloading::{Library, Symbol};
use std::io::{self, Write};
use std::path::PathBuf;

#[derive(Clone, Copy)]
enum Workflow {
    FineTune,
    Sft,
    Pretrain,
    QuantizeGguf,
    SafetensorsToGguf,
    ExportOnnx,
    Inference,
}

impl Workflow {
    fn from_choice(choice: &str) -> Option<Self> {
        match choice.trim() {
            "1" => Some(Self::FineTune),
            "2" => Some(Self::Sft),
            "3" => Some(Self::Pretrain),
            "4" => Some(Self::QuantizeGguf),
            "5" => Some(Self::SafetensorsToGguf),
            "6" => Some(Self::ExportOnnx),
            "7" => Some(Self::Inference),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::FineTune => "Fine Tune",
            Self::Sft => "SFT",
            Self::Pretrain => "Pretrain",
            Self::QuantizeGguf => "Quantize GGUF",
            Self::SafetensorsToGguf => "Safetensors -> GGUF",
            Self::ExportOnnx => "Export ONNX",
            Self::Inference => "Inference",
        }
    }

    fn library_name(self) -> &'static str {
        match self {
            Self::FineTune => "libsmaul_finetune.so",
            Self::Sft => "libsmaul_sft.so",
            Self::Pretrain => "libsmaul_pretrain.so",
            Self::QuantizeGguf => "libsmaul_quantize-gguf.so",
            Self::SafetensorsToGguf => "libsmaul_safetensors-gguf.so",
            Self::ExportOnnx => "libsmaul_export-onnx.so",
            Self::Inference => "libsmaul_inference.so",
        }
    }
}

fn print_menu() {
    println!();
    println!("SmaulNative Rust");
    println!("================");
    println!("1. Fine Tune");
    println!("2. SFT");
    println!("3. Pretrain");
    println!("4. Quantize GGUF");
    println!("5. Safetensors -> GGUF");
    println!("6. Export ONNX");
    println!("7. Inference");
    println!("8. Exit");
    print!("\nSelect: ");
    io::stdout().flush().expect("failed to flush stdout");
}

fn launcher_dir() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| format!("failed to locate launcher: {e}"))?;
    exe.parent()
        .map(PathBuf::from)
        .ok_or_else(|| "launcher has no parent directory".into())
}

fn run_workflow(workflow: Workflow) -> Result<(), String> {
    let library_path = launcher_dir()?.join(workflow.library_name());
    if !library_path.is_file() {
        return Err(format!(
            "workflow library is missing: {}\nRun `cargo build` to build the workflow libraries.",
            library_path.display()
        ));
    }

    println!("Starting {}...", workflow.name());
    let library = unsafe { Library::new(&library_path) }
        .map_err(|e| format!("failed to load {}: {e}", library_path.display()))?;
    let run: Symbol<unsafe extern "C" fn() -> i32> = unsafe {
        library
            .get(b"smaul_workflow_run")
            .map_err(|e| format!("workflow library has no entrypoint: {e}"))?
    };
    let code = unsafe { run() };
    if code == 0 {
        Ok(())
    } else {
        Err(format!("{} exited with status {code}", workflow.name()))
    }
}

fn main() {
    loop {
        print_menu();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            eprintln!("Failed to read selection.");
            continue;
        }

        if input.trim() == "8" {
            break;
        }

        match Workflow::from_choice(&input) {
            Some(workflow) => {
                if let Err(error) = run_workflow(workflow) {
                    eprintln!("[WORKFLOW ERROR] {error}");
                }
            }
            None => println!("Invalid selection."),
        }
    }
}
