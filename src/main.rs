use std::io::{self, Write};

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

fn run_workflow(workflow: Workflow) {
    match workflow {
        Workflow::FineTune => println!("Fine-tuning workflow selected."),
        Workflow::Sft => println!("SFT workflow selected."),
        Workflow::Pretrain => println!("Pretraining workflow selected."),
        Workflow::QuantizeGguf => println!("GGUF quantization workflow selected."),
        Workflow::SafetensorsToGguf => println!("Safetensors -> GGUF workflow selected."),
        Workflow::ExportOnnx => println!("ONNX export workflow selected."),
        Workflow::Inference => println!("Inference workflow selected."),
    }
    println!("This workflow is being implemented natively in Rust.");
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
            Some(workflow) => run_workflow(workflow),
            None => println!("Invalid selection."),
        }
    }
}
