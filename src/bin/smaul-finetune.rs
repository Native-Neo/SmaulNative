use std::path::PathBuf;
use std::process::Command;

fn main() {
    if let Err(error) = run() {
        eprintln!("[WORKFLOW ERROR] {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| format!("failed to locate executable: {e}"))?;
    let train = exe.with_file_name("smaul-train");

    if !train.is_file() {
        return Err(format!(
            "training backend is missing: {}\nBuild the project again to restore smaul-train.",
            display_path(&train)
        ));
    }

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut command = Command::new(&train);
    command.args(&args);

    if !args.iter().any(|arg| arg == "--mode") {
        command.args(["--mode", "pretrain"]);
    }

    let status = command
        .status()
        .map_err(|e| format!("failed to start smaul-train at {}: {e}", display_path(&train)))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("fine-tuning backend exited with {status}"))
    }
}

fn display_path(path: &PathBuf) -> String {
    path.display().to_string()
}
