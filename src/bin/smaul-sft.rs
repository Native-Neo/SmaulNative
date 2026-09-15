use std::path::Path;
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

    require_backend(&train, "SFT")?;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut command = Command::new(&train);
    command.args(&args);

    if !args.iter().any(|arg| arg == "--mode") {
        command.args(["--mode", "sft"]);
    }

    let status = command
        .status()
        .map_err(|e| format!("failed to start smaul-train at {}: {e}", train.display()))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("SFT backend exited with {status}"))
    }
}

fn require_backend(path: &Path, workflow: &str) -> Result<(), String> {
    if path.is_file() {
        Ok(())
    } else {
        Err(format!(
            "{workflow} backend is missing: {}\nBuild the project again to restore smaul-train.",
            path.display()
        ))
    }
}
