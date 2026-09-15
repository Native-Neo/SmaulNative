use std::process::Command;

fn main() -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let train = exe.with_file_name("smaul-train");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut command = Command::new(train);
    command.args(&args);
    if !args.iter().any(|arg| arg == "--mode") {
        command.args(["--mode", "sft"]);
    }
    let status = command.status().map_err(|e| e.to_string())?;
    if status.success() { Ok(()) } else { Err(format!("SFT exited with {status}")) }
}
