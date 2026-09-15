use std::env;
use std::process::Command;

#[no_mangle]
pub extern "C" fn smaul_workflow_run() -> i32 {
    let exe = match env::current_exe() {
        Ok(path) => path,
        Err(e) => {
            eprintln!("failed to locate SmaulNative executable: {e}");
            return 1;
        }
    };
    let app = exe.with_file_name("smaul-finetune");
    match Command::new(app).args(env::args_os().skip(1)).status() {
        Ok(status) => status.code().unwrap_or(1),
        Err(e) => {
            eprintln!("failed to start smaul-finetune: {e}");
            1
        }
    }
}
