use smaul_native::gguf::convert_cli;
fn main(){let args:Vec<String>=std::env::args().collect();if let Err(e)=convert_cli(&args){eprintln!("error: {e}");std::process::exit(1)}}
