use smaul_native::inference::Server;
use std::env;
fn val(name:&str,default:&str)->String{let a:Vec<String>=env::args().collect();a.windows(2).find(|x|x[0]==name).map(|x|x[1].clone()).unwrap_or_else(||default.into())}
fn main(){let model=val("--model","./SmaulNative");let host=val("--host","127.0.0.1");let port:u16=val("--port","8080").parse().unwrap();let max:usize=val("--max-prompt-tokens","4096").parse().unwrap();Server::new(model,max).and_then(|s|s.serve(&host,port)).unwrap_or_else(|e|{eprintln!("{e}");std::process::exit(1)})}
