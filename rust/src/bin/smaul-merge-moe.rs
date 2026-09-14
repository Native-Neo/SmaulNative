use smaul_native::merge_moe::merge;
use std::env;
use std::path::PathBuf;

fn value(args: &[String], name: &str) -> Option<String> { args.windows(2).find(|x| x[0] == name).map(|x| x[1].clone()) }

fn main() -> Result<(), String> {
    let args = env::args().collect::<Vec<_>>();
    let base = value(&args, "--base").ok_or("--base is required")?;
    let out = value(&args, "--out").ok_or("--out is required")?;
    let mut branches = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--branch" && i + 1 < args.len() { branches.push(PathBuf::from(&args[i + 1])); i += 1; }
        i += 1;
    }
    if branches.is_empty() {
        let mut after = false;
        for arg in &args[1..] {
            if arg == "--branches" { after = true; continue; }
            if after && arg.starts_with("--") { break; }
            if after { branches.push(PathBuf::from(arg)); }
        }
    }
    let top_k = value(&args, "--top-k").or_else(|| value(&args, "--top_k")).map(|x| x.parse().map_err(|_| "invalid --top-k")).transpose()?.unwrap_or(1);
    merge(base, &branches, out, top_k)
}
