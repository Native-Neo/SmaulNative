use std::env;
use std::process::Command;

fn main() {
    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        compile("cuda", "nvcc");
    }
    if env::var_os("CARGO_FEATURE_HIP").is_some() {
        compile("hip", "hipcc");
    }
}

fn compile(kind: &str, compiler: &str) {
    let root = env::var("CARGO_MANIFEST_DIR").unwrap();
    let source = format!("{root}/gpu/{kind}/rust_lowbit.cu");
    let out = env::var("OUT_DIR").unwrap();
    let lib = format!("{out}/libsmaul_{kind}_lowbit.a");
    let status = Command::new(compiler)
        .args(["-O3", "-c", &source, "-o", &format!("{out}/rust_lowbit.o")])
        .status()
        .unwrap_or_else(|e| panic!("failed to execute {compiler}: {e}"));
    if !status.success() { panic!("{compiler} failed while building SmaulNative {kind} low-bit backend"); }
    let ar = env::var("AR").unwrap_or_else(|_| "ar".into());
    let status = Command::new(ar).args(["crus", &lib, &format!("{out}/rust_lowbit.o")]).status().expect("failed to execute ar");
    if !status.success() { panic!("ar failed while building SmaulNative {kind} low-bit backend"); }
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=smaul_{kind}_lowbit");
    println!("cargo:rerun-if-changed={source}");
}
