use std::env;
use std::process::Command;

fn main() {
    let cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();
    let hip = env::var_os("CARGO_FEATURE_HIP").is_some();
    if cuda && hip {
        panic!("cuda and hip features are mutually exclusive");
    }
    if cuda { compile("cuda", "rust_lowbit.cu", "nvcc", "cudart"); }
    if hip { compile("hip", "rust_lowbit.hip", "hipcc", "amdhip64"); }
}

fn compile(kind: &str, filename: &str, compiler: &str, runtime: &str) {
    let root = env::var("CARGO_MANIFEST_DIR").unwrap();
    let source = format!("{root}/gpu/{kind}/{filename}");
    let out = env::var("OUT_DIR").unwrap();
    let obj = format!("{out}/rust_lowbit_{kind}.o");
    let lib = format!("{out}/libsmaul_{kind}_lowbit.a");
    let status = Command::new(compiler)
        .args(["-O3", "-c", &source, "-o", &obj])
        .status()
        .unwrap_or_else(|e| panic!("failed to execute {compiler}: {e}"));
    if !status.success() {
        panic!("{compiler} failed while building SmaulNative {kind} low-bit backend");
    }
    let ar = env::var("AR").unwrap_or_else(|_| "ar".into());
    let status = Command::new(ar)
        .args(["crus", &lib, &obj])
        .status()
        .expect("failed to execute ar");
    if !status.success() { panic!("ar failed while building SmaulNative {kind} low-bit backend"); }
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=smaul_{kind}_lowbit");
    println!("cargo:rustc-link-lib=dylib={runtime}");
    println!("cargo:rerun-if-changed={source}");
}
