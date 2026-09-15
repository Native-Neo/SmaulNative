use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();
    let hip = env::var_os("CARGO_FEATURE_HIP").is_some();
    if cuda && hip {
        panic!("cuda and hip features are mutually exclusive");
    }
    if cuda {
        compile("cuda", "rust_lowbit.cu", "nvcc", "cudart");
    }
    if hip {
        compile("hip", "rust_lowbit.hip", "hipcc", "amdhip64");
    }
    compile_workflow_libraries();
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
    if !status.success() {
        panic!("ar failed while building SmaulNative {kind} low-bit backend");
    }
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=smaul_{kind}_lowbit");
    if kind == "cuda" {
        if let Some(cuda_home) = env::var_os("CUDA_HOME") {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_home.to_string_lossy());
        } else {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        }
    } else if let Some(rocm_path) = env::var_os("ROCM_PATH") {
        println!("cargo:rustc-link-search=native={}/lib", rocm_path.to_string_lossy());
    } else {
        println!("cargo:rustc-link-search=native=/opt/rocm/lib");
    }
    println!("cargo:rustc-link-lib=dylib={runtime}");
    println!("cargo:rerun-if-changed={source}");
}

fn compile_workflow_libraries() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let profile_dir = out_dir
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("failed to locate Cargo profile directory")
        .to_path_buf();
    fs::create_dir_all(&profile_dir).expect("failed to create workflow library directory");

    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let target = env::var("TARGET").unwrap();
    let plugins = [
        "finetune",
        "sft",
        "pretrain",
        "quantize-gguf",
        "safetensors-gguf",
        "export-onnx",
        "inference",
    ];

    for name in plugins {
        let source = root.join("workflow_plugins").join(format!("{name}.rs"));
        let output = profile_dir.join(format!("libsmaul_{name}.so"));
        let crate_name = format!("smaul_workflow_{}", name.replace('-', "_"));
        let status = Command::new(&rustc)
            .arg("--crate-name")
            .arg(crate_name)
            .arg("--crate-type")
            .arg("cdylib")
            .arg("--edition")
            .arg("2021")
            .arg("--target")
            .arg(&target)
            .arg("-C")
            .arg("debuginfo=2")
            .arg(&source)
            .arg("-o")
            .arg(&output)
            .status()
            .unwrap_or_else(|e| panic!("failed to execute rustc for workflow {name}: {e}"));
        if !status.success() {
            panic!("rustc failed while building workflow library {name}");
        }
        println!("cargo:rerun-if-changed={}", source.display());
    }
}
