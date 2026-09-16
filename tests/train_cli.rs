// Drives the real smaul-train binary over a real model directory and dataset.
use smaul_native::config::RwkvXConfig;
use smaul_native::checkpoint::PretrainedModel;
use smaul_native::rwkv_model::RwkvModel;
use smaul_native::tokenizer::Tokenizer;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("smaul-train-cli-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn build_model(dir: &Path, qat_bits: u8) -> RwkvXConfig {
    let vocab: Vec<String> = ["<pad>", "<unk>", "<bos>", "<eos>", "<cap>", "<upper>"]
        .iter().map(|s| s.to_string())
        .chain("abcdefghijklmnop".chars().map(|c| c.to_string()))
        .collect();
    let config = RwkvXConfig { vocab_size: vocab.len(), n_embd: 16, n_layer: 2, head_size: 8, n_moba_layer: 0, ctx_len_hint: 16, qat_bits, ..RwkvXConfig::default() };
    config.validate().unwrap();
    let model = RwkvModel::new(config.to_model_config(), 3);
    PretrainedModel { config: config.clone(), model, tokenizer: Tokenizer::from_vocab(vocab) }
        .save_model(dir).unwrap();
    config
}

fn dataset(path: &Path) {
    let lines: String = (0..64).map(|i| format!("{{\"text\":\"{}\"}}\n", "abcdefghijklmnop".chars().cycle().skip(i % 16).take(40).collect::<String>())).collect();
    fs::write(path, lines).unwrap();
}

fn train(model_dir: &Path, data: &Path, out: &Path, extra: &[&str]) -> std::process::Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_smaul-train"));
    cmd.args(extra);
    cmd.args(["--model", model_dir.to_str().unwrap(), "--dataset", data.to_str().unwrap(),
              "--output-dir", out.to_str().unwrap(), "--steps", "2", "--ctx-len", "8",
              "--save-every", "2", "--log-every", "1", "--optimizer", "adamw"]);
    cmd.output().unwrap()
}

#[test]
fn trains_and_writes_a_resumable_checkpoint() {
    let ws = workspace("plain");
    let model_dir = ws.join("model");
    let out = ws.join("out");
    build_model(&model_dir, 0);
    let data = ws.join("train.jsonl");
    dataset(&data);

    let output = train(&model_dir, &data, &out, &[]);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let log = String::from_utf8_lossy(&output.stdout);
    assert!(log.contains("step 1"), "no progress reported: {log}");
    for f in ["config.json", "model.safetensors", "tokenizer.json", "optimizer.bin", "training_state.json"] {
        assert!(out.join(f).is_file(), "checkpoint is missing {f}");
    }
    // the checkpoint must load back through the normal loader
    PretrainedModel::load(&out).unwrap();

    // resuming from a completed run is a no-op rather than an error
    let again = train(&model_dir, &data, &out, &["--resume"]);
    assert!(again.status.success(), "{}", String::from_utf8_lossy(&again.stderr));
    fs::remove_dir_all(&ws).ok();
}

#[test]
fn rqt_training_runs_and_keeps_masters_off_the_grid() {
    let ws = workspace("rqt");
    let model_dir = ws.join("model");
    let out = ws.join("out");
    build_model(&model_dir, 0);
    let data = ws.join("train.jsonl");
    dataset(&data);

    let output = train(&model_dir, &data, &out, &["--rqt-bits", "3"]);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let log = String::from_utf8_lossy(&output.stdout);
    assert!(log.contains("[RQT] 3-bit"), "RQT was not engaged: {log}");

    let trained = PretrainedModel::load(&out).unwrap();
    let mut off_grid = 0usize;
    for row in trained.model.head.weight.rows() {
        let peak = row.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        off_grid += row.iter().filter(|v| **v != 0.0 && (v.abs() / peak * 3.0 - (v.abs() / peak * 3.0).round()).abs() > 1e-4).count();
    }
    assert!(off_grid > 0, "saved weights sit exactly on the 3-bit grid; masters were not preserved");
    fs::remove_dir_all(&ws).ok();
}

#[test]
fn bad_arguments_report_errors_instead_of_panicking() {
    let ws = workspace("args");
    let model_dir = ws.join("model");
    build_model(&model_dir, 0);
    let data = ws.join("train.jsonl");
    dataset(&data);

    for (flag, value, needle) in [("--steps", "many", "--steps expects a number"), ("--rqt-bits", "7", "--rqt-bits must be")] {
        let output = train(&model_dir, &data, &ws.join("out"), &[flag, value]);
        assert!(!output.status.success());
        let err = String::from_utf8_lossy(&output.stderr);
        assert!(err.contains(needle), "expected '{needle}', got: {err}");
        assert!(!err.contains("panicked"), "CLI panicked on bad input: {err}");
    }
    fs::remove_dir_all(&ws).ok();
}

#[test]
fn empty_sft_dataset_fails_instead_of_hanging() {
    let ws = workspace("sft");
    let model_dir = ws.join("model");
    build_model(&model_dir, 0);
    let data = ws.join("sft.jsonl");
    fs::write(&data, "\n\n").unwrap();

    let output = train(&model_dir, &data, &ws.join("out"), &["--mode", "sft"]);
    assert!(!output.status.success());
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(err.contains("no usable examples") || err.contains("SFT"), "unexpected error: {err}");
    fs::remove_dir_all(&ws).ok();
}
