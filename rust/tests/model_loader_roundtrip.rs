use safetensors::tensor::{serialize, Dtype, TensorView};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use smaul_native::rwkv_model::{RwkvModel, RwkvModelConfig};

fn tensor_view<'a>(data: &'a [f32], shape: &[usize]) -> TensorView<'a> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<f32>())
    };
    TensorView::new(Dtype::F32, shape.to_vec(), bytes).unwrap()
}

fn temp_checkpoint_path() -> PathBuf {
    std::env::temp_dir().join(format!("smaulnative-loader-{}.safetensors", std::process::id()))
}

#[test]
fn safetensors_loader_preserves_model_outputs() {
    let config = RwkvModelConfig::new(32, 16, 2, 4);
    let model = RwkvModel::new(config.clone(), 1234);
    let token_ids = [1usize, 7, 3, 12];
    let path = temp_checkpoint_path();
    let mut tensors = HashMap::new();

    let emb = model.embedding.weight.clone().into_raw_vec();
    tensors.insert("emb.weight".to_string(), tensor_view(&emb, &[32, 16]));
    let ln_out_w = model.ln_out.weight.to_vec();
    let ln_out_b = model.ln_out.bias.to_vec();
    tensors.insert("ln_out.weight".to_string(), tensor_view(&ln_out_w, &[16]));
    tensors.insert("ln_out.bias".to_string(), tensor_view(&ln_out_b, &[16]));
    let head = model.head.weight.clone().into_raw_vec();
    tensors.insert("head.weight".to_string(), tensor_view(&head, &[32, 16]));

    for (i, block) in model.rwkv_blocks.iter().enumerate() {
        let p = format!("rwkv_blocks.{i}");
        if let Some(norm) = &block.ln0 {
            let w = norm.weight.to_vec();
            let b = norm.bias.to_vec();
            tensors.insert(format!("{p}.ln0.weight"), tensor_view(&w, &[16]));
            tensors.insert(format!("{p}.ln0.bias"), tensor_view(&b, &[16]));
        }
        for (name, norm) in [("ln1", &block.ln1), ("ln2", &block.ln2)] {
            let w = norm.weight.to_vec();
            let b = norm.bias.to_vec();
            tensors.insert(format!("{p}.{name}.weight"), tensor_view(&w, &[16]));
            tensors.insert(format!("{p}.{name}.bias"), tensor_view(&b, &[16]));
        }
        let t = &block.time_mix;
        for (name, data) in [
            ("x_r", t.x_r.to_vec()), ("x_w", t.x_w.to_vec()), ("x_k", t.x_k.to_vec()),
            ("x_v", t.x_v.to_vec()), ("x_a", t.x_a.to_vec()), ("x_g", t.x_g.to_vec()),
            ("w0", t.w0.to_vec()), ("a0", t.a0.to_vec()), ("v0", t.v0.to_vec()),
            ("k_k", t.k_k.to_vec()), ("k_a", t.k_a.to_vec()),
        ] {
            tensors.insert(format!("{p}.att.{name}"), tensor_view(&data, &[16]));
        }
        for (name, a) in [
            ("w1", &t.w1), ("w2", &t.w2), ("a1", &t.a1), ("a2", &t.a2),
            ("v2", &t.v2), ("g1", &t.g1), ("g2", &t.g2), ("r_k", &t.r_k),
            ("receptance.weight", &t.receptance), ("key.weight", &t.key),
            ("value.weight", &t.value), ("output.weight", &t.output),
        ] {
            let data = a.t().to_owned().into_raw_vec();
            let shape = a.shape();
            let py_shape = vec![shape[1], shape[0]];
            tensors.insert(format!("{p}.att.{name}"), tensor_view(&data, &py_shape));
        }
        let ln_x_w = t.ln_x.weight.to_vec();
        let ln_x_b = t.ln_x.bias.to_vec();
        tensors.insert(format!("{p}.att.ln_x.weight"), tensor_view(&ln_x_w, &[16]));
        tensors.insert(format!("{p}.att.ln_x.bias"), tensor_view(&ln_x_b, &[16]));
        let c = &block.cmix;
        let x_k = c.x_k.to_vec();
        tensors.insert(format!("{p}.ffn.x_k"), tensor_view(&x_k, &[16]));
        let key = c.key.t().to_owned().into_raw_vec();
        let value = c.value.t().to_owned().into_raw_vec();
        tensors.insert(format!("{p}.ffn.key.weight"), tensor_view(&key, &[64, 16]));
        tensors.insert(format!("{p}.ffn.value.weight"), tensor_view(&value, &[16, 64]));
    }

    fs::write(&path, serialize(tensors, &None).unwrap()).unwrap();
    let expected = model.forward(&token_ids, None).0;
    let mut loaded = RwkvModel::new(config, 9999);
    loaded.load_safetensors(&path).unwrap();
    let actual = loaded.forward(&token_ids, None).0;
    fs::remove_file(&path).ok();
    assert_eq!(expected.shape(), actual.shape());
    let max_error = expected.iter().zip(actual.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_error < 1e-5, "loader changed logits; max error = {max_error}");
}
