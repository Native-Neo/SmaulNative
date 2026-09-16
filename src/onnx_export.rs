use safetensors::SafeTensors;
use serde_json::Value;
use std::fs;
use std::path::Path;

fn varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn field(out: &mut Vec<u8>, number: u32, wire: u8, bytes: &[u8]) {
    varint(((number as u64) << 3) | wire as u64, out);
    if wire == 2 {
        varint(bytes.len() as u64, out);
    }
    out.extend_from_slice(bytes);
}

fn bytes_field(out: &mut Vec<u8>, number: u32, bytes: &[u8]) {
    field(out, number, 2, bytes);
}

fn string_field(out: &mut Vec<u8>, number: u32, value: &str) {
    bytes_field(out, number, value.as_bytes());
}

fn int_field(out: &mut Vec<u8>, number: u32, value: i64) {
    varint((number as u64) << 3, out);
    varint(value as u64, out);
}

fn tensor_proto(name: &str, shape: &[usize], dtype: i32, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for &dim in shape {
        int_field(&mut out, 1, dim as i64);
    }
    int_field(&mut out, 2, dtype as i64);
    string_field(&mut out, 8, name);
    bytes_field(&mut out, 10, data);
    out
}

fn dimension_param(name: &str) -> Vec<u8> {
    let mut out = Vec::new();
    string_field(&mut out, 2, name);
    out
}

fn tensor_type(dtype: i32, dims: &[Option<i64>], params: &[&str]) -> Vec<u8> {
    let mut shape = Vec::new();
    for (index, dim) in dims.iter().enumerate() {
        let mut d = Vec::new();
        if let Some(value) = dim {
            int_field(&mut d, 1, *value);
        } else {
            d = dimension_param(params[index]);
        }
        bytes_field(&mut shape, 1, &d);
    }

    let mut tensor = Vec::new();
    int_field(&mut tensor, 1, dtype as i64);
    bytes_field(&mut tensor, 2, &shape);

    let mut ty = Vec::new();
    bytes_field(&mut ty, 1, &tensor);
    ty
}

fn value_info(name: &str, dtype: i32, dims: &[Option<i64>], params: &[&str]) -> Vec<u8> {
    let mut out = Vec::new();
    string_field(&mut out, 1, name);
    let ty = tensor_type(dtype, dims, params);
    bytes_field(&mut out, 2, &ty);
    out
}

fn custom_node(weight_names: &[String], output: &str, config_json: &str) -> Vec<u8> {
    let mut out = Vec::new();
    string_field(&mut out, 1, "tokens");
    for input in weight_names {
        string_field(&mut out, 1, input);
    }
    string_field(&mut out, 2, output);
    string_field(&mut out, 3, "SmaulNativeRWKVX");
    string_field(&mut out, 4, "smaulnative");
    string_field(&mut out, 5, "RWKV-X execution with recurrent state and optional MOBA");

    let mut attr = Vec::new();
    string_field(&mut attr, 1, "config_json");
    int_field(&mut attr, 4, 3);
    bytes_field(&mut attr, 13, config_json.as_bytes());
    bytes_field(&mut out, 6, &attr);
    out
}

fn operator_set(domain: &str, version: i64) -> Vec<u8> {
    let mut out = Vec::new();
    string_field(&mut out, 1, domain);
    int_field(&mut out, 2, version);
    out
}

pub fn export(input_dir: impl AsRef<Path>, output: impl AsRef<Path>) -> Result<(), String> {
    let dir = input_dir.as_ref();
    let output_path = output.as_ref();
    let config_path = dir.join("config.json");
    let weights_path = dir.join("model.safetensors");
    let config = fs::read_to_string(&config_path)
        .map_err(|e| format!("failed to read {}: {e}", config_path.display()))?;
    let _: Value = serde_json::from_str(&config)
        .map_err(|e| format!("invalid config.json: {e}"))?;
    let bytes = fs::read(&weights_path)
        .map_err(|e| format!("failed to read {}: {e}", weights_path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .map_err(|e| format!("invalid Safetensors: {e}"))?;

    let mut initializers = Vec::new();
    let mut weight_names = Vec::new();
    for name in tensors.names() {
        let tensor = tensors
            .tensor(name)
            .map_err(|e| format!("failed to read tensor {name}: {e}"))?;
        let dtype = match tensor.dtype() {
            safetensors::Dtype::F32 => 1,
            safetensors::Dtype::F16 => 10,
            _ => {
                return Err(format!(
                    "ONNX export currently requires F32/F16 weights; tensor {name} is {:?}",
                    tensor.dtype()
                ))
            }
        };
        let shape = tensor.shape().to_vec();
        let data = tensor_proto(name, &shape, dtype, tensor.data());
        bytes_field(&mut initializers, 5, &data);
        weight_names.push(name.to_string());
    }

    let node = custom_node(&weight_names, "logits", &config);
    let mut graph = Vec::new();
    bytes_field(&mut graph, 1, &node);
    string_field(&mut graph, 2, "SmaulNativeRWKVX");
    graph.extend_from_slice(&initializers);

    let input = value_info("tokens", 7, &[None, None], &["batch", "sequence"]);
    let output = value_info(
        "logits",
        1,
        &[None, None, None],
        &["batch", "sequence", "vocab"],
    );
    bytes_field(&mut graph, 11, &input);
    bytes_field(&mut graph, 12, &output);

    let mut model = Vec::new();
    int_field(&mut model, 1, 9);
    string_field(&mut model, 2, "SmaulNative");
    bytes_field(&mut model, 7, &graph);
    bytes_field(&mut model, 8, &operator_set("", 19));
    bytes_field(&mut model, 8, &operator_set("smaulnative", 1));

    fs::write(output_path, model)
        .map_err(|e| format!("failed to write {}: {e}", output_path.display()))?;
    Ok(())
}

pub fn export_cli(args: &[String]) -> Result<(), String> {
    if args.len() != 3 {
        return Err("usage: smaul-export-onnx <model_dir> <output.onnx>".into());
    }
    export(&args[1], &args[2])
}
