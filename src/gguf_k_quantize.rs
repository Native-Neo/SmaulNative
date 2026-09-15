use crate::gguf::{GgufReader, GgufValue};
use crate::gguf_writer::{Tensor, Writer};
use std::path::Path;

fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7fffff;
    if exp <= 0 {
        if exp < -10 { return sign; }
        return sign | (((mantissa | 0x800000) >> (1 - exp) + 0x1000) >> 13) as u16;
    }
    if exp >= 31 { return sign | 0x7c00; }
    sign | ((exp as u16) << 10) | ((mantissa + 0x1000) >> 13) as u16
}

fn round_i32(value: f32) -> i32 { value.round() as i32 }

fn quantize_q3_k(values: &[f32]) -> Result<Vec<u8>, String> {
    if values.len() % 256 != 0 { return Err("Q3_K requires tensor size divisible by 256".into()); }
    let mut out = Vec::with_capacity(values.len() / 256 * 110);
    for block in values.chunks_exact(256) {
        let mut group_scales = [0.0f32; 16];
        let mut max_scale = 0.0f32;
        for group in 0..16 {
            let chunk = &block[group * 16..group * 16 + 16];
            let max_abs = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = max_abs / 4.0;
            group_scales[group] = scale;
            max_scale = max_scale.max(scale);
        }
        let d = max_scale / 32.0;
        let mut scale_codes = [0i32; 16];
        let mut q = [0u8; 256];
        if d != 0.0 {
            for group in 0..16 {
                scale_codes[group] = round_i32(group_scales[group] / d).clamp(0, 32);
                let scale = d * scale_codes[group] as f32;
                if scale == 0.0 { continue; }
                for i in 0..16 {
                    q[group * 16 + i] = (round_i32(block[group * 16 + i] / scale).clamp(-4, 3) + 4) as u8;
                }
            }
        }
        let mut scales = [0u8; 12];
        for j in 0..16 {
            let code = scale_codes[j] as u8;
            if j < 8 { scales[j] = code & 0x0f; } else { scales[j - 8] |= (code & 0x0f) << 4; }
            scales[j % 4 + 8] |= ((code >> 4) & 0x03) << (2 * (j / 4));
        }
        let mut hmask = [0u8; 32];
        let mut low = q;
        for i in 0..256 {
            if low[i] >= 4 { hmask[i / 8] |= 1 << (i % 8); low[i] -= 4; }
        }
        let mut qs = [0u8; 64];
        for base in (0..256).step_by(128) {
            for i in 0..32 {
                qs[base / 4 + i] = low[base + i]
                    | (low[base + i + 32] << 2)
                    | (low[base + i + 64] << 4)
                    | (low[base + i + 96] << 6);
            }
        }
        out.extend_from_slice(&hmask);
        out.extend_from_slice(&qs);
        out.extend_from_slice(&scales);
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    }
    Ok(out)
}

fn quantize_q6_k(values: &[f32]) -> Result<Vec<u8>, String> {
    if values.len() % 256 != 0 { return Err("Q6_K requires tensor size divisible by 256".into()); }
    let mut out = Vec::with_capacity(values.len() / 256 * 210);
    for block in values.chunks_exact(256) {
        let mut group_scales = [0.0f32; 16];
        let mut max_scale = 0.0f32;
        for group in 0..16 {
            let chunk = &block[group * 16..group * 16 + 16];
            let max_abs = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = max_abs / 32.0;
            group_scales[group] = scale;
            max_scale = max_scale.max(scale);
        }
        let d = max_scale / 127.0;
        let mut scale_codes = [0i8; 16];
        let mut q = [0u8; 256];
        if d != 0.0 {
            for group in 0..16 {
                scale_codes[group] = round_i32(group_scales[group] / d).clamp(0, 127) as i8;
                let scale = d * scale_codes[group] as f32;
                if scale == 0.0 { continue; }
                for i in 0..16 {
                    q[group * 16 + i] = (round_i32(block[group * 16 + i] / scale).clamp(-32, 31) + 32) as u8;
                }
            }
        }
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for base in (0..256).step_by(128) {
            for i in 0..32 {
                let q1 = q[base + i];
                let q2 = q[base + i + 32];
                let q3 = q[base + i + 64];
                let q4 = q[base + i + 96];
                ql[base / 2 + i] = (q1 & 0x0f) | ((q3 & 0x0f) << 4);
                ql[base / 2 + i + 32] = (q2 & 0x0f) | ((q4 & 0x0f) << 4);
                qh[base / 4 + i] = ((q1 >> 4) & 0x03)
                    | (((q2 >> 4) & 0x03) << 2)
                    | (((q3 >> 4) & 0x03) << 4)
                    | (((q4 >> 4) & 0x03) << 6);
            }
        }
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        for scale in scale_codes { out.push(scale as u8); }
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    }
    Ok(out)
}

fn copy_metadata(reader: &GgufReader, writer: &mut Writer) {
    for (key, value) in &reader.metadata {
        match value {
            GgufValue::String(v) => writer.add_meta_str(key, v),
            GgufValue::U32(v) => writer.add_meta_u32(key, *v),
            GgufValue::Bool(v) => writer.add_meta_bool(key, *v),
            GgufValue::Array(values) if values.iter().all(|v| matches!(v, GgufValue::String(_))) => {
                let tokens = values.iter().map(|v| match v {
                    GgufValue::String(s) => s.clone(),
                    _ => unreachable!(),
                }).collect::<Vec<_>>();
                writer.add_tokens(key, &tokens);
            }
            _ => {}
        }
    }
}

pub fn quantize(input: impl AsRef<Path>, output: impl AsRef<Path>, kind: &str) -> Result<(), String> {
    if !matches!(kind, "q3_k" | "q6_k") { return Err("K quantization must be q3_k or q6_k".into()); }
    let input = input.as_ref();
    let mut reader = GgufReader::open(input)?;
    let mut writer = Writer::new();
    copy_metadata(&reader, &mut writer);
    writer.add_meta_str("smaulnative.quantization", kind);
    let tensors = reader.tensors.clone();
    for info in tensors {
        let values = reader.tensor_f32(&info.name)?;
        let (dtype, data) = if values.len() % 256 == 0 {
            let data = match kind {
                "q3_k" => quantize_q3_k(&values)?,
                "q6_k" => quantize_q6_k(&values)?,
                _ => unreachable!(),
            };
            (if kind == "q3_k" { 11 } else { 14 }, data)
        } else {
            let mut data = Vec::with_capacity(values.len() * 2);
            for value in values { data.extend_from_slice(&f32_to_f16(value).to_le_bytes()); }
            (1, data)
        };
        writer.tensor(Tensor { name: &info.name, shape: &info.shape, dtype, data: &data });
    }
    writer.finish(output.as_ref().to_str().ok_or("invalid output path")?)
}
