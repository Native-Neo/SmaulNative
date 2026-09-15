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

fn quantize_block(values: &[f32], kind: &str) -> Result<(u32, Vec<u8>), String> {
    if values.len() != 32 { return Err("GGUF classic quantization requires 32-element blocks".into()); }
    let mut out = Vec::new();
    match kind {
        "q4_0" => {
            let max = values.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let d = if max == 0.0 { 0.0 } else { max / 8.0 };
            out.extend(f32_to_f16(d).to_le_bytes());
            for pair in values.chunks_exact(2) {
                let q0 = if d == 0.0 { 8 } else { (pair[0] / d).round().clamp(-8.0, 7.0) as i8 + 8 };
                let q1 = if d == 0.0 { 8 } else { (pair[1] / d).round().clamp(-8.0, 7.0) as i8 + 8 };
                out.push((q0 as u8) | ((q1 as u8) << 4));
            }
            Ok((2, out))
        }
        "q4_1" => {
            let min = values.iter().copied().fold(f32::INFINITY, f32::min);
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let d = if max == min { 0.0 } else { (max - min) / 15.0 };
            out.extend(f32_to_f16(d).to_le_bytes());
            out.extend(f32_to_f16(min).to_le_bytes());
            for pair in values.chunks_exact(2) {
                let q0 = if d == 0.0 { 0 } else { ((pair[0] - min) / d).round().clamp(0.0, 15.0) as u8 };
                let q1 = if d == 0.0 { 0 } else { ((pair[1] - min) / d).round().clamp(0.0, 15.0) as u8 };
                out.push(q0 | (q1 << 4));
            }
            Ok((3, out))
        }
        "q5_0" => {
            let max = values.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let d = if max == 0.0 { 0.0 } else { max / 16.0 };
            out.extend(f32_to_f16(d).to_le_bytes());
            let mut qh = 0u32;
            let mut low = Vec::with_capacity(16);
            for (i, &x) in values.iter().enumerate() {
                let q = if d == 0.0 { 16 } else { (x / d).round().clamp(-16.0, 15.0) as i8 + 16 } as u8;
                if q & 16 != 0 { qh |= 1 << i; }
                if i % 2 == 0 { low.push(q & 15); } else { *low.last_mut().unwrap() |= (q & 15) << 4; }
            }
            out.extend(qh.to_le_bytes());
            out.extend(low);
            Ok((6, out))
        }
        "q5_1" => {
            let min = values.iter().copied().fold(f32::INFINITY, f32::min);
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let d = if max == min { 0.0 } else { (max - min) / 31.0 };
            out.extend(f32_to_f16(d).to_le_bytes());
            out.extend(f32_to_f16(min).to_le_bytes());
            let mut qh = 0u32;
            let mut low = Vec::with_capacity(16);
            for (i, &x) in values.iter().enumerate() {
                let q = if d == 0.0 { 0 } else { ((x - min) / d).round().clamp(0.0, 31.0) as u8 };
                if q & 16 != 0 { qh |= 1 << i; }
                if i % 2 == 0 { low.push(q & 15); } else { *low.last_mut().unwrap() |= (q & 15) << 4; }
            }
            out.extend(qh.to_le_bytes());
            out.extend(low);
            Ok((7, out))
        }
        "q8_0" => {
            let max = values.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let d = if max == 0.0 { 0.0 } else { max / 127.0 };
            out.extend(f32_to_f16(d).to_le_bytes());
            for &x in values {
                let q = if d == 0.0 { 0 } else { (x / d).round().clamp(-128.0, 127.0) as i8 };
                out.push(q as u8);
            }
            Ok((8, out))
        }
        _ => Err(format!("unsupported GGUF quantization type: {kind}")),
    }
}

fn quantize_tensor(values: &[f32], kind: &str) -> Result<(u32, Vec<u8>), String> {
    if values.len() % 32 != 0 { return Err(format!("tensor has {} elements; classic GGUF quantization requires a multiple of 32", values.len())); }
    let mut dtype = 0;
    let mut bytes = Vec::with_capacity(values.len());
    for block in values.chunks_exact(32) {
        let (block_dtype, block_bytes) = quantize_block(block, kind)?;
        dtype = block_dtype;
        bytes.extend(block_bytes);
    }
    Ok((dtype, bytes))
}

fn pack_k4_scales(scales: &[u8; 8], mins: &[u8; 8]) -> [u8; 12] {
    let mut out = [0u8; 12];
    for j in 0..4 {
        out[j] = scales[j] & 63;
        out[j + 4] = mins[j] & 63;
    }
    for j in 4..8 {
        out[j + 4] = (scales[j] & 15) | ((mins[j] & 15) << 4);
        out[j - 4] |= (scales[j] >> 4) << 6;
        out[j] |= (mins[j] >> 4) << 6;
    }
    out
}

fn quantize_k_subblocks(values: &[f32], bits: u8) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), String> {
    let levels = ((1u16 << bits) - 1) as f32;
    let groups = values.len() / 32;
    let mut scales = vec![0u8; groups];
    let mut mins = vec![0u8; groups];
    let mut codes = vec![0u8; values.len()];
    let mut max_scale = 0.0f32;
    let mut max_min = 0.0f32;

    for g in 0..groups {
        let block = &values[g * 32..(g + 1) * 32];
        let min = block.iter().copied().fold(f32::INFINITY, f32::min);
        let max = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = if max == min { 0.0 } else { (max - min) / levels };
        let offset = -min;
        max_scale = max_scale.max(scale);
        max_min = max_min.max(offset);
        let d = if scale == 0.0 { 0.0 } else { scale };
        for (i, &x) in block.iter().enumerate() {
            let q = if d == 0.0 { 0 } else { ((x + offset) / d).round().clamp(0.0, levels) as u8 };
            codes[g * 32 + i] = q;
        }
        scales[g] = if max_scale == 0.0 { 0 } else { (scale / max_scale * 63.0).round().clamp(0.0, 63.0) as u8 };
        mins[g] = if max_min == 0.0 { 0 } else { (offset / max_min * 63.0).round().clamp(0.0, 63.0) as u8 };
    }

    Ok((scales, mins, codes))
}

fn quantize_q4_k(values: &[f32]) -> Result<Vec<u8>, String> {
    if values.len() % 256 != 0 { return Err("Q4_K requires tensor size divisible by 256".into()); }
    let mut out = Vec::with_capacity(values.len() / 256 * 144);
    for block in values.chunks_exact(256) {
        let (scales, mins, codes) = quantize_k_subblocks(block, 4)?;
        let max_scale = block.chunks_exact(32).map(|b| {
            let min = b.iter().copied().fold(f32::INFINITY, f32::min);
            let max = b.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if max == min { 0.0 } else { (max - min) / 15.0 }
        }).fold(0.0, f32::max);
        let max_min = block.chunks_exact(32).map(|b| -b.iter().copied().fold(f32::INFINITY, f32::min)).fold(0.0, f32::max);
        let scale_bytes = pack_k4_scales(
            &scales.try_into().map_err(|_| "invalid Q4_K scales")?,
            &mins.try_into().map_err(|_| "invalid Q4_K mins")?,
        );
        out.extend(f32_to_f16(max_scale / 63.0).to_le_bytes());
        out.extend(f32_to_f16(max_min / 63.0).to_le_bytes());
        out.extend(scale_bytes);
        for j in 0..4 {
            for l in 0..32 {
                let a = codes[j * 64 + l];
                let b = codes[j * 64 + l + 32];
                out.push(a | (b << 4));
            }
        }
    }
    Ok(out)
}

fn quantize_q5_k(values: &[f32]) -> Result<Vec<u8>, String> {
    if values.len() % 256 != 0 { return Err("Q5_K requires tensor size divisible by 256".into()); }
    let mut out = Vec::with_capacity(values.len() / 256 * 176);
    for block in values.chunks_exact(256) {
        let (scales, mins, codes) = quantize_k_subblocks(block, 5)?;
        let max_scale = block.chunks_exact(32).map(|b| {
            let min = b.iter().copied().fold(f32::INFINITY, f32::min);
            let max = b.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if max == min { 0.0 } else { (max - min) / 31.0 }
        }).fold(0.0, f32::max);
        let max_min = block.chunks_exact(32).map(|b| -b.iter().copied().fold(f32::INFINITY, f32::min)).fold(0.0, f32::max);
        let scale_bytes = pack_k4_scales(
            &scales.try_into().map_err(|_| "invalid Q5_K scales")?,
            &mins.try_into().map_err(|_| "invalid Q5_K mins")?,
        );
        out.extend(f32_to_f16(max_scale / 63.0).to_le_bytes());
        out.extend(f32_to_f16(max_min / 63.0).to_le_bytes());
        out.extend(scale_bytes);
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 32];
        for group in 0..4 {
            for l in 0..32 {
                let i0 = group * 64 + l;
                let i1 = i0 + 32;
                ql[group * 32 + l] = (codes[i0] & 15) | ((codes[i1] & 15) << 4);
                if codes[i0] & 16 != 0 { qh[l] |= 1 << (group * 2); }
                if codes[i1] & 16 != 0 { qh[l] |= 2 << (group * 2); }
            }
        }
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
    }
    Ok(out)
}

fn quantize_q2_k(values: &[f32]) -> Result<Vec<u8>, String> {
    if values.len() % 256 != 0 { return Err("Q2_K requires tensor size divisible by 256".into()); }
    let mut out = Vec::with_capacity(values.len() / 256 * 84);
    for block in values.chunks_exact(256) {
        let mut scales = [0u8; 16];
        let mut mins = [0u8; 16];
        let mut codes = [0u8; 256];
        let mut max_scale = 0.0f32;
        let mut max_min = 0.0f32;
        for g in 0..16 {
            let b = &block[g * 16..g * 16 + 16];
            let min = b.iter().copied().fold(f32::INFINITY, f32::min);
            let max = b.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = (max - min) / 3.0;
            let offset = -min;
            max_scale = max_scale.max(scale);
            max_min = max_min.max(offset);
        }
        for g in 0..16 {
            let b = &block[g * 16..g * 16 + 16];
            let min = b.iter().copied().fold(f32::INFINITY, f32::min);
            let max = b.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = (max - min) / 3.0;
            let offset = -min;
            scales[g] = if max_scale == 0.0 { 0 } else { (scale / max_scale * 15.0).round().clamp(0.0, 15.0) as u8 };
            mins[g] = if max_min == 0.0 { 0 } else { (offset / max_min * 15.0).round().clamp(0.0, 15.0) as u8 };
            let d = max_scale / 15.0 * scales[g] as f32;
            let dm = max_min / 15.0 * mins[g] as f32;
            for i in 0..16 {
                codes[g * 16 + i] = if d == 0.0 { 0 } else { ((b[i] + dm) / d).round().clamp(0.0, 3.0) as u8 };
            }
        }
        out.extend(f32_to_f16(max_scale / 15.0).to_le_bytes());
        out.extend(f32_to_f16(max_min / 15.0).to_le_bytes());
        for g in 0..16 { out.push(scales[g] | (mins[g] << 4)); }
        for base in (0..256).step_by(128) {
            for l in 0..32 {
                out.push(codes[base + l] | (codes[base + l + 32] << 2) | (codes[base + l + 64] << 4) | (codes[base + l + 96] << 6));
            }
        }
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
                let tokens = values.iter().map(|v| match v { GgufValue::String(s) => s.clone(), _ => unreachable!() }).collect::<Vec<_>>();
                writer.add_tokens(key, &tokens);
            }
            _ => {}
        }
    }
}

pub fn quantize(input: impl AsRef<Path>, output: impl AsRef<Path>, kind: &str) -> Result<(), String> {
    if !matches!(kind, "q2_k" | "q4_0" | "q4_1" | "q4_k" | "q5_0" | "q5_1" | "q5_k" | "q8_0") {
        return Err("quantization must be q2_k, q4_0, q4_1, q4_k, q5_0, q5_1, q5_k, or q8_0".into());
    }
    let input = input.as_ref();
    let mut reader = GgufReader::open(input)?;
    let mut writer = Writer::new();
    copy_metadata(&reader, &mut writer);
    writer.add_meta_str("smaulnative.quantization", kind);

    let tensors = reader.tensors.clone();
    for info in tensors {
        let values = reader.tensor_f32(&info.name)?;
        let (dtype, data) = match kind {
            "q2_k" => (10, quantize_q2_k(&values)?),
            "q4_k" => (12, quantize_q4_k(&values)?),
            "q5_k" => (13, quantize_q5_k(&values)?),
            _ if values.len() % 32 == 0 => quantize_tensor(&values, kind)?,
            _ => {
                let mut data = Vec::with_capacity(values.len() * 2);
                for value in values { data.extend(f32_to_f16(value).to_le_bytes()); }
                (1, data)
            }
        };
        writer.tensor(Tensor { name: &info.name, shape: &info.shape, dtype, data: &data });
    }
    writer.finish(output.as_ref().to_str().ok_or("invalid output path")?)
}

pub fn quantize_cli(args: &[String]) -> Result<(), String> {
    if args.len() < 4 { return Err("usage: smaul-quantize-gguf <input.gguf> <output.gguf> --type q2_k|q4_0|q4_1|q4_k|q5_0|q5_1|q5_k|q8_0".into()); }
    let mut kind = None;
    let mut i = 3;
    while i < args.len() {
        if args[i] == "--type" && i + 1 < args.len() { kind = Some(args[i + 1].as_str()); i += 2; }
        else { return Err(format!("unknown argument: {}", args[i])); }
    }
    quantize(&args[1], &args[2], kind.ok_or("missing --type")?)
}
