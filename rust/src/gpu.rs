#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBackend { Cpu, Cuda, Hip }

pub fn backend() -> GpuBackend {
    if cfg!(feature = "cuda") { GpuBackend::Cuda }
    else if cfg!(feature = "hip") { GpuBackend::Hip }
    else { GpuBackend::Cpu }
}

pub fn packed_row_bytes(in_features: usize, bits: u8) -> Result<usize, String> {
    if bits != 2 && bits != 4 { return Err("low-bit backend supports 2-bit and 4-bit weights".into()); }
    Ok((in_features + (8 / bits as usize) - 1) / (8 / bits as usize))
}

pub fn decode_fp2(code: u8) -> f32 {
    match code & 3 { 0 => -1.0, 2 => 1.0, _ => 0.0 }
}

pub fn decode_fp4(code: u8) -> f32 {
    match code & 15 { 0 => -2.0, 1 => -1.0, 2 => -0.5, 3 => -0.25, 5 => 0.25, 6 => 0.5, 7 => 1.0, 8 => 2.0, _ => 0.0 }
}

pub fn packed_linear_cpu(x: &[f32], packed: &[u8], scale: &[f32], batch: usize, out_features: usize, in_features: usize, bits: u8) -> Result<Vec<f32>, String> {
    if x.len() != batch * in_features || scale.len() != in_features { return Err("low-bit linear shape mismatch".into()); }
    let row_bytes = packed_row_bytes(in_features, bits)?;
    if packed.len() < out_features * row_bytes { return Err("packed weight is too small".into()); }
    let mut out = vec![0.0; batch * out_features];
    for n in 0..batch {
        for o in 0..out_features {
            let row = &packed[o * row_bytes..(o + 1) * row_bytes];
            let mut sum = 0.0;
            for k in 0..in_features {
                let code = if bits == 2 { (row[k / 4] >> (6 - (k % 4) * 2)) & 3 } else { (row[k / 2] >> (4 - (k % 2) * 4)) & 15 };
                let q = if bits == 2 { decode_fp2(code) } else { decode_fp4(code) };
                sum += x[n * in_features + k] * scale[k] * q;
            }
            out[n * out_features + o] = sum;
        }
    }
    Ok(out)
}
