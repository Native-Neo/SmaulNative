use crate::rwkv_model::{RwkvModel, RwkvModelState};
use ndarray::Array2;

pub const QAT_3BIT_MIN: i8 = -4;
pub const QAT_3BIT_MAX: i8 = 3;
pub const FP8_E4M3_MAX: f32 = 448.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bits { Fp2, Fp3, Fp4, Fp8 }

impl Bits {
    pub fn from_bits(bits: u8) -> Result<Self, String> {
        match bits { 2 => Ok(Self::Fp2), 3 => Ok(Self::Fp3), 4 => Ok(Self::Fp4), 8 => Ok(Self::Fp8), _ => Err("bits must be 2, 3, 4, or 8".into()) }
    }
    pub fn bits(self) -> u8 { match self { Self::Fp2 => 2, Self::Fp3 => 3, Self::Fp4 => 4, Self::Fp8 => 8 } }
    /// Codebook indexed by the stored code. FP8 has none: it stores E4M3 bit patterns directly.
    pub fn levels(self) -> &'static [f32] {
        match self {
            Self::Fp2 => &[-1.0, 0.0, 1.0],
            Self::Fp3 => &[-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            Self::Fp4 => &[-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0],
            Self::Fp8 => &[],
        }
    }
    /// Largest positive codebook entry; a row is scaled so its peak magnitude lands on it.
    pub fn level_max(self) -> f32 { self.levels().iter().copied().fold(0.0, f32::max) }
    pub fn row_bytes(self, cols: usize) -> usize {
        match self { Self::Fp2 => cols.div_ceil(4), Self::Fp3 => (cols * 3).div_ceil(8), Self::Fp4 => cols.div_ceil(2), Self::Fp8 => cols }
    }
}

/// Encode to OCP E4M3: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits.
/// Exponent field 0 is subnormal, 0x7f/0xff are NaN, and the largest finite value is 448.
pub fn fp8_code(v: f32) -> u8 {
    if v.is_nan() { return 0x7f; }
    let s = if v.is_sign_negative() { 0x80u8 } else { 0 };
    let x = v.abs();
    if x >= FP8_E4M3_MAX { return s | 0x7e; }
    if x < 2f32.powi(-6) {
        // subnormal: value = m * 2^-9
        let m = (x * 2f32.powi(9)).round() as i32;
        if m >= 8 { return s | 0x08; }
        return s | m.clamp(0, 7) as u8;
    }
    let mut e = x.log2().floor() as i32;
    let mut m = ((x / 2f32.powi(e) - 1.0) * 8.0).round() as i32;
    if m >= 8 { m = 0; e += 1; }
    if e > 8 { return s | 0x7e; }
    s | (((e + 7) as u8) << 3) | (m.clamp(0, 7) as u8)
}

pub fn fp8_value(c: u8) -> f32 {
    if c & 0x7f == 0x7f { return f32::NAN; }
    let s = if c & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((c >> 3) & 0x0f) as i32;
    let m = (c & 7) as f32;
    if e == 0 { return s * m * 2f32.powi(-9); }
    s * 2f32.powi(e - 7) * (1.0 + m / 8.0)
}

pub fn quantize_codes(x: &Array2<f32>, b: Bits) -> (Array2<u8>, Vec<f32>) {
    let levels = b.levels();
    assert!(!levels.is_empty(), "FP8 stores E4M3 codes directly and has no codebook");
    let level_max = b.level_max();
    let mut codes = Array2::zeros(x.raw_dim());
    let mut scales = vec![0.0; x.nrows()];
    for r in 0..x.nrows() {
        let max_abs = x.row(r).iter().map(|v| v.abs()).fold(0.0, f32::max);
        scales[r] = (max_abs / level_max).max(f32::EPSILON);
        for j in 0..x.ncols() {
            let v = x[[r, j]] / scales[r];
            let mut best = 0;
            let mut distance = f32::INFINITY;
            for (i, level) in levels.iter().enumerate() {
                let d = (v - level).abs();
                if d < distance { distance = d; best = i; }
            }
            codes[[r, j]] = best as u8;
        }
    }
    (codes, scales)
}

pub fn dequantize(codes: &Array2<u8>, scales: &[f32], b: Bits) -> Array2<f32> {
    let levels = b.levels();
    assert_eq!(codes.nrows(), scales.len());
    Array2::from_shape_fn(codes.raw_dim(), |(r, j)| levels[codes[[r, j]] as usize] * scales[r])
}

pub fn fake_quantize(x: &Array2<f32>, b: Bits) -> Array2<f32> {
    match b {
        Bits::Fp8 => x.mapv(|v| fp8_value(fp8_code(v))),
        _ => { let (codes, scales) = quantize_codes(x, b); dequantize(&codes, &scales, b) }
    }
}

pub fn quantized_linear(x: &Array2<f32>, w: &Array2<f32>, b: Bits) -> Array2<f32> { x.dot(&fake_quantize(w, b).t()) }

pub fn pack_3bit(codes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity((codes.len() * 3).div_ceil(8));
    let (mut acc, mut used) = (0u8, 0u32);
    for &code in codes {
        for shift in [2, 1, 0] {
            acc |= ((code & 7) >> shift & 1) << (7 - used);
            used += 1;
            if used == 8 { out.push(acc); acc = 0; used = 0; }
        }
    }
    if used > 0 { out.push(acc); }
    out
}

pub fn unpack_3bit(packed: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n * 3 {
        let bit = (packed[i / 8] >> (7 - i % 8)) & 1;
        match i % 3 { 0 => out.push(bit << 2), 1 => *out.last_mut().unwrap() |= bit << 1, _ => *out.last_mut().unwrap() |= bit }
    }
    out
}

pub fn pack_lowbit(codes: &[u8], b: Bits) -> Vec<u8> {
    match b {
        Bits::Fp2 => {
            let mut out = Vec::with_capacity((codes.len() * 2).div_ceil(8));
            let (mut acc, mut used) = (0u8, 0u32);
            for &code in codes {
                acc |= (code & 3) << (6 - used);
                used += 2;
                if used == 8 { out.push(acc); acc = 0; used = 0; }
            }
            if used > 0 { out.push(acc); }
            out
        }
        Bits::Fp3 => pack_3bit(codes),
        Bits::Fp4 => codes.chunks(2).map(|x| ((x[0] & 15) << 4) | (x.get(1).copied().unwrap_or(0) & 15)).collect(),
        Bits::Fp8 => codes.to_vec(),
    }
}

pub fn unpack_lowbit(packed: &[u8], n: usize, b: Bits) -> Vec<u8> {
    match b {
        Bits::Fp2 => (0..n).map(|i| (packed[i / 4] >> (6 - 2 * (i % 4))) & 3).collect(),
        Bits::Fp3 => unpack_3bit(packed, n),
        Bits::Fp4 => (0..n).map(|i| if i % 2 == 0 { packed[i / 2] >> 4 & 15 } else { packed[i / 2] & 15 }).collect(),
        Bits::Fp8 => packed[..n].to_vec(),
    }
}

fn pack_lowbit_rows(codes: &Array2<u8>, b: Bits) -> Vec<u8> {
    let mut out = Vec::with_capacity(codes.nrows() * b.row_bytes(codes.ncols()));
    for row in codes.rows() { out.extend(pack_lowbit(row.as_standard_layout().as_slice().unwrap(), b)); }
    out
}

fn unpack_lowbit_rows(packed: &[u8], shape: (usize, usize), b: Bits) -> Vec<u8> {
    let (rows, cols) = shape;
    let bytes_per_row = b.row_bytes(cols);
    assert!(packed.len() >= rows * bytes_per_row, "packed weight is too small for {rows}x{cols} at {} bits", b.bits());
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows { out.extend(unpack_lowbit(&packed[r * bytes_per_row..(r + 1) * bytes_per_row], cols, b)); }
    out
}

pub fn quantize_weight(w: &Array2<f32>, b: Bits) -> (Vec<u8>, Vec<f32>) {
    match b {
        Bits::Fp8 => (w.iter().map(|&v| fp8_code(v)).collect(), vec![1.0; w.nrows()]),
        _ => { let (codes, scales) = quantize_codes(w, b); (pack_lowbit_rows(&codes, b), scales) }
    }
}

pub fn dequantize_weight(packed: &[u8], scales: &[f32], shape: (usize, usize), b: Bits) -> Array2<f32> {
    match b {
        Bits::Fp8 => Array2::from_shape_vec(shape, (0..shape.0 * shape.1).map(|i| fp8_value(packed[i])).collect()).unwrap(),
        _ => {
            assert_eq!(scales.len(), shape.0);
            let codes = unpack_lowbit_rows(packed, shape, b);
            let levels = b.levels();
            Array2::from_shape_fn(shape, |(r, j)| levels[codes[r * shape.1 + j] as usize] * scales[r])
        }
    }
}

pub fn quantize_weight_3bit(w: &Array2<f32>) -> (Vec<u8>, Vec<f32>) { quantize_weight(w, Bits::Fp3) }
pub fn dequantize_weight_3bit(packed: &[u8], scales: &[f32], shape: (usize, usize)) -> Array2<f32> { dequantize_weight(packed, scales, shape, Bits::Fp3) }

#[derive(Clone, Debug)]
pub struct QatLinear { pub weight: Array2<f32>, pub bits: Bits }
impl QatLinear {
    pub fn new(weight: Array2<f32>, bits: Bits) -> Self { Self { weight, bits } }
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> { quantized_linear(x, &self.weight, self.bits) }
    pub fn convert(&self) -> PackedLinear { let (packed, scales) = quantize_weight(&self.weight, self.bits); PackedLinear { packed, scales, shape: self.weight.dim(), bits: self.bits } }
}

#[derive(Clone, Debug)]
pub struct PackedLinear { pub packed: Vec<u8>, pub scales: Vec<f32>, pub shape: (usize, usize), pub bits: Bits }
impl PackedLinear { pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> { let weight = dequantize_weight(&self.packed, &self.scales, self.shape, self.bits); x.dot(&weight.t()) } }

#[derive(Clone, Debug)]
pub struct QatLinear3Bit { pub weight: Array2<f32>, pub signed_activation: bool, pub activation_min: f32, pub activation_max: f32, pub momentum: f32 }
impl QatLinear3Bit {
    pub fn new(weight: Array2<f32>, signed_activation: bool) -> Self { Self { weight, signed_activation, activation_min: 0.0, activation_max: 0.0, momentum: 0.1 } }
    pub fn observe(&mut self, x: &Array2<f32>) { if x.is_empty() { return; } self.activation_min = x.iter().copied().fold(f32::INFINITY, f32::min); self.activation_max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max); }
    pub fn fake_quantize_activation(&self, x: &Array2<f32>) -> Array2<f32> { let max_abs = self.activation_min.abs().max(self.activation_max.abs()); let scale = (max_abs / Bits::Fp3.level_max()).max(f32::EPSILON); x.mapv(|v| (v / scale).round().clamp(QAT_3BIT_MIN as f32, QAT_3BIT_MAX as f32) * scale) }
    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> { self.observe(x); let (packed, scales) = quantize_weight_3bit(&self.weight); let weight = dequantize_weight_3bit(&packed, &scales, self.weight.dim()); self.fake_quantize_activation(x).dot(&weight.t()) }
    pub fn convert(&self) -> QuantizedLinear3Bit { let (packed, scales) = quantize_weight_3bit(&self.weight); QuantizedLinear3Bit { packed, scales, shape: self.weight.dim() } }
}

#[derive(Clone, Debug)]
pub struct QuantizedLinear3Bit { pub packed: Vec<u8>, pub scales: Vec<f32>, pub shape: (usize, usize) }
impl QuantizedLinear3Bit { pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> { x.dot(&dequantize_weight_3bit(&self.packed, &self.scales, self.shape).t()) } }

#[cfg(test)]
mod tests_qat {
    use super::*;
    use ndarray::array;
    #[test] fn bit_modes_are_supported() { for b in [2u8, 3, 4, 8] { assert_eq!(Bits::from_bits(b).unwrap().bits(), b); } assert!(Bits::from_bits(5).is_err()); assert!(Bits::from_bits(0).is_err()); }
    #[test] fn fp8_codes_round_trip_through_values() { for c in 0u8..=255 { if c & 0x7f == 0x7f { assert!(fp8_value(c).is_nan()); continue; } let v = fp8_value(c); assert!(v.is_finite()); assert_eq!(fp8_code(v), c, "code {c:#04x} decoded to {v} and re-encoded differently"); } }
    #[test] fn fp8_encodes_subnormals_instead_of_inflating_them() { // 1e-8 used to snap up to 2^-6
        assert_eq!(fp8_value(fp8_code(1e-8)), 0.0);
        let smallest = 2f32.powi(-9); assert_eq!(fp8_value(fp8_code(smallest)), smallest);
        assert_eq!(fp8_value(fp8_code(3.0 * smallest)), 3.0 * smallest);
        assert_eq!(fp8_value(fp8_code(1e30)), 448.0); assert_eq!(fp8_value(fp8_code(-1e30)), -448.0); }
    #[test] fn fp8_decoding_is_monotonic_in_magnitude() { let mut prev = -1.0; for c in 0u8..0x7f { let v = fp8_value(c); assert!(v > prev, "code {c:#04x} broke monotonicity"); prev = v; } }
    #[test] fn low_bit_scales_use_the_whole_codebook() { let w = array![[1.0, -0.6, 0.1, -1.0]];
        for b in [Bits::Fp2, Bits::Fp3, Bits::Fp4] { let (codes, scales) = quantize_codes(&w, b); let peak = codes.iter().map(|&c| b.levels()[c as usize].abs()).fold(0.0, f32::max);
            assert_eq!(peak, b.level_max(), "{b:?} never reaches its outermost level"); assert!((scales[0] - 1.0 / b.level_max()).abs() < 1e-6); } }
    #[test] fn finer_formats_quantize_more_accurately() { let w = Array2::from_shape_fn((4, 16), |(r, c)| ((r * 16 + c) as f32 * 0.37).sin());
        let err = |b| (&fake_quantize(&w, b) - &w).iter().map(|v| v * v).sum::<f32>();
        let (e2, e3, e4) = (err(Bits::Fp2), err(Bits::Fp3), err(Bits::Fp4));
        assert!(e3 < e2, "3-bit ({e3}) should beat 2-bit ({e2})"); assert!(e4 < e2, "4-bit ({e4}) should beat 2-bit ({e2})"); }
    #[test] fn packed_round_trip_for_every_low_bit_format() { for b in [Bits::Fp2, Bits::Fp3, Bits::Fp4] { for cols in [1usize, 3, 7, 8, 11, 16] {
        let w = Array2::from_shape_fn((3, cols), |(r, c)| ((r * 13 + c * 7) as f32 * 0.21).cos());
        let (packed, scales) = quantize_weight(&w, b);
        assert_eq!(packed.len(), 3 * b.row_bytes(cols), "{b:?} packed size wrong for {cols} columns");
        assert_eq!(dequantize_weight(&packed, &scales, w.dim(), b), fake_quantize(&w, b), "{b:?} packing changed values at {cols} columns"); } } }
    #[test] fn three_bit_uses_the_signed_int3_range() { let w = array![[3.0, -3.0, 0.0, 1.0]]; let q = fake_quantize(&w, Bits::Fp3);
        assert_eq!(q, w, "an exactly representable int3 row must survive unchanged");
        assert_eq!(QAT_3BIT_MIN as f32, Bits::Fp3.levels()[0]); assert_eq!(QAT_3BIT_MAX as f32, Bits::Fp3.level_max()); }
    #[test] fn fp2_round_trip_uses_three_levels() { let x = array![[-2.0, 0.0], [0.5, 2.0]]; let q = fake_quantize(&x, Bits::Fp2); assert_eq!(q[[0, 0]], -2.0); assert_eq!(q[[0, 1]], 0.0); assert_eq!(q[[1, 0]], 0.0); assert_eq!(q[[1, 1]], 2.0); }
    #[test] fn fp4_quantization_has_expected_shape() { let x = array![[1.0, -0.5, 0.2]]; assert_eq!(fake_quantize(&x, Bits::Fp4).dim(), x.dim()); }
    #[test] fn fp8_quantization_is_finite() { assert!(fake_quantize(&array![[1.0, -0.5, 0.2]], Bits::Fp8).iter().all(|v| v.is_finite())); }
    #[test] fn quantized_linear_has_expected_shape() { let x = array![[1.0, 2.0, 3.0]]; let w = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]; assert_eq!(quantized_linear(&x, &w, Bits::Fp4).shape(), &[1, 2]); }
    #[test] fn pack_3bit_round_trips() { let codes = vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 7, 3]; assert_eq!(unpack_3bit(&pack_3bit(&codes), codes.len()), codes); }
    #[test] fn packed_linear_matches_dequantized_weight() { let x = array![[1.0, 2.0, 3.0]]; let w = array![[1.0, -0.5, 0.25], [-0.75, 0.5, 1.0]]; let qat = QatLinear::new(w.clone(), Bits::Fp4); let packed = qat.convert(); let expected = x.dot(&dequantize_weight(&packed.packed, &packed.scales, w.dim(), Bits::Fp4).t()); assert_eq!(packed.forward(&x), expected); }
    #[test] fn rowwise_packing_preserves_per_row_scales() { let w = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]; for b in [Bits::Fp2, Bits::Fp4] { let (packed, scales) = quantize_weight(&w, b); let restored = dequantize_weight(&packed, &scales, w.dim(), b); assert_eq!(restored, fake_quantize(&w, b)); } }
}


// ===== rqt =====

#[derive(Clone,Debug)]
pub struct RealQuantLinear { pub weight:Array2<f32>, pub bias:Option<Vec<f32>>, pub bits:Bits, pub quant:PackedLinear }
impl RealQuantLinear {
 pub fn new(weight:Array2<f32>,bias:Option<Vec<f32>>,bits:u8)->Result<Self,String>{let bits=Bits::from_bits(bits)?;let quant=QatLinear::new(weight.clone(),bits).convert();Ok(Self{weight,bias,bits,quant})}
 pub fn refresh(&mut self){self.quant=QatLinear::new(self.weight.clone(),self.bits).convert();}
 pub fn dequantized(&self)->Array2<f32>{dequantize_weight(&self.quant.packed,&self.quant.scales,self.quant.shape,self.bits)}
 pub fn forward(&self,input:&Array2<f32>)->Array2<f32>{let input=input.as_standard_layout();let values=crate::gpu::packed_linear(input.as_slice().expect("standard layout input"),&self.quant.packed,&self.quant.scales,input.nrows(),self.quant.shape.0,self.quant.shape.1,self.bits.bits()).unwrap();let mut out=Array2::from_shape_vec((input.nrows(),self.quant.shape.0),values).unwrap();if let Some(b)=&self.bias{assert_eq!(b.len(),out.ncols());for r in 0..out.nrows(){for c in 0..out.ncols(){out[[r,c]]+=b[c];}}}out}
 pub fn backward(&self,input:&Array2<f32>,grad:&Array2<f32>)->(Array2<f32>,Array2<f32>){(grad.dot(&self.dequantized()),grad.t().dot(input))}
}

#[cfg(test)]
mod tests_rqt{use super::*;use ndarray::array;#[test]fn packed_training_roundtrip(){let l=RealQuantLinear::new(array![[1.0,0.0],[0.0,1.0]],None,4).unwrap();let y=l.forward(&array![[2.0,3.0]]);assert_eq!(y.shape(),&[1,2]);assert_eq!(l.dequantized().shape(),&[2,2]);}#[test]fn supported_bits(){for b in [2,3,4,8]{assert!(RealQuantLinear::new(array![[1.0,0.0],[0.0,1.0]],None,b).is_ok())}assert!(RealQuantLinear::new(array![[1.0]],None,5).is_err());}
#[test]fn forward_accepts_non_contiguous_input(){let l=RealQuantLinear::new(array![[1.0,0.0],[0.0,1.0]],None,4).unwrap();let x=array![[2.0,3.0],[4.0,5.0]];let y=l.forward(&x.t().to_owned());assert_eq!(y.shape(),&[2,2]);let view=x.slice(ndarray::s![..;1,..]).to_owned();assert_eq!(l.forward(&view).shape(),&[2,2]);}#[test]fn direct_fp8_forward(){let l=RealQuantLinear::new(array![[1.0,0.0]],None,8).unwrap();let y=l.forward(&array![[2.0,3.0]]);assert_eq!(y.shape(),&[1,1]);assert!(y[[0,0]].is_finite());}}


// ===== rqt_model =====

/// Real Quantization Training: the model keeps full-precision master weights and
/// every forward pass runs on the quantized values. The masters are restored
/// afterwards so the optimizer keeps updating full precision, which is what lets
/// updates smaller than a quantization step accumulate instead of being erased.
pub struct RqtModel { pub model: RwkvModel, pub bits: u8 }

/// Visits every weight matrix RQT quantizes. `io` marks nn.Linear-style weights,
/// which are stored transposed relative to the row-wise quantization grid.
fn visit_linears(model: &mut RwkvModel, mut f: impl FnMut(&mut Array2<f32>, bool)) {
    f(&mut model.head.weight, false);
    for b in &mut model.rwkv_blocks {
        let t = &mut b.time_mix;
        for w in [&mut t.w1, &mut t.w2, &mut t.a1, &mut t.a2, &mut t.g1, &mut t.g2, &mut t.receptance, &mut t.key, &mut t.value, &mut t.output] { f(w, true); }
        if let Some(v) = &mut t.v1 { f(v, true); }
        if let Some(v) = &mut t.v2 { f(v, true); }
        f(&mut b.cmix.key, true); f(&mut b.cmix.value, true);
        if let Some(m) = &mut b.moe { f(&mut m.router.weight, false); for e in &mut m.experts { f(&mut e.key, true); f(&mut e.value, true); } }
    }
    for b in &mut model.moba_blocks {
        f(&mut b.att.receptance.weight, false); f(&mut b.att.key.weight, false); f(&mut b.att.value.weight, false); f(&mut b.att.output.weight, false);
        f(&mut b.ffn.key, true); f(&mut b.ffn.value, true);
    }
}

fn refresh_packed(model: &mut RwkvModel) { for b in &mut model.rwkv_blocks { b.time_mix.refresh_quantized(); } }

/// Replaces every weight with its quantized value and returns the full-precision
/// masters. Pair with `restore_masters` around a forward/backward pass.
pub fn quantize_in_place(model: &mut RwkvModel, bits: u8) -> Result<Vec<Array2<f32>>,String> {
    if !matches!(bits,2|3|4|8) { return Err("RQT supports 2, 3, 4, or 8 bits".into()) }
    let mut masters = Vec::new();
    let mut error = None;
    visit_linears(model, |w, io| {
        if error.is_some() { return; }
        masters.push(w.clone());
        let source = if io { w.t().to_owned() } else { w.clone() };
        match RealQuantLinear::new(source, None, bits) {
            Ok(l) => { let d = l.dequantized(); *w = if io { d.t().to_owned() } else { d }; }
            Err(e) => error = Some(e),
        }
    });
    match error { Some(e) => { restore_masters(model, masters); Err(e) } None => { refresh_packed(model); Ok(masters) } }
}

pub fn restore_masters(model: &mut RwkvModel, masters: Vec<Array2<f32>>) {
    let mut it = masters.into_iter();
    visit_linears(model, |w, _| { if let Some(m) = it.next() { *w = m; } });
    refresh_packed(model);
}

impl RqtModel {
    pub fn new(model: RwkvModel, bits: u8) -> Result<Self,String> { if !matches!(bits,2|3|4|8){return Err("RQT supports 2, 3, 4, or 8 bits".into())} Ok(Self{model,bits}) }

    pub fn take_masters(&mut self) -> Result<Vec<Array2<f32>>,String> { quantize_in_place(&mut self.model, self.bits) }
    pub fn restore_masters(&mut self, masters: Vec<Array2<f32>>) { restore_masters(&mut self.model, masters) }

    /// Runs `f` with quantized weights in place, then restores the masters.
    fn with_quantized<T>(&mut self, f: impl FnOnce(&mut RwkvModel) -> T) -> Result<T,String> {
        let masters = self.take_masters()?;
        let out = f(&mut self.model);
        self.restore_masters(masters);
        Ok(out)
    }

    /// Permanently collapses the masters onto the quantization grid. Use at export time.
    pub fn freeze_quantized(&mut self) -> Result<(),String> { self.take_masters()?; Ok(()) }

    pub fn forward(&mut self,tokens:&[usize],state:Option<&RwkvModelState>)->Result<(Array2<f32>,RwkvModelState),String>{self.with_quantized(|m|m.forward(tokens,state))}
    pub fn forward_with_tape(&mut self,tokens:&[usize])->Result<(Array2<f32>,crate::model_backward::ModelBackwardTape),String>{self.with_quantized(|m|m.forward_with_tape(tokens))}
    pub fn forward_with_state_and_tape(&mut self,tokens:&[usize],state:Option<&RwkvModelState>)->Result<(Array2<f32>,crate::model_backward::ModelBackwardTape,RwkvModelState),String>{self.with_quantized(|m|m.forward_with_tape_and_state(tokens,state))}
    pub fn parameter_count(&self)->usize{self.model.parameter_count()}
}

#[cfg(test)]
mod tests_rqt_model {
    use super::*; use crate::rwkv_model::RwkvModelConfig;
    fn model() -> RwkvModel { RwkvModel::new(RwkvModelConfig::new(32, 16, 2, 4).with_moe(true, 2, 1), 7) }

    #[test] fn quantizes_every_linear_path(){let mut q=RqtModel::new(model(),4).unwrap();let(y,_)=q.forward(&[1,2,3],None).unwrap();assert_eq!(y.dim(),(3,32));assert!(y.iter().all(|v|v.is_finite()));}
    #[test] fn supports_2_3_4_8_bits(){for bits in [2,3,4,8]{assert!(RqtModel::new(model(),bits).is_ok())}assert!(RqtModel::new(model(),5).is_err());}

    #[test] fn forward_preserves_full_precision_masters(){
        let mut q=RqtModel::new(model(),2).unwrap();
        let before=q.model.head.weight.clone();
        q.forward(&[1,2,3],None).unwrap();
        assert_eq!(q.model.head.weight,before,"RQT forward must not overwrite the master weights");
    }

    #[test] fn sub_step_updates_accumulate_in_the_masters(){
        // A 2-bit grid is coarse, so a small update is invisible in the quantized
        // weights but must still survive in the masters and eventually cross a level.
        let mut q=RqtModel::new(model(),2).unwrap();
        let start=q.model.head.weight.clone();
        for _ in 0..200 { q.forward(&[1,2],None).unwrap(); q.model.head.weight.mapv_inplace(|v|v+1e-3); }
        let expected=start.mapv(|v|v+0.2);
        let drift=(&q.model.head.weight-&expected).iter().map(|v|v.abs()).fold(0.0f32,f32::max);
        assert!(drift<1e-4,"200 sub-quantization-step updates drifted by {drift}; the masters are being rewritten");
    }

    #[test] fn forward_actually_uses_quantized_weights(){
        let reference=model().forward(&[1,2,3],None).0;
        let mut q=RqtModel::new(model(),2).unwrap();
        let (y,_)=q.forward(&[1,2,3],None).unwrap();
        let diff=y.iter().zip(reference.iter()).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
        assert!(diff>1e-4,"2-bit RQT logits matched full precision exactly ({diff}); nothing was quantized");
    }

    #[test] fn freeze_collapses_masters_onto_the_grid(){
        let mut q=RqtModel::new(model(),2).unwrap();
        q.freeze_quantized().unwrap();
        let frozen=q.model.head.weight.clone();
        q.freeze_quantized().unwrap();
        assert_eq!(q.model.head.weight,frozen,"quantization must be idempotent once frozen");
    }
}
