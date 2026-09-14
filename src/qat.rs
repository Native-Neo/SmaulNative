use ndarray::Array2;

pub const QAT_3BIT_MIN: i8 = -4;
pub const QAT_3BIT_MAX: i8 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bits {
    Fp2,
    Fp4,
    Fp8,
}

impl Bits {
    pub fn from_bits(bits: u8) -> Result<Self, String> {
        match bits {
            2 => Ok(Self::Fp2),
            4 => Ok(Self::Fp4),
            8 => Ok(Self::Fp8),
            _ => Err("bits must be 2, 4, or 8".into()),
        }
    }

    pub fn bits(self) -> u8 {
        match self {
            Self::Fp2 => 2,
            Self::Fp4 => 4,
            Self::Fp8 => 8,
        }
    }

    fn levels(self) -> &'static [f32] {
        match self {
            Self::Fp2 => &[-1.0, 0.0, 1.0],
            Self::Fp4 => &[-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0],
            Self::Fp8 => &[],
        }
    }
}

fn fp8_code(v: f32) -> u8 {
    const M: f32 = 448.0;
    if v.is_nan() {
        return 0x7f;
    }
    if v == 0.0 {
        return if v.is_sign_negative() { 0x80 } else { 0 };
    }
    let s = if v.is_sign_negative() { 0x80 } else { 0 };
    let x = v.abs();
    if x.is_infinite() || x >= M {
        return s | 0x7e;
    }
    let mut e = x.log2().floor() as i32;
    if e < -6 {
        e = -6;
    }
    if e > 8 {
        e = 8;
    }
    let mut m = ((x / 2f32.powi(e) - 1.0) * 8.0).round() as i32;
    if m >= 8 {
        m = 0;
        e += 1;
    }
    if e > 8 || e == 8 && m > 6 {
        return s | 0x7e;
    }
    s | (((e + 7) as u8) << 3) | (m.clamp(0, 7) as u8)
}

fn fp8_value(c: u8) -> f32 {
    if c == 0x7f || c == 0xff {
        return f32::NAN;
    }
    if c & 0x7f == 0 {
        return if c & 0x80 != 0 { -0.0 } else { 0.0 };
    }
    let s = if c & 0x80 != 0 { -1.0 } else { 1.0 };
    s * 2f32.powi(((c >> 3) & 15) as i32 - 7) * (1.0 + (c & 7) as f32 / 8.0)
}

pub fn quantize_codes(x: &Array2<f32>, b: Bits) -> (Array2<u8>, Vec<f32>) {
    let levels = b.levels();
    let mut codes = Array2::zeros(x.raw_dim());
    let mut scales = vec![0.0; x.nrows()];
    for r in 0..x.nrows() {
        let max_abs = x.row(r).iter().map(|v| v.abs()).fold(0.0, f32::max);
        scales[r] = max_abs.max(f32::EPSILON);
        for j in 0..x.ncols() {
            let v = x[[r, j]] / scales[r];
            let mut best = 0;
            let mut distance = f32::INFINITY;
            for (i, level) in levels.iter().enumerate() {
                let d = (v - level).abs();
                if d < distance {
                    distance = d;
                    best = i;
                }
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
        Bits::Fp8 => x.mapv(|v| if v == 0.0 { v } else { fp8_value(fp8_code(v)) }),
        Bits::Fp2 | Bits::Fp4 => {
            let (codes, scales) = quantize_codes(x, b);
            dequantize(&codes, &scales, b)
        }
    }
}

pub fn quantized_linear(x: &Array2<f32>, w: &Array2<f32>, b: Bits) -> Array2<f32> {
    x.dot(&fake_quantize(w, b).t())
}

pub fn pack_3bit(codes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity((codes.len() * 3 + 7) / 8);
    let (mut acc, mut used) = (0, 0);
    for &code in codes {
        for shift in [2, 1, 0] {
            acc |= ((code & 7) >> shift) << (7 - used);
            used += 1;
            if used == 8 {
                out.push(acc);
                acc = 0;
                used = 0;
            }
        }
    }
    if used > 0 {
        out.push(acc);
    }
    out
}

pub fn unpack_3bit(packed: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n * 3 {
        let bit = (packed[i / 8] >> (7 - i % 8)) & 1;
        match i % 3 {
            0 => out.push(bit << 2),
            1 => *out.last_mut().unwrap() |= bit << 1,
            _ => *out.last_mut().unwrap() |= bit,
        }
    }
    out
}

pub fn pack_lowbit(codes: &[u8], b: Bits) -> Vec<u8> {
    match b {
        Bits::Fp2 => {
            let mut out = Vec::with_capacity((codes.len() * 2 + 7) / 8);
            let (mut acc, mut used) = (0, 0);
            for &code in codes {
                acc |= (code & 3) << (6 - used);
                used += 2;
                if used == 8 {
                    out.push(acc);
                    acc = 0;
                    used = 0;
                }
            }
            if used > 0 {
                out.push(acc);
            }
            out
        }
        Bits::Fp4 => codes
            .chunks(2)
            .map(|x| ((x[0] & 15) << 4) | (x.get(1).copied().unwrap_or(0) & 15))
            .collect(),
        Bits::Fp8 => codes.to_vec(),
    }
}

pub fn unpack_lowbit(packed: &[u8], n: usize, b: Bits) -> Vec<u8> {
    match b {
        Bits::Fp2 => (0..n).map(|i| (packed[i / 4] >> (6 - 2 * (i % 4))) & 3).collect(),
        Bits::Fp4 => (0..n)
            .map(|i| if i % 2 == 0 { packed[i / 2] >> 4 & 15 } else { packed[i / 2] & 15 })
            .collect(),
        Bits::Fp8 => packed[..n].to_vec(),
    }
}

pub fn quantize_weight(w: &Array2<f32>, b: Bits) -> (Vec<u8>, Vec<f32>) {
    match b {
        Bits::Fp8 => (w.iter().map(|&v| fp8_code(v)).collect(), vec![1.0]),
        Bits::Fp2 | Bits::Fp4 => {
            let (codes, scales) = quantize_codes(w, b);
            (pack_lowbit(codes.as_slice().unwrap(), b), scales)
        }
    }
}

pub fn dequantize_weight(
    packed: &[u8],
    scales: &[f32],
    shape: (usize, usize),
    b: Bits,
) -> Array2<f32> {
    match b {
        Bits::Fp8 => Array2::from_shape_vec(
            shape,
            (0..shape.0 * shape.1).map(|i| fp8_value(packed[i])).collect(),
        )
        .unwrap(),
        Bits::Fp2 | Bits::Fp4 => {
            assert_eq!(scales.len(), shape.0);
            let codes = unpack_lowbit(packed, shape.0 * shape.1, b);
            let levels = b.levels();
            Array2::from_shape_fn(shape, |(r, j)| levels[codes[r * shape.1 + j] as usize] * scales[r])
        }
    }
}

pub fn quantize_weight_3bit(w: &Array2<f32>) -> (Vec<u8>, Vec<f32>) {
    let mut codes = Vec::with_capacity(w.len());
    let mut scales = Vec::with_capacity(w.nrows());
    for row in w.rows() {
        let max_abs = row.iter().map(|v| v.abs()).fold(0.0, f32::max);
        let scale = (max_abs / 3.0).max(f32::EPSILON);
        scales.push(scale);
        for &v in row {
            codes.push(((v / scale).round().clamp(-4.0, 3.0) as i8 + 4) as u8);
        }
    }
    (pack_3bit(&codes), scales)
}

pub fn dequantize_weight_3bit(
    packed: &[u8],
    scales: &[f32],
    shape: (usize, usize),
) -> Array2<f32> {
    let (rows, cols) = shape;
    assert_eq!(scales.len(), rows);
    let codes = unpack_3bit(packed, rows * cols);
    Array2::from_shape_fn(shape, |(r, c)| ((codes[r * cols + c] as i8 - 4) as f32) * scales[r])
}

#[derive(Clone, Debug)]
pub struct QatLinear {
    pub weight: Array2<f32>,
    pub bits: Bits,
}

impl QatLinear {
    pub fn new(weight: Array2<f32>, bits: Bits) -> Self {
        Self { weight, bits }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        quantized_linear(x, &self.weight, self.bits)
    }

    pub fn convert(&self) -> PackedLinear {
        let (packed, scales) = quantize_weight(&self.weight, self.bits);
        PackedLinear { packed, scales, shape: self.weight.dim(), bits: self.bits }
    }
}

#[derive(Clone, Debug)]
pub struct PackedLinear {
    pub packed: Vec<u8>,
    pub scales: Vec<f32>,
    pub shape: (usize, usize),
    pub bits: Bits,
}

impl PackedLinear {
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let weight = dequantize_weight(&self.packed, &self.scales, self.shape, self.bits);
        x.dot(&weight.t())
    }
}

#[derive(Clone, Debug)]
pub struct QatLinear3Bit {
    pub weight: Array2<f32>,
    pub signed_activation: bool,
    pub activation_min: f32,
    pub activation_max: f32,
    pub momentum: f32,
}

impl QatLinear3Bit {
    pub fn new(weight: Array2<f32>, signed_activation: bool) -> Self {
        Self {
            weight,
            signed_activation,
            activation_min: 0.0,
            activation_max: 0.0,
            momentum: 0.1,
        }
    }

    pub fn observe(&mut self, x: &Array2<f32>) {
        if x.is_empty() {
            return;
        }
        self.activation_min = x.iter().copied().fold(f32::INFINITY, f32::min);
        self.activation_max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    }

    pub fn fake_quantize_activation(&self, x: &Array2<f32>) -> Array2<f32> {
        let max_abs = self.activation_min.abs().max(self.activation_max.abs());
        let scale = (max_abs / 3.0).max(f32::EPSILON);
        x.mapv(|v| (v / scale).round().clamp(-4.0, 3.0) * scale)
    }

    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        self.observe(x);
        let (packed, scales) = quantize_weight_3bit(&self.weight);
        let weight = dequantize_weight_3bit(&packed, &scales, self.weight.dim());
        self.fake_quantize_activation(x).dot(&weight.t())
    }

    pub fn convert(&self) -> QuantizedLinear3Bit {
        let (packed, scales) = quantize_weight_3bit(&self.weight);
        QuantizedLinear3Bit { packed, scales, shape: self.weight.dim() }
    }
}

#[derive(Clone, Debug)]
pub struct QuantizedLinear3Bit {
    pub packed: Vec<u8>,
    pub scales: Vec<f32>,
    pub shape: (usize, usize),
}

impl QuantizedLinear3Bit {
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        x.dot(&dequantize_weight_3bit(&self.packed, &self.scales, self.shape).t())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn bit_modes_are_supported() {
        assert_eq!(Bits::from_bits(2).unwrap().bits(), 2);
        assert_eq!(Bits::from_bits(4).unwrap().bits(), 4);
        assert_eq!(Bits::from_bits(8).unwrap().bits(), 8);
        assert!(Bits::from_bits(3).is_err());
    }

    #[test]
    fn fp2_round_trip_uses_three_levels() {
        let x = array![[-2.0, 0.0], [0.5, 2.0]];
        let q = fake_quantize(&x, Bits::Fp2);
        assert_eq!(q[[0, 0]], -2.0);
        assert_eq!(q[[0, 1]], 0.0);
        assert_eq!(q[[1, 0]], 0.0);
        assert_eq!(q[[1, 1]], 2.0);
    }

    #[test]
    fn fp4_quantization_has_expected_shape() {
        let x = array![[1.0, -0.5, 0.2]];
        assert_eq!(fake_quantize(&x, Bits::Fp4).dim(), x.dim());
    }

    #[test]
    fn fp8_quantization_is_finite() {
        assert!(fake_quantize(&array![[1.0, -0.5, 0.2]], Bits::Fp8).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn quantized_linear_has_expected_shape() {
        let x = array![[1.0, 2.0, 3.0]];
        let w = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert_eq!(quantized_linear(&x, &w, Bits::Fp4).shape(), &[1, 2]);
    }

    #[test]
    fn pack_3bit_round_trips() {
        let codes = vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 7, 3];
        assert_eq!(unpack_3bit(&pack_3bit(&codes), codes.len()), codes);
    }

    #[test]
    fn packed_linear_matches_dequantized_weight() {
        let x = array![[1.0, 2.0, 3.0]];
        let w = array![[1.0, -0.5, 0.25], [-0.75, 0.5, 1.0]];
        let qat = QatLinear::new(w.clone(), Bits::Fp4);
        let packed = qat.convert();
        let expected = x.dot(&dequantize_weight(&packed.packed, &packed.scales, w.dim(), Bits::Fp4).t());
        assert_eq!(packed.forward(&x), expected);
    }
}
