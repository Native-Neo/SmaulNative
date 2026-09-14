use ndarray::Array2;

pub const QAT_3BIT_MIN: i8 = -4;
pub const QAT_3BIT_MAX: i8 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bits {
    Fp2,
    Fp4,
}

impl Bits {
    pub fn from_bits(bits: u8) -> Result<Self, String> {
        match bits {
            2 => Ok(Self::Fp2),
            4 => Ok(Self::Fp4),
            _ => Err("bits must be 2 or 4".into()),
        }
    }

    fn levels(self) -> &'static [f32] {
        match self {
            Self::Fp2 => &[-1.0, 0.0, 1.0],
            Self::Fp4 => &[-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0],
        }
    }
}

pub fn quantize_codes(x: &Array2<f32>, bits: Bits) -> (Array2<u8>, Vec<f32>) {
    let levels = bits.levels();
    let mut codes = Array2::zeros(x.raw_dim());
    let mut scales = vec![0.0; x.ncols()];

    for col in 0..x.ncols() {
        let mut scale = x.column(col).iter().map(|v| v.abs()).fold(0.0, f32::max);
        if scale < f32::EPSILON {
            scale = 1.0;
        }
        scales[col] = scale;
        for row in 0..x.nrows() {
            let value = x[[row, col]] / scale;
            let mut best = 0;
            let mut distance = f32::INFINITY;
            for (index, level) in levels.iter().enumerate() {
                let current = (value - level).abs();
                if current < distance {
                    distance = current;
                    best = index;
                }
            }
            codes[[row, col]] = best as u8;
        }
    }
    (codes, scales)
}

pub fn dequantize(codes: &Array2<u8>, scales: &[f32], bits: Bits) -> Array2<f32> {
    assert_eq!(codes.ncols(), scales.len());
    let levels = bits.levels();
    Array2::from_shape_fn(codes.raw_dim(), |(row, col)| {
        levels[codes[[row, col]] as usize] * scales[col]
    })
}

pub fn fake_quantize(x: &Array2<f32>, bits: Bits) -> Array2<f32> {
    let (codes, scales) = quantize_codes(x, bits);
    dequantize(&codes, &scales, bits)
}

pub fn quantized_linear(input: &Array2<f32>, weight: &Array2<f32>, bits: Bits) -> Array2<f32> {
    assert_eq!(input.ncols(), weight.ncols());
    input.dot(&fake_quantize(weight, bits).t())
}

pub fn pack_3bit(codes: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity((codes.len() * 3 + 7) / 8);
    let mut accumulator = 0u8;
    let mut bit_count = 0u8;
    for &code in codes {
        let code = code & 7;
        for shift in [2u8, 1, 0] {
            accumulator |= ((code >> shift) & 1) << (7 - bit_count);
            bit_count += 1;
            if bit_count == 8 {
                output.push(accumulator);
                accumulator = 0;
                bit_count = 0;
            }
        }
    }
    if bit_count != 0 {
        output.push(accumulator);
    }
    output
}

pub fn unpack_3bit(packed: &[u8], numel: usize) -> Vec<u8> {
    assert!(packed.len() * 8 >= numel * 3);
    let mut output = Vec::with_capacity(numel);
    for i in 0..numel * 3 {
        let byte = packed[i / 8];
        let bit = (byte >> (7 - (i % 8))) & 1;
        match i % 3 {
            0 => output.push(bit << 2),
            1 => *output.last_mut().unwrap() |= bit << 1,
            _ => *output.last_mut().unwrap() |= bit,
        }
    }
    output
}

pub fn quantize_weight_3bit(weight: &Array2<f32>) -> (Vec<u8>, Vec<f32>) {
    let mut codes = Vec::with_capacity(weight.len());
    let mut scales = Vec::with_capacity(weight.nrows());

    for row in weight.rows() {
        let max_abs = row.iter().map(|v| v.abs()).fold(0.0, f32::max);
        let scale = (max_abs / QAT_3BIT_MAX as f32).max(f32::EPSILON);
        scales.push(scale);
        for &value in row {
            let quantized = (value / scale)
                .round()
                .clamp(QAT_3BIT_MIN as f32, QAT_3BIT_MAX as f32) as i8;
            codes.push((quantized - QAT_3BIT_MIN) as u8);
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
    Array2::from_shape_fn(shape, |(row, col)| {
        ((codes[row * cols + col] as i8 + QAT_3BIT_MIN) as f32) * scales[row]
    })
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
        assert!(weight.nrows() > 0 && weight.ncols() > 0);
        Self {
            weight,
            signed_activation,
            activation_min: 0.0,
            activation_max: 0.0,
            momentum: 0.1,
        }
    }

    pub fn observe(&mut self, input: &Array2<f32>) {
        if input.is_empty() {
            return;
        }
        let min = input.iter().copied().fold(f32::INFINITY, f32::min);
        let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if self.activation_min == 0.0 && self.activation_max == 0.0 {
            self.activation_min = min;
            self.activation_max = max;
        } else {
            self.activation_min = (1.0 - self.momentum) * self.activation_min + self.momentum * min;
            self.activation_max = (1.0 - self.momentum) * self.activation_max + self.momentum * max;
        }
    }

    pub fn fake_quantize_activation(&self, input: &Array2<f32>) -> Array2<f32> {
        if self.signed_activation {
            let max_abs = self.activation_min.abs().max(self.activation_max.abs());
            let scale = (max_abs / QAT_3BIT_MAX as f32).max(f32::EPSILON);
            return Array2::from_shape_fn(input.raw_dim(), |index| {
                (input[index] / scale)
                    .round()
                    .clamp(QAT_3BIT_MIN as f32, QAT_3BIT_MAX as f32)
                    * scale
            });
        }

        let range = (self.activation_max - self.activation_min).max(f32::EPSILON);
        Array2::from_shape_fn(input.raw_dim(), |index| {
            let normalized = ((input[index] - self.activation_min) / range).clamp(0.0, 1.0);
            let q = (normalized * 7.0).round();
            self.activation_min + q * range / 7.0
        })
    }

    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.observe(input);
        let activation = self.fake_quantize_activation(input);
        let (packed, scales) = quantize_weight_3bit(&self.weight);
        let weight = dequantize_weight_3bit(&packed, &scales, self.weight.dim());
        activation.dot(&weight.t())
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
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let weight = dequantize_weight_3bit(&self.packed, &self.scales, self.shape);
        input.dot(&weight.t())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn fp2_round_trip_uses_three_levels() {
        let x = array![[-2.0, 0.0], [0.5, 2.0]];
        let q = fake_quantize(&x, Bits::Fp2);
        assert_eq!(q[[0, 0]], -2.0);
        assert_eq!(q[[0, 1]], 0.0);
        assert_eq!(q[[1, 0]], 1.0);
        assert_eq!(q[[1, 1]], 2.0);
    }

    #[test]
    fn quantized_linear_has_expected_shape() {
        let x = array![[1.0, 2.0, 3.0]];
        let w = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert_eq!(quantized_linear(&x, &w, Bits::Fp4).shape(), &[1, 2]);
    }

    #[test]
    fn pack_3bit_round_trips() {
        let codes = vec![0, 1, 2, 3, 4, 5, 6, 7, 3, 1, 6];
        assert_eq!(unpack_3bit(&pack_3bit(&codes), codes.len()), codes);
    }

    #[test]
    fn qat_3bit_weight_preserves_shape() {
        let weight = array![[0.2, -0.4, 0.7], [1.0, -0.5, 0.25]];
        let (packed, scales) = quantize_weight_3bit(&weight);
        let quantized = dequantize_weight_3bit(&packed, &scales, weight.dim());
        assert_eq!(quantized.dim(), weight.dim());
        assert!(quantized.iter().all(|value| value.is_finite()));
        assert!(quantized[[0, 2]] > 0.6);
    }

    #[test]
    fn qat_linear_observes_and_converts() {
        let weight = array![[1.0, 0.0], [0.0, 1.0]];
        let mut qat = QatLinear3Bit::new(weight, true);
        let input = array![[1.0, -0.5]];
        let output = qat.forward(&input);
        assert_eq!(output.dim(), (1, 2));
        assert!(qat.activation_min <= -0.5);
        assert!(qat.activation_max >= 1.0);
        assert_eq!(qat.convert().shape, (2, 2));
    }
}
