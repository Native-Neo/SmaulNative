use ndarray::Array2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bits { Fp2, Fp4 }

impl Bits {
    pub fn from_bits(bits: u8) -> Result<Self, String> {
        match bits { 2 => Ok(Self::Fp2), 4 => Ok(Self::Fp4), _ => Err("bits must be 2 or 4".into()) }
    }

    fn levels(self) -> &'static [f32] {
        match self { Self::Fp2 => &[-1.0, 0.0, 1.0], Self::Fp4 => &[-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0] }
    }
}

pub fn quantize_codes(x: &Array2<f32>, bits: Bits) -> (Array2<u8>, Vec<f32>) {
    let levels = bits.levels();
    let mut codes = Array2::zeros(x.raw_dim());
    let mut scales = vec![0.0; x.ncols()];
    for col in 0..x.ncols() {
        let mut scale = x.column(col).iter().map(|v| v.abs()).fold(0.0, f32::max);
        if scale < f32::EPSILON { scale = 1.0; }
        scales[col] = scale;
        for row in 0..x.nrows() {
            let value = x[[row, col]] / scale;
            let mut best = 0;
            let mut distance = f32::INFINITY;
            for (i, level) in levels.iter().enumerate() {
                let d = (value - level).abs();
                if d < distance { distance = d; best = i; }
            }
            codes[[row, col]] = best as u8;
        }
    }
    (codes, scales)
}

pub fn dequantize(codes: &Array2<u8>, scales: &[f32], bits: Bits) -> Array2<f32> {
    assert_eq!(codes.ncols(), scales.len());
    let levels = bits.levels();
    Array2::from_shape_fn(codes.raw_dim(), |(row, col)| levels[codes[[row, col]] as usize] * scales[col])
}

pub fn fake_quantize(x: &Array2<f32>, bits: Bits) -> Array2<f32> {
    let (codes, scales) = quantize_codes(x, bits);
    dequantize(&codes, &scales, bits)
}

pub fn quantized_linear(input: &Array2<f32>, weight: &Array2<f32>, bits: Bits) -> Array2<f32> {
    assert_eq!(input.ncols(), weight.ncols());
    input.dot(&fake_quantize(weight, bits).t())
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
}
