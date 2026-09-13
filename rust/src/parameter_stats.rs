use ndarray::Array2;

#[derive(Clone, Copy, Debug, Default)]
pub struct ParameterStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
}

pub fn stats(value: &Array2<f32>) -> ParameterStats {
    if value.is_empty() { return ParameterStats::default(); }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for &x in value.iter() { min = min.min(x); max = max.max(x); sum += x as f64; }
    ParameterStats { min, max, mean: (sum / value.len() as f64) as f32 }
}
