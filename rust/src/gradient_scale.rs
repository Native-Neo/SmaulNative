use ndarray::Array2;

pub fn scale(gradient: &mut Array2<f32>, factor: f32) {
    assert!(factor.is_finite());
    gradient.mapv_inplace(|value| value * factor);
}
