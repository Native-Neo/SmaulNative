use ndarray::Array2;

#[derive(Clone, Debug)]
pub struct ParameterSnapshot {
    pub value: Array2<f32>,
}

impl ParameterSnapshot {
    pub fn capture(value: &Array2<f32>) -> Self { Self { value: value.clone() } }
    pub fn restore(&self, target: &mut Array2<f32>) { *target = self.value.clone(); }
}
