use ndarray::Array2;

pub struct Parameter {
    pub value: Array2<f32>,
    pub grad: Array2<f32>,
}

impl Parameter {
    pub fn new(value: Array2<f32>) -> Self {
        let grad = Array2::zeros(value.dim());
        Self { value, grad }
    }

    pub fn zero_grad(&mut self) { self.grad.fill(0.0); }
    pub fn len(&self) -> usize { self.value.len() }
    pub fn is_empty(&self) -> bool { self.value.is_empty() }

    pub fn value_slice(&self) -> &[f32] { self.value.as_slice().expect("contiguous") }
    pub fn value_slice_mut(&mut self) -> &mut [f32] { self.value.as_slice_mut().expect("contiguous") }
    pub fn grad_slice(&self) -> &[f32] { self.grad.as_slice().expect("contiguous") }
    pub fn grad_slice_mut(&mut self) -> &mut [f32] { self.grad.as_slice_mut().expect("contiguous") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_starts_with_zero_gradient() {
        let p = Parameter::new(Array2::ones((2, 3)));
        assert_eq!(p.len(), 6);
        assert!(p.grad.iter().all(|v| *v == 0.0));
    }
}
