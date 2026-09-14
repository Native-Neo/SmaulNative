use ndarray::{Array1, Array2};

#[derive(Clone, Debug)]
pub struct Tensor2 {
    pub data: Array2<f32>,
    pub grad: Option<Array2<f32>>,
}

impl Tensor2 {
    pub fn new(data: Array2<f32>) -> Self { Self { data, grad: None } }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(Array2::zeros((rows, cols)))
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self::new(Array2::ones((rows, cols)))
    }

    pub fn zero_grad(&mut self) { self.grad = None; }

    pub fn add_grad(&mut self, grad: &Array2<f32>) {
        match &mut self.grad {
            Some(existing) => *existing += grad,
            None => self.grad = Some(grad.clone()),
        }
    }

    pub fn sum(&self) -> f32 { self.data.sum() }

    pub fn mean(&self) -> f32 { self.data.mean().unwrap_or(0.0) }

    pub fn flatten(&self) -> Array1<f32> { Array1::from_iter(self.data.iter().copied()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_accumulates_gradients() {
        let mut x = Tensor2::zeros(2, 2);
        let g = Array2::ones((2, 2));
        x.add_grad(&g);
        x.add_grad(&g);
        assert_eq!(x.grad.unwrap(), Array2::from_elem((2, 2), 2.0));
    }

    #[test]
    fn tensor_statistics_work() {
        let x = Tensor2::new(Array2::from_elem((2, 3), 2.0));
        assert_eq!(x.sum(), 12.0);
        assert_eq!(x.mean(), 2.0);
    }
}
