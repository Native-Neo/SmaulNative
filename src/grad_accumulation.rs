use ndarray::Array2;

pub struct GradientAccumulator {
    pub sum: Array2<f32>,
    pub micro_steps: usize,
}

impl GradientAccumulator {
    pub fn new(shape: (usize, usize)) -> Self { Self { sum: Array2::zeros(shape), micro_steps: 0 } }

    pub fn add(&mut self, gradient: &Array2<f32>) {
        assert_eq!(self.sum.dim(), gradient.dim());
        for (a, b) in self.sum.iter_mut().zip(gradient.iter()) { *a += *b; }
        self.micro_steps += 1;
    }

    pub fn mean(&self) -> Array2<f32> {
        if self.micro_steps == 0 { return Array2::zeros(self.sum.dim()); }
        self.sum.mapv(|v| v / self.micro_steps as f32)
    }

    pub fn reset(&mut self) {
        self.sum.fill(0.0);
        self.micro_steps = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn averages_microbatch_gradients() {
        let mut acc = GradientAccumulator::new((1, 2));
        acc.add(&Array2::from_shape_vec((1, 2), vec![1.0, 3.0]).unwrap());
        acc.add(&Array2::from_shape_vec((1, 2), vec![3.0, 5.0]).unwrap());
        assert_eq!(acc.mean(), Array2::from_shape_vec((1, 2), vec![2.0, 4.0]).unwrap());
        acc.reset();
        assert_eq!(acc.micro_steps, 0);
    }
}
