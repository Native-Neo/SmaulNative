use ndarray::{Array1, Array2};

pub enum Parameter {
    Vector { value: Array1<f32>, grad: Array1<f32> },
    Matrix { value: Array2<f32>, grad: Array2<f32> },
}

impl Parameter {
    pub fn zero_grad(&mut self) {
        match self {
            Self::Vector { grad, .. } => grad.fill(0.0),
            Self::Matrix { grad, .. } => grad.fill(0.0),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Vector { value, .. } => value.len(),
            Self::Matrix { value, .. } => value.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn parameters_store_values_and_gradients() {
        let mut parameter = Parameter::Vector {
            value: array![1.0, 2.0],
            grad: array![3.0, 4.0],
        };
        assert_eq!(parameter.len(), 2);
        parameter.zero_grad();
        assert!(matches!(parameter, Parameter::Vector { ref grad, .. } if grad.iter().all(|x| *x == 0.0)));

        let matrix = Parameter::Matrix {
            value: Array2::zeros((2, 3)),
            grad: Array2::ones((2, 3)),
        };
        assert_eq!(matrix.len(), 6);
        assert!(!matrix.is_empty());
    }
}
