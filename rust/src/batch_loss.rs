use ndarray::{Array1, Array2};

use crate::loss::cross_entropy;

pub fn loss(logits: &Array2<f32>, targets: &Array1<usize>) -> f32 {
    cross_entropy(logits, targets).0
}
