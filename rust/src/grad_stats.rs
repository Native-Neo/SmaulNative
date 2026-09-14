use ndarray::Array2;

pub fn l2_norm(gradient: &Array2<f32>) -> f32 {
    gradient.iter().map(|value| value * value).sum::<f32>().sqrt()
}

pub fn mean_square(gradient: &Array2<f32>) -> f32 {
    assert!(!gradient.is_empty());
    gradient.iter().map(|value| value * value).sum::<f32>() / gradient.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn gradient_statistics_are_correct() {
        let gradient = array![[3.0, 4.0]];
        assert!((l2_norm(&gradient) - 5.0).abs() < 1e-6);
        assert!((mean_square(&gradient) - 12.5).abs() < 1e-6);
    }
}
