use ndarray::{Array1, Array2};

pub fn add_in_place(dst: &mut Array2<f32>, src: &Array2<f32>) {
    assert_eq!(dst.dim(), src.dim());
    for (a, b) in dst.iter_mut().zip(src.iter()) { *a += *b; }
}

pub fn add_vector_in_place(dst: &mut Array1<f32>, src: &Array1<f32>) {
    assert_eq!(dst.len(), src.len());
    for (a, b) in dst.iter_mut().zip(src.iter()) { *a += *b; }
}

pub fn zeros_like(value: &Array2<f32>) -> Array2<f32> { Array2::zeros(value.dim()) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_matrix_gradients() {
        let mut a = Array2::from_elem((2, 2), 1.0);
        let b = Array2::from_elem((2, 2), 2.0);
        add_in_place(&mut a, &b);
        assert_eq!(a, Array2::from_elem((2, 2), 3.0));
    }

    #[test]
    fn accumulates_vector_gradients() {
        let mut a = Array1::from_elem(3, 1.0);
        let b = Array1::from_elem(3, 2.0);
        add_vector_in_place(&mut a, &b);
        assert_eq!(a, Array1::from_elem(3, 3.0));
    }
}
