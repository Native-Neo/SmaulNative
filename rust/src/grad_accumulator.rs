use ndarray::{Array1, Array2};

pub fn add_matrix(dst: &mut Array2<f32>, src: &Array2<f32>) {
    assert_eq!(dst.dim(), src.dim());
    *dst += src;
}

pub fn add_vector(dst: &mut Array1<f32>, src: &Array1<f32>) {
    assert_eq!(dst.len(), src.len());
    *dst += src;
}

pub fn scale_matrix(value: &mut Array2<f32>, scale: f32) {
    value.mapv_inplace(|x| x * scale);
}

pub fn scale_vector(value: &mut Array1<f32>, scale: f32) {
    value.mapv_inplace(|x| x * scale);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn matrix_gradients_accumulate_and_scale() {
        let mut a = Array2::ones((2, 2));
        let b = Array2::from_elem((2, 2), 2.0);
        add_matrix(&mut a, &b);
        scale_matrix(&mut a, 0.5);
        assert_eq!(a, Array2::from_elem((2, 2), 1.5));
    }

    #[test]
    fn vector_gradients_accumulate() {
        let mut a = array![1.0, 2.0];
        add_vector(&mut a, &array![3.0, 4.0]);
        assert_eq!(a, array![4.0, 6.0]);
    }
}
