use ndarray::{Array2, Axis};

pub fn matmul(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    assert_eq!(a.ncols(), b.nrows());
    a.dot(b)
}

pub fn matmul_backward(a: &Array2<f32>, b: &Array2<f32>, grad: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    assert_eq!(grad.nrows(), a.nrows());
    assert_eq!(grad.ncols(), b.ncols());
    let da = grad.dot(&b.t());
    let db = a.t().dot(grad);
    (da, db)
}

pub fn row_sum(x: &Array2<f32>) -> Array2<f32> {
    x.sum_axis(Axis(1)).insert_axis(Axis(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_matches_expected() {
        let a = Array2::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let b = Array2::from_shape_vec((3, 2), vec![1., 2., 3., 4., 5., 6.]).unwrap();
        assert_eq!(matmul(&a, &b), Array2::from_shape_vec((2, 2), vec![22., 28., 49., 64.]).unwrap());
    }

    #[test]
    fn matmul_backward_has_correct_shapes() {
        let a = Array2::ones((2, 3));
        let b = Array2::ones((3, 4));
        let g = Array2::ones((2, 4));
        let (da, db) = matmul_backward(&a, &b, &g);
        assert_eq!(da.dim(), a.dim());
        assert_eq!(db.dim(), b.dim());
    }
}
