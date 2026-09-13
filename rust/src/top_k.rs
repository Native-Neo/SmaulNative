use ndarray::Array1;

pub fn top_k_indices(values: &Array1<f32>, k: usize) -> Vec<usize> {
    assert!(k > 0);
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_unstable_by(|&a, &b| values[b].total_cmp(&values[a]).then_with(|| a.cmp(&b)));
    indices.truncate(k.min(indices.len()));
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_descending_top_k() {
        let values = Array1::from_vec(vec![1.0, 4.0, 3.0, 2.0]);
        assert_eq!(top_k_indices(&values, 2), vec![1, 2]);
    }
}
