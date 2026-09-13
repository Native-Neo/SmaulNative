use ndarray::{Array1, Array2};

pub fn accuracy(logits: &Array2<f32>, targets: &Array1<usize>) -> f32 {
    assert_eq!(logits.nrows(), targets.len());
    if targets.is_empty() { return 0.0; }
    let correct = logits.outer_iter().zip(targets.iter()).filter(|(row, target)| {
        let mut best = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, &value) in row.iter().enumerate() {
            if value > best_value { best_value = value; best = index; }
        }
        best == **target
    }).count();
    correct as f32 / targets.len() as f32
}
