use ndarray::{Array1, Array2};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutedChunks {
    pub selected: Vec<usize>,
    pub current: usize,
}

pub fn select_topk(scores: &Array1<f32>, top_k: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..scores.len()).collect();
    ids.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.cmp(&b)));
    ids.truncate(top_k.min(ids.len()));
    ids.sort_unstable();
    ids
}

pub fn route_query(query: &Array1<f32>, chunk_means: &Array2<f32>, current_chunk: usize, top_k: usize) -> RoutedChunks {
    assert_eq!(query.len(), chunk_means.ncols());
    let mut scores = Array1::zeros(chunk_means.nrows());
    for c in 0..chunk_means.nrows() {
        scores[c] = query.iter().zip(chunk_means.row(c).iter()).map(|(a,b)| a*b).sum();
    }
    let mut selected = select_topk(&scores, top_k);
    selected.retain(|&c| c != current_chunk);
    RoutedChunks { selected, current: current_chunk }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_excludes_current_chunk() {
        let q = Array1::from_vec(vec![1.0, 0.0]);
        let means = Array2::from_shape_vec((3, 2), vec![0.1,0.0, 0.9,0.0, 2.0,0.0]).unwrap();
        let route = route_query(&q, &means, 2, 2);
        assert_eq!(route.current, 2);
        assert!(!route.selected.contains(&2));
    }
}
