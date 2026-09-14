use ndarray::Array2;

pub fn uniform(rows: usize, cols: usize, low: f32, high: f32, seed: u64) -> Array2<f32> {
    assert!(rows > 0 && cols > 0);
    let mut out = Array2::<f32>::zeros((rows, cols));
    let span = high - low;
    for r in 0..rows {
        for c in 0..cols {
            let mut x = seed.wrapping_add((r * cols + c) as u64 + 1);
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            let unit = (x.wrapping_mul(2685821657736338717) >> 11) as f32 / ((1u64 << 53) as f32);
            out[[r, c]] = low + span * unit;
        }
    }
    out
}

pub fn orthogonal(rows: usize, cols: usize, gain: f32, seed: u64) -> Array2<f32> {
    assert!(rows > 0 && cols > 0);
    let mut out = uniform(rows, cols, -1.0, 1.0, seed);
    let vectors = if rows >= cols { cols } else { rows };
    let mut basis = Vec::with_capacity(vectors);

    if rows >= cols {
        for c in 0..cols {
            let mut v = vec![0.0f32; rows];
            for r in 0..rows { v[r] = out[[r, c]]; }
            orthogonalize(&mut v, &basis);
            normalize(&mut v);
            for r in 0..rows { out[[r, c]] = v[r]; }
            basis.push(v);
        }
    } else {
        for r in 0..rows {
            let mut v = vec![0.0f32; cols];
            for c in 0..cols { v[c] = out[[r, c]]; }
            orthogonalize(&mut v, &basis);
            normalize(&mut v);
            for c in 0..cols { out[[r, c]] = v[c]; }
            basis.push(v);
        }
    }

    out.mapv(|x| x * gain)
}

fn orthogonalize(v: &mut [f32], basis: &[Vec<f32>]) {
    for b in basis {
        let projection: f32 = v.iter().zip(b).map(|(x, y)| x * y).sum();
        for (x, y) in v.iter_mut().zip(b) { *x -= projection * *y; }
    }
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() { *x /= norm; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthogonal_columns_are_unit_length() {
        let x = orthogonal(8, 3, 0.1, 7);
        for c in 0..3 {
            let norm = (0..8).map(|r| x[[r, c]] * x[[r, c]]).sum::<f32>().sqrt();
            assert!((norm - 0.1).abs() < 1e-5);
        }
    }

    #[test]
    fn uniform_is_deterministic() {
        assert_eq!(uniform(2, 3, -1.0, 1.0, 9), uniform(2, 3, -1.0, 1.0, 9));
    }
}
