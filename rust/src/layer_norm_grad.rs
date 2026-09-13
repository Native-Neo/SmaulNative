use ndarray::{Array1, Array2};

pub fn backward(input: &Array2<f32>, grad_output: &Array2<f32>, weight: &Array1<f32>, eps: f32) -> (Array2<f32>, Array1<f32>, Array1<f32>) {
    assert_eq!(input.dim(), grad_output.dim());
    assert_eq!(input.ncols(), weight.len());
    let n = input.ncols() as f32;
    let mut dx = Array2::zeros(input.raw_dim());
    let mut dw = Array1::zeros(weight.len());
    let mut db = Array1::zeros(weight.len());
    for r in 0..input.nrows() {
        let mean = input.row(r).sum() / n;
        let var = input.row(r).iter().map(|v| (*v - mean).powi(2)).sum::<f32>() / n;
        let inv = (var + eps).sqrt().recip();
        let mut xhat = vec![0.0; input.ncols()];
        for c in 0..input.ncols() { xhat[c] = (input[[r,c]] - mean) * inv; dw[c] += grad_output[[r,c]] * xhat[c]; db[c] += grad_output[[r,c]]; }
        let mut a = 0.0; let mut b = 0.0;
        for c in 0..input.ncols() { let dy=grad_output[[r,c]]*weight[c]; a+=dy; b+=dy*xhat[c]; }
        for c in 0..input.ncols() { let dy=grad_output[[r,c]]*weight[c]; dx[[r,c]]=inv*(dy-a/n-xhat[c]*b/n); }
    }
    (dx,dw,db)
}

#[cfg(test)]
mod tests { use super::*; use ndarray::array; #[test] fn gradients_are_finite(){let x=array![[1.,2.,3.,4.]];let g=Array2::ones((1,4));let w=Array1::ones(4);let (dx,dw,db)=backward(&x,&g,&w,1e-5);assert!(dx.iter().all(|v|v.is_finite()));assert!(dw.iter().all(|v|v.is_finite()));assert!(db.iter().all(|v|v.is_finite()));} }
