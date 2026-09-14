use ndarray::{Array1, Array2};

pub fn backward(input: &Array2<f32>, grad_output: &Array2<f32>, weight: &Array1<f32>, groups: usize, eps: f32) -> (Array2<f32>, Array1<f32>, Array1<f32>) {
    assert_eq!(input.dim(), grad_output.dim());
    assert_eq!(input.ncols(), weight.len());
    assert!(groups > 0 && input.ncols() % groups == 0);
    let size = input.ncols() / groups;
    let n = size as f32;
    let mut dx=Array2::zeros(input.raw_dim()); let mut dw=Array1::zeros(weight.len()); let mut db=Array1::zeros(weight.len());
    for r in 0..input.nrows() { for group in 0..groups {
        let start=group*size; let end=start+size; let mean=(start..end).map(|c|input[[r,c]]).sum::<f32>()/n;
        let var=(start..end).map(|c|(input[[r,c]]-mean).powi(2)).sum::<f32>()/n; let inv=(var+eps).sqrt().recip();
        let mut xhat=vec![0.0;size]; for c in start..end {xhat[c-start]=(input[[r,c]]-mean)*inv; dw[c]+=grad_output[[r,c]]*xhat[c-start]; db[c]+=grad_output[[r,c]];}
        let mut a=0.0;let mut b=0.0;for c in start..end{let dy=grad_output[[r,c]]*weight[c];a+=dy;b+=dy*xhat[c-start];}
        for c in start..end{let dy=grad_output[[r,c]]*weight[c];dx[[r,c]]=inv*(dy-a/n-xhat[c-start]*b/n);}
    }}
    (dx,dw,db)
}

#[cfg(test)]
mod tests {use super::*;use ndarray::array;#[test]fn finite(){let x=array![[1.,3.,10.,14.]];let g=Array2::ones((1,4));let w=Array1::ones(4);let (dx,_,_)=backward(&x,&g,&w,2,1e-5);assert!(dx.iter().all(|v|v.is_finite()));}}
