use ndarray::{Array2, Array4};

pub struct Gradients { pub state: Array4<f32>, pub w: Array2<f32>, pub k: Array2<f32>, pub v: Array2<f32>, pub kk: Array2<f32>, pub a: Array2<f32>, pub r: Array2<f32> }

pub fn backward(initial: &Array4<f32>, w:&Array2<f32>, k:&Array2<f32>, v:&Array2<f32>, kk:&Array2<f32>, a:&Array2<f32>, r:&Array2<f32>, grad_output:&Array2<f32>, heads:usize, head_size:usize) -> Gradients {
    let steps=w.nrows(); let channels=heads*head_size; assert_eq!(w.dim(),(steps,channels)); assert_eq!(grad_output.dim(),(steps,channels));
    let mut states=Vec::with_capacity(steps+1); states.push(initial.clone()); let mut cur=initial.clone();
    for t in 0..steps { for b in 0..cur.shape()[0] { for h in 0..heads { let base=h*head_size; for i in 0..head_size { let ci=base+i; let p=(0..head_size).map(|j|cur[[b,h,i,j]]*kk[[t,base+j]]).sum::<f32>(); for j in 0..head_size {cur[[b,h,i,j]]=cur[[b,h,i,j]]*w[[t,ci]-0.0]-p*kk[[t,ci]]*a[[t,base+j]]+v[[t,ci]]*k[[t,base+j]];}}}} states.push(cur.clone()); }
    let mut gw=Array2::zeros(w.raw_dim());let mut gk=Array2::zeros(k.raw_dim());let mut gv=Array2::zeros(v.raw_dim());let mut gkk=Array2::zeros(kk.raw_dim());let mut ga=Array2::zeros(a.raw_dim());let mut gr=Array2::zeros(r.raw_dim());let mut gs=Array4::zeros(initial.raw_dim());
    for t in (0..steps).rev() { let prev=&states[t]; let next=&states[t+1]; for b in 0..prev.shape()[0] { for h in 0..heads { let base=h*head_size; for i in 0..head_size { let ci=base+i; let p=(0..head_size).map(|j|prev[[b,h,i,j]]*kk[[t,base+j]]).sum::<f32>(); let mut gnext=vec![0.0;head_size]; for j in 0..head_size {let gj=grad_output[[t,base+j]]*r[[t,base+j]]+gs[[b,h,i,j]];gnext[j]=gj;gr[[t,base+j]]+=grad_output[[t,base+j]]*next[[b,h,i,j]];}
        let dp=-kk[[t,ci]]*(0..head_size).map(|j|gnext[j]*a[[t,base+j]]).sum::<f32>();
        for j in 0..head_size {let sj=prev[[b,h,i,j]];gkk[[t,base+j]]+=dp*sj;gs[[b,h,i,j]]+=dp*kk[[t,base+j]];gkk[[t,ci]]+=-gnext[j]*p*a[[t,base+j]];ga[[t,base+j]]+=-gnext[j]*p*kk[[t,ci]];gv[[t,ci]]+=gnext[j]*k[[t,base+j]];gk[[t,base+j]]+=gnext[j]*v[[t,ci]];gw[[t,ci]]+=gnext[j]*sj;gs[[b,h,i,j]]+=gnext[j]*w[[t,ci]];}
    }}} }
    Gradients{state:gs,w:gw,k:gk,v:gv,kk:gkk,a:ga,r:gr}
}

#[cfg(test)]
mod tests {use super::*;use ndarray::{Array2,Array4};#[test]fn shapes(){let s=Array4::zeros((1,1,2,2));let z=Array2::ones((2,2));let g=backward(&s,&z,&z,&z,&z,&z,&z,&z,1,2);assert_eq!(g.w.dim(),(2,2));assert!(g.w.iter().all(|v|v.is_finite()));}}
