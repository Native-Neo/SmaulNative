use ndarray::{Array2, Array3, Array4, Axis};

pub fn run(state:Array4<f32>,w:&Array2<f32>,k:&Array2<f32>,v:&Array2<f32>,kk:&Array2<f32>,a:&Array2<f32>,r:&Array2<f32>,heads:usize,head_size:usize)->(Array4<f32>,Array2<f32>){assert_eq!(state.shape()[0],1,"run is the single-batch compatibility API; use run_batched for batch > 1");let(state,output)=run_batched(state,w,k,v,kk,a,r,heads,head_size);(state,output.index_axis(Axis(0),0).to_owned())}

pub fn run_batched(mut state:Array4<f32>,w:&Array2<f32>,k:&Array2<f32>,v:&Array2<f32>,kk:&Array2<f32>,a:&Array2<f32>,r:&Array2<f32>,heads:usize,head_size:usize)->(Array4<f32>,Array3<f32>){
    let steps=w.nrows();let channels=heads*head_size;assert_eq!(state.shape()[1],heads);assert_eq!(state.shape()[2],head_size);assert_eq!(state.shape()[3],head_size);assert_eq!(w.dim(),(steps,channels));for input in [k,v,kk,a,r]{assert_eq!(input.dim(),(steps,channels));}
    let batch=state.shape()[0];let mut output=Array3::<f32>::zeros((batch,steps,channels));
    for t in 0..steps{for b in 0..batch{for h in 0..heads{let base=h*head_size;for i in 0..head_size{let ci=base+i;let wt=w[[t,ci]];let kkt=kk[[t,ci]];let mut projected=0.0f32;for j in 0..head_size{projected+=state[[b,h,i,j]]*kk[[t,base+j]];}for j in 0..head_size{let old=state[[b,h,i,j]];let decay=old*wt;let correction=-projected*kkt*a[[t,base+j]];let value_update=v[[t,ci]]*k[[t,base+j]];state[[b,h,i,j]]=decay+correction+value_update;}}for i in 0..head_size{let ci=base+i;let mut value=0.0f32;for j in 0..head_size{value+=state[[b,h,i,j]]*r[[t,base+j]];}output[[b,t,ci]]=value;}}}}
    (state,output)
}

#[cfg(test)]mod tests_wkv{use super::*;use ndarray::{array,Array4};
#[test]fn single_step_matches_recurrence(){let state=Array4::zeros((1,1,2,2));let w=array![[0.5,0.25]];let k=array![[2.0,3.0]];let v=array![[4.0,5.0]];let kk=array![[0.5,0.25]];let a=array![[0.2,0.4]];let r=array![[1.0,2.0]];let(next,output)=run(state,&w,&k,&v,&kk,&a,&r,1,2);assert_eq!(next[[0,0,0,0]],8.0);assert_eq!(next[[0,0,0,1]],12.0);assert_eq!(next[[0,0,1,0]],10.0);assert_eq!(next[[0,0,1,1]],15.0);assert_eq!(output,array![[32.0,40.0]]);}
#[test]fn batched_outputs_are_independent(){let mut state=Array4::zeros((2,1,1,1));state[[1,0,0,0]]=10.0;let w=array![[1.0]];let k=array![[1.0]];let v=array![[0.0]];let kk=array![[0.0]];let a=array![[0.0]];let r=array![[1.0]];let(_,output)=run_batched(state,&w,&k,&v,&kk,&a,&r,1,1);assert_eq!(output[[0,0,0]],0.0);assert_eq!(output[[1,0,0]],10.0);}
#[test]fn single_batch_compatibility_api_rejects_multi_batch(){let state=Array4::zeros((2,1,1,1));let result=std::panic::catch_unwind(||run(state,&array![[1.0]],&array![[1.0]],&array![[0.0]],&array![[0.0]],&array![[0.0]],&array![[1.0]],1,1));assert!(result.is_err());}
#[test]fn zero_state_stays_zero_for_zero_inputs(){let state=Array4::zeros((2,2,4,4));let w=Array2::zeros((3,8));let k=Array2::zeros((3,8));let v=Array2::zeros((3,8));let kk=Array2::zeros((3,8));let a=Array2::zeros((3,8));let r=Array2::zeros((3,8));let(next,output)=run_batched(state,&w,&k,&v,&kk,&a,&r,2,4);assert!(next.iter().all(|x|*x==0.0));assert!(output.iter().all(|x|*x==0.0));}}


// ===== wkv_backward =====

pub struct Gradients { pub state: Array4<f32>, pub w: Array2<f32>, pub k: Array2<f32>, pub v: Array2<f32>, pub kk: Array2<f32>, pub a: Array2<f32>, pub r: Array2<f32> }
pub fn backward(initial:&Array4<f32>,w:&Array2<f32>,k:&Array2<f32>,v:&Array2<f32>,kk:&Array2<f32>,a:&Array2<f32>,r:&Array2<f32>,grad_output:&Array2<f32>,heads:usize,head_size:usize)->Gradients{let steps=w.nrows();let channels=heads*head_size;assert_eq!(initial.shape(),&[1,heads,head_size,head_size]);for x in[k,v,kk,a,r,grad_output]{assert_eq!(x.dim(),(steps,channels));}assert_eq!(w.dim(),(steps,channels));let mut states=Vec::with_capacity(steps+1);states.push(initial.clone());let mut state=initial.clone();for t in 0..steps{let mut next=state.clone();for h in 0..heads{let base=h*head_size;for i in 0..head_size{let ci=base+i;let mut p=0.0f32;for j in 0..head_size{p+=state[[0,h,i,j]]*kk[[t,base+j]];}for j in 0..head_size{let cj=base+j;next[[0,h,i,j]]=state[[0,h,i,j]]*w[[t,ci]]-p*kk[[t,ci]]*a[[t,cj]]+v[[t,ci]]*k[[t,cj]];}}}state=next;states.push(state.clone());}let mut gw=Array2::zeros(w.raw_dim());let mut gk=Array2::zeros(k.raw_dim());let mut gv=Array2::zeros(v.raw_dim());let mut gkk=Array2::zeros(kk.raw_dim());let mut ga=Array2::zeros(a.raw_dim());let mut gr=Array2::zeros(r.raw_dim());let mut gstate=Array4::zeros(initial.raw_dim());for t in(0..steps).rev(){let prev=&states[t];let next=&states[t+1];let mut gprev=Array4::zeros(initial.raw_dim());for h in 0..heads{let base=h*head_size;for i in 0..head_size{let ci=base+i;let mut p=0.0f32;for j in 0..head_size{p+=prev[[0,h,i,j]]*kk[[t,base+j]];}let mut g_p=0.0f32;for j in 0..head_size{let cj=base+j;let gs:f32=gstate[[0,h,i,j]]+grad_output[[t,ci]]*r[[t,cj]];gr[[t,cj]]+=grad_output[[t,ci]]*next[[0,h,i,j]];gw[[t,ci]]+=gs*prev[[0,h,i,j]];gprev[[0,h,i,j]]+=gs*w[[t,ci]];gv[[t,ci]]+=gs*k[[t,cj]];gk[[t,cj]]+=gs*v[[t,ci]];g_p+=-gs*kk[[t,ci]]*a[[t,cj]];ga[[t,cj]]+=-gs*p*kk[[t,ci]];gkk[[t,ci]]+=-gs*p*a[[t,cj]];}for j in 0..head_size{let cj=base+j;gprev[[0,h,i,j]]+=g_p*kk[[t,cj]];gkk[[t,cj]]+=g_p*prev[[0,h,i,j]];}}}gstate=gprev;}Gradients{state:gstate,w:gw,k:gk,v:gv,kk:gkk,a:ga,r:gr}}
#[cfg(test)]mod tests_wkv_backward{use super::*;use ndarray::{Array2,Array4};#[test]fn backward_shapes(){let s=Array4::zeros((1,1,2,2));let z=Array2::ones((3,2));let g=backward(&s,&z,&z,&z,&z,&z,&z,&z,1,2);assert_eq!(g.w.dim(),(3,2));assert!(g.w.iter().all(|v|v.is_finite()));assert!(g.state.iter().all(|v|v.is_finite()));}

    // Regression test for the head_size>1 index-swap bug: head_size=1 makes the bug a
    // no-op (i and j both 0), so this must use head_size>1 to actually exercise it.
    #[test]
    fn backward_matches_finite_difference_for_head_size_greater_than_one() {
        use {dot_loss, finite_difference};

        let (heads, head_size, steps) = (2, 3, 2);
        let channels = heads * head_size;
        let fill = |seed: u64, scale: f32| {
            let mut s = seed | 1;
            Array2::<f32>::from_shape_fn((steps, channels), |_| {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                ((s as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0) * scale
            })
        };
        let state = Array4::<f32>::from_shape_fn((1, heads, head_size, head_size), |_| 0.05);
        let w = fill(1, 0.1).mapv(|x| 0.6 + x);
        let k = fill(2, 0.3);
        let v = fill(3, 0.3);
        let kk = fill(4, 0.3);
        let a = fill(5, 0.3);
        let r = fill(6, 0.3);
        let grad_output = fill(7, 1.0);

        let analytic = backward(&state, &w, &k, &v, &kk, &a, &r, &grad_output, heads, head_size);

        let loss_with = |w: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>, kk: &Array2<f32>, a: &Array2<f32>, r: &Array2<f32>| {
            let (_, out) = run_batched(state.clone(), w, k, v, kk, a, r, heads, head_size);
            dot_loss(&out.index_axis(ndarray::Axis(0), 0).to_owned(), &grad_output)
        };

        let eps = 1e-3;
        let tensors: [(&str, &Array2<f32>, &Array2<f32>); 6] = [
            ("w", &w, &analytic.w), ("k", &k, &analytic.k), ("v", &v, &analytic.v),
            ("kk", &kk, &analytic.kk), ("a", &a, &analytic.a), ("r", &r, &analytic.r),
        ];
        for (name, tensor, grad) in tensors {
            for t in 0..steps {
                for c in 0..channels {
                    let mut cell = tensor[[t, c]];
                    let numeric = finite_difference(&mut cell, eps, |x| {
                        let mut w2 = w.clone(); let mut k2 = k.clone(); let mut v2 = v.clone();
                        let mut kk2 = kk.clone(); let mut a2 = a.clone(); let mut r2 = r.clone();
                        match name {
                            "w" => w2[[t, c]] = x, "k" => k2[[t, c]] = x, "v" => v2[[t, c]] = x,
                            "kk" => kk2[[t, c]] = x, "a" => a2[[t, c]] = x, _ => r2[[t, c]] = x,
                        }
                        loss_with(&w2, &k2, &v2, &kk2, &a2, &r2)
                    });
                    assert!((numeric - grad[[t, c]]).abs() < 1e-2,
                        "{name}[{t},{c}]: analytic={} numeric={numeric}", grad[[t, c]]);
                }
            }
        }
    }
}

// ===== wkv_backward_check =====
pub fn finite_difference<F>(value:&mut f32,epsilon:f32,f:F)->f32 where F:Fn(f32)->f32{let original=*value;*value=original+epsilon;let plus=f(*value);*value=original-epsilon;let minus=f(*value);*value=original;(plus-minus)/(2.0*epsilon)}
pub fn dot_loss(output:&Array2<f32>,upstream:&Array2<f32>)->f32{assert_eq!(output.dim(),upstream.dim());output.iter().zip(upstream.iter()).map(|(a,b)|a*b).sum()}
pub fn state_shape(state:&Array4<f32>,heads:usize,head_size:usize){assert_eq!(state.ndim(),4);assert_eq!(state.shape()[1],heads);assert_eq!(state.shape()[2],head_size);assert_eq!(state.shape()[3],head_size)}
#[cfg(test)]mod tests_wkv_backward_check{use super::*;#[test]fn central_difference_is_correct_for_square(){let mut x=3.0;let grad=finite_difference(&mut x,1e-2,|v|v*v);assert!((grad-6.0).abs()<1e-3)}#[test]fn validates_wkv_state_shape(){let state=Array4::zeros((1,2,4,4));state_shape(&state,2,4)}}
