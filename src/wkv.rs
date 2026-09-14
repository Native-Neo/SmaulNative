use ndarray::{Array2,Array3,Array4,Axis};

pub fn run(state:Array4<f32>,w:&Array2<f32>,k:&Array2<f32>,v:&Array2<f32>,kk:&Array2<f32>,a:&Array2<f32>,r:&Array2<f32>,heads:usize,head_size:usize)->(Array4<f32>,Array2<f32>){assert_eq!(state.shape()[0],1,"wkv::run is the single-batch compatibility API; use run_batched for batch > 1");let(state,output)=run_batched(state,w,k,v,kk,a,r,heads,head_size);(state,output.index_axis(Axis(0),0).to_owned())}

pub fn run_batched(mut state:Array4<f32>,w:&Array2<f32>,k:&Array2<f32>,v:&Array2<f32>,kk:&Array2<f32>,a:&Array2<f32>,r:&Array2<f32>,heads:usize,head_size:usize)->(Array4<f32>,Array3<f32>){
    let steps=w.nrows();let channels=heads*head_size;assert_eq!(state.shape()[1],heads);assert_eq!(state.shape()[2],head_size);assert_eq!(state.shape()[3],head_size);assert_eq!(w.dim(),(steps,channels));for input in [k,v,kk,a,r]{assert_eq!(input.dim(),(steps,channels));}
    let batch=state.shape()[0];let mut output=Array3::<f32>::zeros((batch,steps,channels));
    for t in 0..steps{for b in 0..batch{for h in 0..heads{let base=h*head_size;for i in 0..head_size{let ci=base+i;let wt=w[[t,ci]];let kkt=kk[[t,ci]];let mut projected=0.0f32;for j in 0..head_size{projected+=state[[b,h,i,j]]*kk[[t,base+j]];}for j in 0..head_size{let old=state[[b,h,i,j]];let decay=old*wt;let correction=-projected*kkt*a[[t,base+j]];let value_update=v[[t,ci]]*k[[t,base+j]];state[[b,h,i,j]]=decay+correction+value_update;}}for i in 0..head_size{let ci=base+i;let mut value=0.0f32;for j in 0..head_size{value+=state[[b,h,i,j]]*r[[t,base+j]];}output[[b,t,ci]]=value;}}}}
    (state,output)
}

#[cfg(test)]mod tests{use super::*;use ndarray::{array,Array4};
#[test]fn single_step_matches_recurrence(){let state=Array4::zeros((1,1,2,2));let w=array![[0.5,0.25]];let k=array![[2.0,3.0]];let v=array![[4.0,5.0]];let kk=array![[0.5,0.25]];let a=array![[0.2,0.4]];let r=array![[1.0,2.0]];let(next,output)=run(state,&w,&k,&v,&kk,&a,&r,1,2);assert_eq!(next[[0,0,0,0]],8.0);assert_eq!(next[[0,0,0,1]],12.0);assert_eq!(next[[0,0,1,0]],10.0);assert_eq!(next[[0,0,1,1]],15.0);assert_eq!(output,array![[32.0,40.0]]);}
#[test]fn batched_outputs_are_independent(){let mut state=Array4::zeros((2,1,1,1));state[[1,0,0,0]]=10.0;let w=array![[1.0]];let k=array![[1.0]];let v=array![[0.0]];let kk=array![[0.0]];let a=array![[0.0]];let r=array![[1.0]];let(_,output)=run_batched(state,&w,&k,&v,&kk,&a,&r,1,1);assert_eq!(output[[0,0,0]],0.0);assert_eq!(output[[1,0,0]],10.0);}
#[test]fn single_batch_compatibility_api_rejects_multi_batch(){let state=Array4::zeros((2,1,1,1));let result=std::panic::catch_unwind(||run(state,&array![[1.0]],&array![[1.0]],&array![[0.0]],&array![[0.0]],&array![[0.0]],&array![[1.0]],1,1));assert!(result.is_err());}
#[test]fn zero_state_stays_zero_for_zero_inputs(){let state=Array4::zeros((2,2,4,4));let w=Array2::zeros((3,8));let k=Array2::zeros((3,8));let v=Array2::zeros((3,8));let kk=Array2::zeros((3,8));let a=Array2::zeros((3,8));let r=Array2::zeros((3,8));let(next,output)=run_batched(state,&w,&k,&v,&kk,&a,&r,2,4);assert!(next.iter().all(|x|*x==0.0));assert!(output.iter().all(|x|*x==0.0));}}
