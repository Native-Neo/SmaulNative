use ndarray::{Array2,Array4};
pub fn finite_difference<F>(value:&mut f32,epsilon:f32,f:F)->f32 where F:Fn(f32)->f32{let original=*value;*value=original+epsilon;let plus=f(*value);*value=original-epsilon;let minus=f(*value);*value=original;(plus-minus)/(2.0*epsilon)}
pub fn dot_loss(output:&Array2<f32>,upstream:&Array2<f32>)->f32{assert_eq!(output.dim(),upstream.dim());output.iter().zip(upstream.iter()).map(|(a,b)|a*b).sum()}
pub fn state_shape(state:&Array4<f32>,heads:usize,head_size:usize){assert_eq!(state.ndim(),4);assert_eq!(state.shape()[1],heads);assert_eq!(state.shape()[2],head_size);assert_eq!(state.shape()[3],head_size)}
#[cfg(test)]mod tests{use super::*;#[test]fn central_difference_is_correct_for_square(){let mut x=3.0;let grad=finite_difference(&mut x,1e-2,|v|v*v);assert!((grad-6.0).abs()<1e-3)}#[test]fn validates_wkv_state_shape(){let state=Array4::zeros((1,2,4,4));state_shape(&state,2,4)}}
