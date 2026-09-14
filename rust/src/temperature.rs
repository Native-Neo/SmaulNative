pub fn scale(logit:f32,temperature:f32)->f32{assert!(temperature.is_finite()&&temperature>0.0);logit/temperature}
pub fn scale_slice(logits:&mut [f32],temperature:f32){for v in logits.iter_mut(){*v=scale(*v,temperature);}}
#[cfg(test)]
mod tests{use super::*;#[test]fn scales(){let mut x=[2.0,4.0];scale_slice(&mut x,2.0);assert_eq!(x,[1.0,2.0]);}}
