use ndarray::Array2;
pub fn check_gradient(gradient:&Array2<f32>)->bool{gradient.iter().all(|v|v.is_finite())}
pub fn sanitize_gradient(gradient:&mut Array2<f32>){for v in gradient.iter_mut(){if !v.is_finite(){*v=0.0;}}}
#[cfg(test)]
mod tests{use super::*;use ndarray::array;#[test]fn detects_and_sanitizes(){let mut g=array![[1.0,f32::NAN]];assert!(!check_gradient(&g));sanitize_gradient(&mut g);assert!(check_gradient(&g));assert_eq!(g[[0,1]],0.0);}}
