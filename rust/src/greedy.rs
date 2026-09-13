use ndarray::Array1;
pub fn select(logits:&Array1<f32>)->usize{assert!(!logits.is_empty());let mut best=0;for i in 1..logits.len(){if logits[i]>logits[best]{best=i;}}best}
#[cfg(test)]
mod tests{use super::*;use ndarray::array;#[test]fn selects(){assert_eq!(select(&array![1.0,3.0,2.0]),1);}}
