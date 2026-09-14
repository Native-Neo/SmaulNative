pub fn apply(logits:&mut [f32],seen:&[usize],penalty:f32){assert!(penalty.is_finite()&&penalty>0.0);for &id in seen{if id<logits.len(){let v=logits[id];logits[id]=if v>=0.0{v/penalty}else{v*penalty};}}}
#[cfg(test)]
mod tests{use super::*;#[test]fn penalizes_seen_tokens(){let mut x=[2.0,-2.0,1.0];apply(&mut x,&[0,1],2.0);assert_eq!(x,[1.0,-4.0,1.0]);}}
