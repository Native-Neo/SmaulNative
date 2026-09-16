use ndarray::Array2;
use crate::qat::{Bits,QatLinear,PackedLinear};

#[derive(Clone,Debug)]
pub struct RealQuantLinear { pub weight:Array2<f32>, pub bias:Option<Vec<f32>>, pub bits:Bits, pub quant:PackedLinear }
impl RealQuantLinear {
 pub fn new(weight:Array2<f32>,bias:Option<Vec<f32>>,bits:u8)->Result<Self,String>{let bits=Bits::from_bits(bits)?;let quant=QatLinear::new(weight.clone(),bits).convert();Ok(Self{weight,bias,bits,quant})}
 pub fn refresh(&mut self){self.quant=QatLinear::new(self.weight.clone(),self.bits).convert();}
 pub fn dequantized(&self)->Array2<f32>{crate::qat::dequantize_weight(&self.quant.packed,&self.quant.scales,self.quant.shape,self.bits)}
 pub fn forward(&self,input:&Array2<f32>)->Array2<f32>{let input=input.as_standard_layout();let values=crate::gpu::packed_linear(input.as_slice().expect("standard layout input"),&self.quant.packed,&self.quant.scales,input.nrows(),self.quant.shape.0,self.quant.shape.1,self.bits.bits()).unwrap();let mut out=Array2::from_shape_vec((input.nrows(),self.quant.shape.0),values).unwrap();if let Some(b)=&self.bias{assert_eq!(b.len(),out.ncols());for r in 0..out.nrows(){for c in 0..out.ncols(){out[[r,c]]+=b[c];}}}out}
 pub fn backward(&self,input:&Array2<f32>,grad:&Array2<f32>)->(Array2<f32>,Array2<f32>){(grad.dot(&self.dequantized()),grad.t().dot(input))}
}

#[cfg(test)]
mod tests{use super::*;use ndarray::array;#[test]fn packed_training_roundtrip(){let l=RealQuantLinear::new(array![[1.0,0.0],[0.0,1.0]],None,4).unwrap();let y=l.forward(&array![[2.0,3.0]]);assert_eq!(y.shape(),&[1,2]);assert_eq!(l.dequantized().shape(),&[2,2]);}#[test]fn supported_bits(){for b in [2,3,4,8]{assert!(RealQuantLinear::new(array![[1.0,0.0],[0.0,1.0]],None,b).is_ok())}assert!(RealQuantLinear::new(array![[1.0]],None,5).is_err());}
#[test]fn forward_accepts_non_contiguous_input(){let l=RealQuantLinear::new(array![[1.0,0.0],[0.0,1.0]],None,4).unwrap();let x=array![[2.0,3.0],[4.0,5.0]];let y=l.forward(&x.t().to_owned());assert_eq!(y.shape(),&[2,2]);let view=x.slice(ndarray::s![..;1,..]).to_owned();assert_eq!(l.forward(&view).shape(),&[2,2]);}#[test]fn direct_fp8_forward(){let l=RealQuantLinear::new(array![[1.0,0.0]],None,8).unwrap();let y=l.forward(&array![[2.0,3.0]]);assert_eq!(y.shape(),&[1,1]);assert!(y[[0,0]].is_finite());}}
