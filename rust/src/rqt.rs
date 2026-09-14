use ndarray::Array2;
use crate::qat::{Bits,QatLinear,PackedLinear};

pub const REFRESH_ROWS: usize = 64;

#[derive(Clone,Debug)]
pub struct RealQuantLinear { pub weight:Array2<f32>, pub bias:Option<Vec<f32>>, pub bits:Bits, pub quant:PackedLinear }
impl RealQuantLinear {
 pub fn new(weight:Array2<f32>,bias:Option<Vec<f32>>,bits:u8)->Result<Self,String>{let bits=Bits::from_bits(bits)?;let quant=QatLinear::new(weight.clone(),bits).convert();Ok(Self{weight,bias,bits,quant})}
 pub fn refresh(&mut self){self.quant=QatLinear::new(self.weight.clone(),self.bits).convert();}
 pub fn forward(&self,input:&Array2<f32>)->Array2<f32>{let mut out=self.quant.forward(input);if let Some(b)=&self.bias{for r in 0..out.nrows(){for c in 0..out.ncols(){out[[r,c]]+=b[c];}}}out}
 pub fn backward(&self,input:&Array2<f32>,grad:&Array2<f32>)->(Array2<f32>,Array2<f32>){let q=crate::qat::dequantize_weight(&self.quant.packed,&self.quant.scales,self.quant.shape,self.bits);(grad.dot(&q),grad.t().dot(input))}
}

#[derive(Clone,Debug)]
pub struct RqtConfig { pub bits:Bits }
impl RqtConfig { pub fn new(bits:u8)->Result<Self,String>{Ok(Self{bits:Bits::from_bits(bits)?})} }

pub fn prepare_linear(weight:Array2<f32>,bias:Option<Vec<f32>>,bits:u8)->Result<RealQuantLinear,String>{RealQuantLinear::new(weight,bias,bits)}
pub fn refresh(linear:&mut RealQuantLinear){linear.refresh()}

#[cfg(test)]
mod tests{use super::*;use ndarray::array;#[test]fn packed_training_roundtrip(){let l=RealQuantLinear::new(array![[1.0,0.0],[0.0,1.0]],None,4).unwrap();let y=l.forward(&array![[2.0,3.0]]);assert_eq!(y.shape(),&[1,2]);}#[test]fn supported_bits(){for b in [2,4,8]{assert!(RqtConfig::new(b).is_ok())}}}
