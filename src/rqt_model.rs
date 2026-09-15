use crate::rqt::RealQuantLinear;
use crate::rwkv_model::{RwkvModel,RwkvModelState};
use ndarray::Array2;

pub struct RqtModel { pub model: RwkvModel, pub bits: u8 }
impl RqtModel {
    pub fn new(model: RwkvModel, bits: u8) -> Result<Self,String> { if !matches!(bits,2|4|8){return Err("RQT supports 2, 4, or 8 bits".into())} Ok(Self{model,bits}) }
    pub fn refresh_quantized(&mut self) -> Result<(),String> {
        macro_rules! q { ($w:expr) => {{ let l=RealQuantLinear::new($w.clone(),None,self.bits)?; *$w=l.dequantized(); }} }
        macro_rules! q_io { ($w:expr) => {{ let l=RealQuantLinear::new($w.t().to_owned(),None,self.bits)?; *$w=l.dequantized().t().to_owned(); }} }
        q!(&mut self.model.head.weight);
        for b in &mut self.model.rwkv_blocks {
            q_io!(&mut b.time_mix.w1); q_io!(&mut b.time_mix.w2); q_io!(&mut b.time_mix.a1); q_io!(&mut b.time_mix.a2);
            if let Some(v)=&mut b.time_mix.v1 { q_io!(v); } if let Some(v)=&mut b.time_mix.v2 { q_io!(v); }
            q_io!(&mut b.time_mix.g1); q_io!(&mut b.time_mix.g2); q_io!(&mut b.time_mix.receptance); q_io!(&mut b.time_mix.key); q_io!(&mut b.time_mix.value); q_io!(&mut b.time_mix.output);
            b.time_mix.refresh_quantized();
            q_io!(&mut b.cmix.key); q_io!(&mut b.cmix.value);
            if let Some(m)=&mut b.moe { q!(&mut m.router.weight); for e in &mut m.experts { q_io!(&mut e.key); q_io!(&mut e.value); } }
        }
        for b in &mut self.model.moba_blocks {
            q!(&mut b.att.receptance.weight); q!(&mut b.att.key.weight); q!(&mut b.att.value.weight); q!(&mut b.att.output.weight);
            q_io!(&mut b.ffn.key); q_io!(&mut b.ffn.value);
        }
        Ok(())
    }
    pub fn forward(&mut self,tokens:&[usize],state:Option<&RwkvModelState>)->Result<(Array2<f32>,RwkvModelState),String>{self.refresh_quantized()?;Ok(self.model.forward(tokens,state))}
    pub fn forward_with_tape(&mut self,tokens:&[usize])->Result<(Array2<f32>,crate::model_backward::ModelBackwardTape),String>{self.refresh_quantized()?;Ok(self.model.forward_with_tape(tokens))}
    pub fn forward_with_state_and_tape(&mut self,tokens:&[usize],state:Option<&RwkvModelState>)->Result<(Array2<f32>,crate::model_backward::ModelBackwardTape,RwkvModelState),String>{self.refresh_quantized()?;Ok(self.model.forward_with_tape_and_state(tokens,state))}
    pub fn parameter_count(&self)->usize{self.model.parameter_count()}
}

#[cfg(test)]
mod tests {
    use super::*; use crate::rwkv_model::RwkvModelConfig;
    #[test] fn quantizes_every_linear_path(){let m=RwkvModel::new(RwkvModelConfig::new(32,16,2,4).with_moe(true,2,1),7);let mut q=RqtModel::new(m,4).unwrap();q.refresh_quantized().unwrap();let (y,_)=q.forward(&[1,2,3],None).unwrap();assert_eq!(y.dim(),(3,32));assert!(y.iter().all(|v|v.is_finite()));}
    #[test] fn supports_2_4_8_bits(){for bits in [2,4,8]{let m=RwkvModel::new(RwkvModelConfig::new(16,8,1,4),1);assert!(RqtModel::new(m,bits).is_ok())}}
}