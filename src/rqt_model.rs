use crate::rqt::RealQuantLinear;
use crate::rwkv_model::{RwkvModel,RwkvModelState};
use ndarray::Array2;

/// Real Quantization Training: the model keeps full-precision master weights and
/// every forward pass runs on the quantized values. The masters are restored
/// afterwards so the optimizer keeps updating full precision, which is what lets
/// updates smaller than a quantization step accumulate instead of being erased.
pub struct RqtModel { pub model: RwkvModel, pub bits: u8 }

/// Visits every weight matrix RQT quantizes. `io` marks nn.Linear-style weights,
/// which are stored transposed relative to the row-wise quantization grid.
fn visit_linears(model: &mut RwkvModel, mut f: impl FnMut(&mut Array2<f32>, bool)) {
    f(&mut model.head.weight, false);
    for b in &mut model.rwkv_blocks {
        let t = &mut b.time_mix;
        for w in [&mut t.w1, &mut t.w2, &mut t.a1, &mut t.a2, &mut t.g1, &mut t.g2, &mut t.receptance, &mut t.key, &mut t.value, &mut t.output] { f(w, true); }
        if let Some(v) = &mut t.v1 { f(v, true); }
        if let Some(v) = &mut t.v2 { f(v, true); }
        f(&mut b.cmix.key, true); f(&mut b.cmix.value, true);
        if let Some(m) = &mut b.moe { f(&mut m.router.weight, false); for e in &mut m.experts { f(&mut e.key, true); f(&mut e.value, true); } }
    }
    for b in &mut model.moba_blocks {
        f(&mut b.att.receptance.weight, false); f(&mut b.att.key.weight, false); f(&mut b.att.value.weight, false); f(&mut b.att.output.weight, false);
        f(&mut b.ffn.key, true); f(&mut b.ffn.value, true);
    }
}

fn refresh_packed(model: &mut RwkvModel) { for b in &mut model.rwkv_blocks { b.time_mix.refresh_quantized(); } }

/// Replaces every weight with its quantized value and returns the full-precision
/// masters. Pair with `restore_masters` around a forward/backward pass.
pub fn quantize_in_place(model: &mut RwkvModel, bits: u8) -> Result<Vec<Array2<f32>>,String> {
    if !matches!(bits,2|3|4|8) { return Err("RQT supports 2, 3, 4, or 8 bits".into()) }
    let mut masters = Vec::new();
    let mut error = None;
    visit_linears(model, |w, io| {
        if error.is_some() { return; }
        masters.push(w.clone());
        let source = if io { w.t().to_owned() } else { w.clone() };
        match RealQuantLinear::new(source, None, bits) {
            Ok(l) => { let d = l.dequantized(); *w = if io { d.t().to_owned() } else { d }; }
            Err(e) => error = Some(e),
        }
    });
    match error { Some(e) => { restore_masters(model, masters); Err(e) } None => { refresh_packed(model); Ok(masters) } }
}

pub fn restore_masters(model: &mut RwkvModel, masters: Vec<Array2<f32>>) {
    let mut it = masters.into_iter();
    visit_linears(model, |w, _| { if let Some(m) = it.next() { *w = m; } });
    refresh_packed(model);
}

impl RqtModel {
    pub fn new(model: RwkvModel, bits: u8) -> Result<Self,String> { if !matches!(bits,2|3|4|8){return Err("RQT supports 2, 3, 4, or 8 bits".into())} Ok(Self{model,bits}) }

    pub fn take_masters(&mut self) -> Result<Vec<Array2<f32>>,String> { quantize_in_place(&mut self.model, self.bits) }
    pub fn restore_masters(&mut self, masters: Vec<Array2<f32>>) { restore_masters(&mut self.model, masters) }

    /// Runs `f` with quantized weights in place, then restores the masters.
    fn with_quantized<T>(&mut self, f: impl FnOnce(&mut RwkvModel) -> T) -> Result<T,String> {
        let masters = self.take_masters()?;
        let out = f(&mut self.model);
        self.restore_masters(masters);
        Ok(out)
    }

    /// Permanently collapses the masters onto the quantization grid. Use at export time.
    pub fn freeze_quantized(&mut self) -> Result<(),String> { self.take_masters()?; Ok(()) }

    pub fn forward(&mut self,tokens:&[usize],state:Option<&RwkvModelState>)->Result<(Array2<f32>,RwkvModelState),String>{self.with_quantized(|m|m.forward(tokens,state))}
    pub fn forward_with_tape(&mut self,tokens:&[usize])->Result<(Array2<f32>,crate::model_backward::ModelBackwardTape),String>{self.with_quantized(|m|m.forward_with_tape(tokens))}
    pub fn forward_with_state_and_tape(&mut self,tokens:&[usize],state:Option<&RwkvModelState>)->Result<(Array2<f32>,crate::model_backward::ModelBackwardTape,RwkvModelState),String>{self.with_quantized(|m|m.forward_with_tape_and_state(tokens,state))}
    pub fn parameter_count(&self)->usize{self.model.parameter_count()}
}

#[cfg(test)]
mod tests {
    use super::*; use crate::rwkv_model::RwkvModelConfig;
    fn model() -> RwkvModel { RwkvModel::new(RwkvModelConfig::new(32, 16, 2, 4).with_moe(true, 2, 1), 7) }

    #[test] fn quantizes_every_linear_path(){let mut q=RqtModel::new(model(),4).unwrap();let(y,_)=q.forward(&[1,2,3],None).unwrap();assert_eq!(y.dim(),(3,32));assert!(y.iter().all(|v|v.is_finite()));}
    #[test] fn supports_2_3_4_8_bits(){for bits in [2,3,4,8]{assert!(RqtModel::new(model(),bits).is_ok())}assert!(RqtModel::new(model(),5).is_err());}

    #[test] fn forward_preserves_full_precision_masters(){
        let mut q=RqtModel::new(model(),2).unwrap();
        let before=q.model.head.weight.clone();
        q.forward(&[1,2,3],None).unwrap();
        assert_eq!(q.model.head.weight,before,"RQT forward must not overwrite the master weights");
    }

    #[test] fn sub_step_updates_accumulate_in_the_masters(){
        // A 2-bit grid is coarse, so a small update is invisible in the quantized
        // weights but must still survive in the masters and eventually cross a level.
        let mut q=RqtModel::new(model(),2).unwrap();
        let start=q.model.head.weight.clone();
        for _ in 0..200 { q.forward(&[1,2],None).unwrap(); q.model.head.weight.mapv_inplace(|v|v+1e-3); }
        let expected=start.mapv(|v|v+0.2);
        let drift=(&q.model.head.weight-&expected).iter().map(|v|v.abs()).fold(0.0f32,f32::max);
        assert!(drift<1e-4,"200 sub-quantization-step updates drifted by {drift}; the masters are being rewritten");
    }

    #[test] fn forward_actually_uses_quantized_weights(){
        let reference=model().forward(&[1,2,3],None).0;
        let mut q=RqtModel::new(model(),2).unwrap();
        let (y,_)=q.forward(&[1,2,3],None).unwrap();
        let diff=y.iter().zip(reference.iter()).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
        assert!(diff>1e-4,"2-bit RQT logits matched full precision exactly ({diff}); nothing was quantized");
    }

    #[test] fn freeze_collapses_masters_onto_the_grid(){
        let mut q=RqtModel::new(model(),2).unwrap();
        q.freeze_quantized().unwrap();
        let frozen=q.model.head.weight.clone();
        q.freeze_quantized().unwrap();
        assert_eq!(q.model.head.weight,frozen,"quantization must be idempotent once frozen");
    }
}
