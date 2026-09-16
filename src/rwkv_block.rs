use crate::layer_norm::LayerNorm;
use crate::moe::{MoeBackward, MoeCmix};
use crate::rwkv_cmix::{CmixBackward, RwkvCmix};
use crate::rwkv_time_mix::{RwkvTimeMix, RwkvTimeMixTape};

use ndarray::{Array1, Array2, Array4};

#[derive(Clone)]pub struct RwkvBlockState{pub time_state:Array4<f32>,pub time_prev:Array1<f32>,pub cmix_prev:Array1<f32>,pub v_first:Option<Array2<f32>>}
pub struct RwkvBlock{pub channels:usize,pub heads:usize,pub layer_id:usize,pub ln0:Option<LayerNorm>,pub ln1:LayerNorm,pub ln2:LayerNorm,pub time_mix:RwkvTimeMix,pub cmix:RwkvCmix,pub moe:Option<MoeCmix>}
impl RwkvBlock{pub fn new(c:usize,h:usize,l:usize,n:usize)->Self{Self::new_with_moe_and_qat(c,h,l,n,false,1,1,0,0)}pub fn new_with_moe(c:usize,h:usize,l:usize,n:usize,m:bool,e:usize,k:usize,s:u64)->Self{Self::new_with_moe_and_qat_seed(c,h,l,n,m,e,k,s,0)}pub fn new_with_moe_and_qat(c:usize,h:usize,l:usize,n:usize,m:bool,e:usize,k:usize,s:u64,bits:u8)->Self{Self::new_with_moe_and_qat_seed(c,h,l,n,m,e,k,s,bits)}fn new_with_moe_and_qat_seed(c:usize,h:usize,l:usize,n:usize,m:bool,e:usize,k:usize,s:u64,bits:u8)->Self{assert!(c>0&&h>0);assert_eq!(c%h,0);let cmix=RwkvCmix::new_seeded(c,l,n,s^0x434d_4958).with_qat_bits(bits);let moe=if m{Some(MoeCmix::new(c,l,n,e,k,s).with_qat_bits(bits))}else{None};Self{channels:c,heads:h,layer_id:l,ln0:if l==0{Some(LayerNorm::new(c,1e-5))}else{None},ln1:LayerNorm::new(c,1e-5),ln2:LayerNorm::new(c,1e-5),time_mix:RwkvTimeMix::new_with_qat(c,h,l,n,bits),cmix,moe}}
pub fn forward(&self,x:&Array2<f32>,state:Option<&RwkvBlockState>,vf:Option<&Array2<f32>>)->(Array2<f32>,RwkvBlockState){let input=match&self.ln0{Some(n)=>n.forward(x),None=>x.clone()};let ti=self.ln1.forward(&input);let ts=state.map(|s|s.time_state.clone());let tp=state.map(|s|s.time_prev.clone());let iv=vf.cloned().or_else(||state.and_then(|s|s.v_first.clone()));let(to,nts,ntp,nv)=self.time_mix.forward(&ti,ts,tp,iv);let residual=&input+&to;let ci=self.ln2.forward(&residual);let cp=state.map(|s|&s.cmix_prev);let(co,ncp)=match&self.moe{Some(m)=>m.forward(&ci,cp),None=>self.cmix.forward(&ci,cp)};(&residual+&co,RwkvBlockState{time_state:nts,time_prev:ntp,cmix_prev:ncp,v_first:Some(nv)})}
pub fn forward_with_full_tape(&self,x:&Array2<f32>,state:Option<&RwkvBlockState>,vf:Option<&Array2<f32>>)->(Array2<f32>,RwkvBlockState,RwkvBlockFullTape){let input=x.clone();let(l0i,l0o)=match&self.ln0{Some(n)=>(Some(input.clone()),Some(n.forward(&input))),None=>(None,None)};let bi=l0o.clone().unwrap_or_else(||input.clone());let l1i=bi.clone();let l1o=self.ln1.forward(&l1i);let ts=state.map(|s|s.time_state.clone());let tp=state.map(|s|s.time_prev.clone());let iv=vf.cloned().or_else(||state.and_then(|s|s.v_first.clone()));let(to,nts,ntp,nv,tt)=self.time_mix.forward_with_tape(&l1o,ts,tp,iv);let residual=&bi+&to;let l2i=residual.clone();let l2o=self.ln2.forward(&l2i);let cp=state.map(|s|s.cmix_prev.clone()).unwrap_or_else(||Array1::zeros(self.channels));let(co,ncp,mixed,pre,hidden)=match&self.moe{Some(m)=>{let(o,p)=m.forward(&l2o,Some(&cp));(o,p,l2o.clone(),Array2::zeros((l2o.nrows(),self.channels*4)),Array2::zeros((l2o.nrows(),self.channels*4)))}None=>{let mut m=l2o.clone();for t in 0..m.nrows(){for c in 0..self.channels{let p=if t==0{cp[c]}else{l2o[[t-1,c]]};m[[t,c]]=l2o[[t,c]]+(p-l2o[[t,c]])*self.cmix.x_k[c]}}let(k,v)=self.cmix.effective_weights();let p=m.dot(&k);let h=p.mapv(|v|v.max(0.0).powi(2));(h.dot(&v),l2o.row(l2o.nrows()-1).to_owned(),m,p,h)}};let out=&residual+&co;let ns=RwkvBlockState{time_state:nts,time_prev:ntp,cmix_prev:ncp,v_first:Some(nv)};let mut tape=RwkvBlockFullTape::new(input,tt,ns.clone());tape.initial_state=state.cloned();tape.ln0_input=l0i;tape.ln0_output=l0o;tape.ln1_input=l1i;tape.ln1_output=l1o;tape.residual=residual;tape.ln2_input=l2i;tape.ln2_output=l2o.clone();tape.cmix_input=l2o;tape.cmix_prev=cp;tape.cmix_mixed=mixed;tape.cmix_pre=pre;tape.cmix_hidden=hidden;tape.cmix_output=co;tape.output=out.clone();(out,ns,tape)}pub fn parameter_count(&self)->usize{let n=self.ln0.as_ref().map_or(0,|x|x.weight.len()+x.bias.len())+self.ln1.weight.len()+self.ln1.bias.len()+self.ln2.weight.len()+self.ln2.bias.len();n+self.time_mix.parameter_count()+match&self.moe{Some(m)=>m.parameter_count(),None=>self.cmix.parameter_count()}}}
#[cfg(test)]mod tests_rwkv_block{use super::*;#[test]fn qat_widths_construct(){for b in[0u8,2,3,4,8]{let x=RwkvBlock::new_with_moe_and_qat(16,2,0,4,false,1,1,1,b);assert_eq!(x.cmix.qat_bits,b);}}}

// ===== rwkv_block_backward =====

pub struct BlockBackward {
    pub grad_input: Array2<f32>,
    pub grad_residual: Array2<f32>,
}

pub fn residual_backward(grad_output: &Array2<f32>) -> BlockBackward {
    BlockBackward {
        grad_input: grad_output.clone(),
        grad_residual: grad_output.clone(),
    }
}

pub fn chain_residual(
    grad_branch: &Array2<f32>,
    grad_skip: &Array2<f32>,
) -> Array2<f32> {
    assert_eq!(grad_branch.dim(), grad_skip.dim());
    grad_branch + grad_skip
}

#[cfg(test)]
mod tests_rwkv_block_backward {
    use super::*;
    use ndarray::array;

    #[test]
    fn residual_gradient_reaches_both_paths() {
        let grad = array![[1.0, 2.0]];
        let result = residual_backward(&grad);
        assert_eq!(result.grad_input, grad);
        assert_eq!(result.grad_residual, grad);
    }

    #[test]
    fn residual_paths_accumulate() {
        assert_eq!(chain_residual(&array![[1.0]], &array![[2.0]]), array![[3.0]]);
    }
}

// ===== rwkv_block_backward_full =====
pub struct RwkvBlockBackward{pub grad_input:Array2<f32>,pub grad_time_prev:Array1<f32>,pub grad_cmix_prev:Array1<f32>,pub grad_time_state:Array4<f32>,pub grad_v_first:Array2<f32>,pub grad_ln0_weight:Option<Array1<f32>>,pub grad_ln0_bias:Option<Array1<f32>>,pub grad_ln1_weight:Array1<f32>,pub grad_ln1_bias:Array1<f32>,pub grad_ln2_weight:Array1<f32>,pub grad_ln2_bias:Array1<f32>,pub time:crate::rwkv_time_mix::RwkvTimeMixBackward,pub cmix:Option<CmixBackward>,pub moe:Option<MoeBackward>}
pub fn backward(block:&RwkvBlock,tape:&RwkvBlockFullTape,go:&Array2<f32>,gns:Option<&Array4<f32>>,gnt:Option<&Array1<f32>>,gnc:Option<&Array1<f32>>,gnv:Option<&Array2<f32>>)->RwkvBlockBackward{let prev=tape.initial_state.as_ref().map(|s|&s.cmix_prev);let(gi,gp,cm,m)=match&block.moe{Some(x)=>{let mut r=crate::moe::backward(x,&tape.cmix_input,prev,go);if let Some(g)=gnc{r.grad_prev+=g}(r.grad_input.clone(),r.grad_prev.clone(),None,Some(r))}None=>{let(k,v)=block.cmix.effective_weights();let mut r=crate::rwkv_cmix::backward(&tape.cmix_input,prev,go,&block.cmix.x_k,&k,&v);if let Some(g)=gnc{r.grad_prev+=g}(r.grad_input.clone(),r.grad_prev.clone(),Some(r),None)}};let(l2i,l2w,l2b)=crate::layer_norm::backward(&tape.ln2_input,&gi,&block.ln2.weight,block.ln2.eps);let gr=go+&l2i;let mut time=crate::rwkv_time_mix::backward(&block.time_mix,&tape.time,&gr);if let Some(g)=gns{time.grad_state+=g}if let Some(g)=gnt{time.grad_prev+=g}if let Some(g)=gnv{time.grad_v_first+=g}let(l1i,l1w,l1b)=crate::layer_norm::backward(&tape.ln1_input,&time.grad_input,&block.ln1.weight,block.ln1.eps);let into=gr+&l1i;let(mut l0w,mut l0b)=(None,None);let input=if let(Some(n),Some(i))=(block.ln0.as_ref(),tape.ln0_input.as_ref()){let(d,w,b)=crate::layer_norm::backward(i,&into,&n.weight,n.eps);l0w=Some(w);l0b=Some(b);d}else{into};RwkvBlockBackward{grad_input:input,grad_time_prev:time.grad_prev.clone(),grad_cmix_prev:gp,grad_time_state:time.grad_state.clone(),grad_v_first:time.grad_v_first.clone(),grad_ln0_weight:l0w,grad_ln0_bias:l0b,grad_ln1_weight:l1w,grad_ln1_bias:l1b,grad_ln2_weight:l2w,grad_ln2_bias:l2b,time,cmix:cm,moe:m}}
#[cfg(test)]mod tests_rwkv_block_backward_full{use super::*;#[test]fn qat_backward_uses_effective_weights(){let b=RwkvBlock::new_with_moe_and_qat(8,2,0,4,false,1,1,1,8);let x=Array2::ones((2,8));let(_,_,t)=b.forward_with_full_tape(&x,None,None);let r=backward(&b,&t,&Array2::ones((2,8)),None,None,None,None);assert!(r.grad_input.iter().all(|v|v.is_finite()));}}

// ===== rwkv_block_tape =====

pub struct RwkvBlockTape {
    pub input: Array2<f32>,
    pub ln0_output: Option<Array2<f32>>,
    pub ln1_output: Array2<f32>,
    pub time_output: Array2<f32>,
    pub residual: Array2<f32>,
    pub ln2_output: Array2<f32>,
    pub cmix_output: Array2<f32>,
    pub output: Array2<f32>,
    pub time_state: Array4<f32>,
    pub time_prev: Array1<f32>,
    pub cmix_prev: Array1<f32>,
}

impl RwkvBlockTape {
    pub fn new(input: Array2<f32>, output: Array2<f32>) -> Self {
        let rows = input.nrows();
        let channels = input.ncols();
        Self {
            input,
            ln0_output: None,
            ln1_output: Array2::zeros((rows, channels)),
            time_output: Array2::zeros((rows, channels)),
            residual: Array2::zeros((rows, channels)),
            ln2_output: Array2::zeros((rows, channels)),
            cmix_output: Array2::zeros((rows, channels)),
            output,
            time_state: Array4::zeros((1, 1, 1, 1)),
            time_prev: Array1::zeros(channels),
            cmix_prev: Array1::zeros(channels),
        }
    }

    pub fn clear(&mut self) {
        self.ln0_output = None;
        self.ln1_output.fill(0.0);
        self.time_output.fill(0.0);
        self.residual.fill(0.0);
        self.ln2_output.fill(0.0);
        self.cmix_output.fill(0.0);
        self.time_state.fill(0.0);
        self.time_prev.fill(0.0);
        self.cmix_prev.fill(0.0);
    }
}

#[cfg(test)]
mod tests_rwkv_block_tape {
    use super::*;

    #[test]
    fn tape_has_all_residual_stages() {
        let tape = RwkvBlockTape::new(Array2::zeros((2, 4)), Array2::zeros((2, 4)));
        assert_eq!(tape.ln1_output.dim(), (2, 4));
        assert_eq!(tape.residual.dim(), (2, 4));
        assert_eq!(tape.cmix_output.dim(), (2, 4));
    }
}

// ===== rwkv_block_full_tape =====

pub struct RwkvBlockFullTape {
    pub input: Array2<f32>, pub ln0_input: Option<Array2<f32>>, pub ln0_output: Option<Array2<f32>>,
    pub ln1_input: Array2<f32>, pub ln1_output: Array2<f32>, pub time: RwkvTimeMixTape,
    pub residual: Array2<f32>, pub ln2_input: Array2<f32>, pub ln2_output: Array2<f32>,
    pub cmix_input: Array2<f32>, pub cmix_prev: Array1<f32>, pub cmix_mixed: Array2<f32>,
    pub cmix_pre: Array2<f32>, pub cmix_hidden: Array2<f32>, pub cmix_output: Array2<f32>,
    pub output: Array2<f32>, pub initial_state: Option<RwkvBlockState>, pub next_state: RwkvBlockState,
}

impl RwkvBlockFullTape {
    pub fn new(input: Array2<f32>, time: RwkvTimeMixTape, next_state: RwkvBlockState) -> Self {
        let rows = input.nrows();
        let channels = input.ncols();
        let shape = input.raw_dim();
        Self {
            input,
            ln0_input: None,
            ln0_output: None,
            ln1_input: Array2::zeros(shape.clone()),
            ln1_output: Array2::zeros(shape.clone()),
            time,
            residual: Array2::zeros(shape.clone()),
            ln2_input: Array2::zeros(shape.clone()),
            ln2_output: Array2::zeros(shape.clone()),
            cmix_input: Array2::zeros(shape.clone()),
            cmix_prev: Array1::zeros(channels),
            cmix_mixed: Array2::zeros(shape.clone()),
            cmix_pre: Array2::zeros((rows, channels * 4)),
            cmix_hidden: Array2::zeros((rows, channels * 4)),
            cmix_output: Array2::zeros(shape.clone()),
            output: Array2::zeros(shape),
            initial_state: None,
            next_state,
        }
    }
}
