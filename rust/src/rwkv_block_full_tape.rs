use ndarray::{Array1, Array2};
use crate::rwkv_block::RwkvBlockState;
use crate::rwkv_time_mix_tape::RwkvTimeMixTape;

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
        let rows = input.nrows(); let channels = input.ncols(); let shape = input.raw_dim();
        Self { input, ln0_input: None, ln0_output: None, ln1_input: Array2::zeros(shape.clone()), ln1_output: Array2::zeros(shape.clone()), time, residual: Array2::zeros(shape.clone()), ln2_input: Array2::zeros(shape.clone()), ln2_output: Array2::zeros(shape.clone()), cmix_input: Array2::zeros(shape.clone()), cmix_prev: Array1::zeros(channels), cmix_mixed: Array2::zeros(shape.clone()), cmix_pre: Array2::zeros((rows, channels * 4)), cmix_hidden: Array2::zeros((rows, channels * 4)), cmix_output: Array2::zeros(shape.clone()), output: Array2::zeros(shape), initial_state: None, next_state }
    }
}
