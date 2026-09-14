use smaul_native::rwkv_model::{RwkvModel, RwkvModelConfig};

#[test]
fn model_decode_reuses_state() {
    let model = RwkvModel::new(RwkvModelConfig::new(24, 16, 2, 4), 321);
    let (_, state) = model.forward(&[1, 2], None);
    let (logits, next) = model.forward(&[3], Some(&state));
    assert_eq!(logits.dim(), (1, 24));
    assert_eq!(next.rwkv_blocks.len(), 2);
}
