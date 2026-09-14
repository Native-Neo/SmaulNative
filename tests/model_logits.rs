use smaul_native::rwkv_model::{RwkvModel, RwkvModelConfig};

#[test]
fn model_logits_match_token_count_and_vocab_size() {
    let model = RwkvModel::new(RwkvModelConfig::new(24, 16, 2, 4), 321);
    let (logits, _) = model.forward(&[1, 2, 3], None);
    assert_eq!(logits.dim(), (3, 24));
    assert!(logits.iter().all(|v| v.is_finite()));
}
