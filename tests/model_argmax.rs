use ndarray::array;
use smaul_native::rwkv_model::RwkvModel;

#[test]
fn model_argmax_returns_vocab_indices() {
    let logits = array![[0.0, 4.0, 2.0], [7.0, 1.0, 6.0]];
    assert_eq!(RwkvModel::argmax(&logits).to_vec(), vec![1, 0]);
}
