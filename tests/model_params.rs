use smaul_native::rwkv_model::{RwkvModel, RwkvModelConfig};

#[test]
fn model_has_parameters() {
    let model = RwkvModel::new(RwkvModelConfig::new(32, 16, 2, 4), 123);
    assert!(model.parameter_count() > 0);
}
