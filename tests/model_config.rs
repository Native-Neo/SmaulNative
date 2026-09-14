use smaul_native::rwkv_model::RwkvModelConfig;

#[test]
fn model_config_computes_head_count() {
    let config = RwkvModelConfig::new(64, 32, 6, 8);
    assert_eq!(config.n_head(), 4);
    assert_eq!(config.head_size_divisor, 8);
}
