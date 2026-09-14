use smaul_native::rwkv_model::RwkvModelConfig;

#[test]
fn model_moba_configuration_is_recorded() {
    let config = RwkvModelConfig::new(32, 16, 5, 4).with_moba(2, 8, 1);
    assert_eq!(config.n_moba_layer, 2);
    assert_eq!(config.moba_chunk_size, 8);
    assert_eq!(config.moba_topk, 1);
}
