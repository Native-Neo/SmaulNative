use smaul_native::rwkv_model::RwkvModelConfig;

#[test]
fn model_config_computes_head_count() {
    let config = RwkvModelConfig::new(64, 32, 6, 8);
    assert_eq!(config.n_head(), 4);
    assert_eq!(config.head_size_divisor, 8);
}

#[test]
fn qat_bits_three_survives_config_round_trip_and_reaches_the_model(){
 use smaul_native::config::RwkvXConfig;
 use smaul_native::rwkv_model::RwkvModel;
 let path=std::env::temp_dir().join(format!("smaul-qat3-{}.json",std::process::id()));
 let c=RwkvXConfig{vocab_size:64,n_embd:32,n_layer:3,head_size:16,n_moba_layer:1,qat_bits:3,..RwkvXConfig::default()};
 c.validate().unwrap();
 c.save(&path).unwrap();
 let back=RwkvXConfig::load(&path).unwrap();
 assert_eq!(back.effective_qat_bits(),3);
 let model=RwkvModel::new(back.to_model_config(),5);
 assert_eq!(model.config.effective_qat_bits(),3);
 // a 3-bit model must not produce the same logits as an unquantized one
 let dense=RwkvModel::new(RwkvXConfig{qat_bits:0,..c.clone()}.to_model_config(),5);
 let a=model.forward(&[1,2,3],None).0; let b=dense.forward(&[1,2,3],None).0;
 let diff=a.iter().zip(b.iter()).map(|(x,y)|(x-y).abs()).fold(0.0f32,f32::max);
 assert!(diff>1e-5,"qat_bits=3 changed nothing; it is not connected to the model");
 std::fs::remove_file(&path).ok();
}

#[test]
fn legacy_qat_3bit_flag_still_loads(){
 use smaul_native::config::RwkvXConfig;
 let path=std::env::temp_dir().join(format!("smaul-legacy-{}.json",std::process::id()));
 std::fs::write(&path,r#"{"vocab_size":64,"n_embd":32,"n_layer":3,"head_size":16,"n_moba_layer":1,"qat_3bit":true}"#).unwrap();
 assert_eq!(RwkvXConfig::load(&path).unwrap().effective_qat_bits(),3);
 std::fs::remove_file(&path).ok();
}
