use smaul_native::inference::Inference;
use smaul_native::model_io::PretrainedModel;
use smaul_native::model_train_step::ModelTrainStep;
use smaul_native::rwkv_model::{RwkvModel,RwkvModelConfig};
use smaul_native::tokenizer::{BOS,CAP,EOS,PAD,UNK,UPPER,Tokenizer};
use ndarray::Array1;

#[test]
fn checkpoint_load_inference_and_training_roundtrip(){
    let dir=std::env::temp_dir().join(format!("smaul-e2e-{}",std::process::id()));
    let tokenizer=Tokenizer::from_vocab(vec![PAD.into(),UNK.into(),BOS.into(),EOS.into(),CAP.into(),UPPER.into(),"a".into(),"b".into()," ".into(),".".into()]);
    let config=RwkvModelConfig::new(tokenizer.vocab_size(),8,1,4);
    let model=RwkvModel::new(config.clone(),7);
    let cfg=smaul_native::config::RwkvXConfig::from_model_config(&config);
    let pretrained=PretrainedModel{config:cfg,model,tokenizer};
    pretrained.save(&dir).unwrap();
    let loaded=PretrainedModel::load(&dir).unwrap();
    assert_eq!(loaded.model.parameter_count(),pretrained.model.parameter_count());
    let engine=Inference{model:&loaded.model,tokenizer:&loaded.tokenizer,eos_id:loaded.tokenizer.eos_id()};
    let text=engine.generate("a",4,1.0,0,1.0,1.0,&[],1);
    assert!(text.is_some());
    let tokens=loaded.tokenizer.encode("ab");
    let step=ModelTrainStep::run(&loaded.model,&tokens,&Array1::from_vec(vec![6,7]));
    assert!(step.loss.is_finite());
    std::fs::remove_dir_all(dir).ok();
}
