use smaul_native::inference::Inference;
use smaul_native::checkpoint::PretrainedModel;
use smaul_native::model_backward::ModelTrainStep;
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
    let text=engine.generate("a",4,1.0,0,1.0,1.0,1);
    assert!(text.len()<=4);
    let tokens=loaded.tokenizer.encode("ab");
    let step=ModelTrainStep::run(&loaded.model,&tokens,&Array1::from_vec(vec![6,7]));
    assert!(step.loss.is_finite());
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn a_loaded_checkpoint_can_still_be_trained() {
    use ndarray::Array1;
    use smaul_native::model_backward::ModelTrainStep;
    use smaul_native::rwkv_model::{RwkvModel, RwkvModelConfig};
    use smaul_native::training::TrainStep;

    let config = RwkvModelConfig::new(24, 16, 2, 8);
    let model = RwkvModel::new(config.clone(), 3);
    let path = std::env::temp_dir().join(format!("smaul-train-after-load-{}.safetensors", std::process::id()));
    smaul_native::checkpoint::save_model_safetensors(&model, &path).unwrap();

    let mut loaded = RwkvModel::new(config, 99);
    loaded.load_safetensors(&path).unwrap();
    std::fs::remove_file(&path).ok();

    // every parameter must come back row-major, or the optimizer cannot read it
    assert!(loaded.head.weight.as_slice().is_some());
    for block in &loaded.rwkv_blocks {
        for w in [&block.time_mix.receptance, &block.time_mix.key, &block.time_mix.value, &block.time_mix.output, &block.cmix.key, &block.cmix.value] {
            assert!(w.as_slice().is_some(), "checkpoint produced a non-contiguous parameter");
        }
    }

    let tokens = [1usize, 2, 3];
    let targets = Array1::from_vec(vec![2, 3, 4]);
    let step = ModelTrainStep::run(&loaded, &tokens, &targets);
    let before = loaded.head.weight[[0, 0]];
    TrainStep::new(&loaded, 1e-3).step(&mut loaded, &step.gradients);
    assert_ne!(before, loaded.head.weight[[0, 0]]);
}
