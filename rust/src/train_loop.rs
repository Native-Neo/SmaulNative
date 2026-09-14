use crate::dataset::{MultiFileTextStream, TextStream};
use crate::model_train_step::ModelTrainStep;
use crate::rwkv_model::RwkvModel;
use crate::training::TrainStep;
use crate::training_state::{CheckpointBundle, TrainingState};
use ndarray::Array1;
use std::path::{Path, PathBuf};

pub struct TrainingConfig {
    pub max_steps: usize,
    pub log_every: usize,
    pub save_every: usize,
    pub max_grad_norm: Option<f32>,
    pub checkpoint_dir: Option<PathBuf>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self { max_steps: 0, log_every: 1, save_every: 0, max_grad_norm: Some(1.0), checkpoint_dir: None }
    }
}

pub struct TrainingRunner { pub optimizer: TrainStep, pub state: TrainingState }

impl TrainingRunner {
    pub fn new(model: &RwkvModel, learning_rate: f32) -> Self {
        Self { optimizer: TrainStep::new(model, learning_rate), state: TrainingState::new() }
    }

    pub fn run_stream(&mut self, model: &mut RwkvModel, stream: &mut TextStream, config: &TrainingConfig) -> Result<TrainingState, String> {
        self.run_batches(model, config, || stream.next_batch())
    }

    pub fn run_multi_stream(&mut self, model: &mut RwkvModel, stream: &mut MultiFileTextStream, config: &TrainingConfig) -> Result<TrainingState, String> {
        self.run_batches(model, config, || stream.next_batch())
    }

    fn run_batches<F>(&mut self, model: &mut RwkvModel, config: &TrainingConfig, mut next: F) -> Result<TrainingState, String>
    where F: FnMut() -> Result<Option<crate::dataset::TokenBatch>, String> {
        if config.log_every == 0 { return Err("log_every must be greater than zero".into()); }
        if config.save_every > 0 && config.checkpoint_dir.is_none() { return Err("checkpoint_dir is required when save_every is enabled".into()); }

        while config.max_steps == 0 || self.state.step < config.max_steps {
            let Some(batch) = next()? else { break };
            self.state.record_micro_step(batch.input.len());
            let targets = Array1::from_vec(batch.target);
            let result = ModelTrainStep::run(model, &batch.input, &targets);
            self.optimizer.max_grad_norm = config.max_grad_norm;
            self.optimizer.step(model, &result.gradients);
            self.state.record_optimizer_step();
            if self.state.step % config.log_every == 0 { println!("step {} | loss {:.6} | tokens {}", self.state.step, result.loss, self.state.tokens_seen); }
            if config.save_every > 0 && self.state.step % config.save_every == 0 { self.save_checkpoint(model, config.checkpoint_dir.as_ref().unwrap())?; }
            self.state.reset_accumulation();
        }
        Ok(self.state.clone())
    }

    fn save_checkpoint(&self, model: &RwkvModel, directory: &Path) -> Result<(), String> {
        std::fs::create_dir_all(directory).map_err(|e| format!("failed to create checkpoint directory: {e}"))?;
        let step = self.state.step;
        crate::model_saver::save_model_safetensors(model, directory.join(format!("model-{step}.safetensors")))?;
        CheckpointBundle::new(self.state.clone()).save(directory.join(format!("training-{step}.json")))?;
        Ok(())
    }

    pub fn save_state(&self, path: impl AsRef<Path>) -> Result<(), String> {
        CheckpointBundle::new(self.state.clone()).save(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rwkv_model::RwkvModelConfig;
    use crate::tokenizer::Tokenizer;
    use std::fs;

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_vocab(vec!["<pad>".into(), "<unk>".into(), "<bos>".into(), "<eos>".into(), "<cap>".into(), "<upper>".into(), "a".into(), "b".into(), " ".into()])
    }

    #[test]
    fn trains_native_stream() {
        let path = std::env::temp_dir().join("smaul-train.txt");
        fs::write(&path, "a b a b a b a b").unwrap();
        let mut stream = TextStream::open(&path, tokenizer(), 3).unwrap();
        let mut model = RwkvModel::new(RwkvModelConfig::new(9, 8, 1, 4), 42);
        let mut runner = TrainingRunner::new(&model, 1e-4);
        let state = runner.run_stream(&mut model, &mut stream, &TrainingConfig { max_steps: 1, ..Default::default() }).unwrap();
        assert_eq!(state.step, 1);
        assert!(state.tokens_seen >= 3);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn trains_multiple_files() {
        let dir = std::env::temp_dir().join("smaul-train-multi");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.txt"), "a b a b a b").unwrap();
        fs::write(dir.join("b.txt"), "b a b a b a").unwrap();
        let mut stream = MultiFileTextStream::open_discovered(&dir, tokenizer(), 3).unwrap();
        let mut model = RwkvModel::new(RwkvModelConfig::new(9, 8, 1, 4), 42);
        let mut runner = TrainingRunner::new(&model, 1e-4);
        let state = runner.run_multi_stream(&mut model, &mut stream, &TrainingConfig { max_steps: 2, ..Default::default() }).unwrap();
        assert_eq!(state.step, 2);
        let _ = fs::remove_dir_all(dir);
    }
}
