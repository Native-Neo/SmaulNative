use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use smaul_native::dataset::{DatasetStream, TokenBatch};
use smaul_native::dataset::MultiFileDatasetStream;
use smaul_native::model_backward::ModelGradients;
use smaul_native::checkpoint::PretrainedModel;
use smaul_native::model_backward::ModelTrainStep;
use smaul_native::dataset::SftDataset;
use smaul_native::qat;
use smaul_native::training::{OptimizerKind, TrainStep};
use smaul_native::config::RwkvXConfig;
use smaul_native::rwkv_model::RwkvModel;
use smaul_native::tokenizer::Tokenizer;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Default)]
struct Args {
    model_dir: PathBuf,
    dataset: PathBuf,
    output_dir: PathBuf,
    mode: String,
    steps: usize,
    ctx_len: usize,
    batch_size: usize,
    grad_accum: usize,
    lr: f32,
    min_lr: f32,
    warmup: usize,
    save_every: usize,
    log_every: usize,
    resume: bool,
    optimizer: String,
    rqt_bits: u8,
}

#[derive(Serialize, Deserialize, Default)]
struct State { step: usize, tokens: usize }

enum Data { Single(DatasetStream), Multi(MultiFileDatasetStream), Sft(SftDataset) }
impl Data { fn next(&mut self)->Result<Option<TokenBatch>,String>{match self{Self::Single(x)=>x.next_batch(),Self::Multi(x)=>x.next_batch(),Self::Sft(_)=>Ok(None)}} }
fn value(args:&[String],name:&str,default:&str)->String{args.windows(2).find(|x|x[0]==name).map(|x|x[1].clone()).unwrap_or_else(||default.into())}
fn has(args:&[String],name:&str)->bool{args.iter().any(|x|x==name)}
fn number<T:std::str::FromStr>(args:&[String],name:&str,default:&str)->Result<T,String>{let raw=value(args,name,default);raw.parse().map_err(|_|format!("{name} expects a number, got '{raw}'"))}
fn parse()->Result<Args,String>{let a:Vec<String>=env::args().collect();let dataset=value(&a,"--dataset",&value(&a,"--dataset-dir","./datasets/train.jsonl"));Ok(Args{model_dir:value(&a,"--model","./SmaulNative").into(),dataset:dataset.into(),output_dir:value(&a,"--output-dir",&value(&a,"--output","./SmaulNative-trained")).into(),mode:value(&a,"--mode","pretrain"),steps:number(&a,"--steps","1000")?,ctx_len:number(&a,"--ctx-len","1024")?,batch_size:number(&a,"--batch-size","1")?,grad_accum:number(&a,"--grad-accum","1")?,lr:number(&a,"--lr","1e-4")?,min_lr:number(&a,"--min-lr","0")?,warmup:number(&a,"--warmup","0")?,save_every:number(&a,"--save-every","100")?,log_every:number(&a,"--log-every","10")?,resume:has(&a,"--resume"),optimizer:value(&a,"--optimizer","lion"),rqt_bits:number(&a,"--rqt-bits","0")?})}
fn cosine_lr(base:f32,min:f32,step:usize,total:usize,warmup:usize)->f32{if warmup>0&&step<warmup{return base*(step+1)as f32/warmup as f32;}let p=if total<=warmup{1.0}else{(step.saturating_sub(warmup)as f32/(total-warmup)as f32).min(1.0)};min+(base-min)*0.5*(1.0+(std::f32::consts::PI*p).cos())}
fn load_state(path:&Path)->State{fs::read_to_string(path).ok().and_then(|s|serde_json::from_str(&s).ok()).unwrap_or_default()}
fn save_state(path:&Path,state:&State){let _=fs::write(path,serde_json::to_string_pretty(state).unwrap());}
fn require_directory(path:&Path,label:&str)->Result<(),String>{if !path.is_dir(){return Err(format!("{label} directory does not exist: {}",path.display()));}Ok(())}
fn require_file(path:&Path,label:&str)->Result<(),String>{if !path.is_file(){return Err(format!("{label} file does not exist: {}",path.display()));}Ok(())}
fn initialize_pretrain_model(path:&Path)->Result<(),String>{let config=RwkvXConfig::default();let tokenizer=Tokenizer::character_vocab(config.vocab_size)?;let model=RwkvModel::new(config.to_model_config(),0);let pretrained=PretrainedModel{config,model,tokenizer};fs::create_dir_all(path).map_err(|e|format!("failed to create model directory {}: {e}",path.display()))?;pretrained.save(path)?;println!("[INIT] created fresh model at {}",path.display());Ok(())}
fn sft_step(model:&smaul_native::rwkv_model::RwkvModel,example:&smaul_native::dataset::SftExample,scale:f32)->ModelTrainStep{let(logits,tape,state)=model.forward_with_tape_and_state(&example.input,None);let mut grad=Array2::zeros(logits.dim());let mut loss=0.0;let mut count=0usize;for r in 0..logits.nrows(){let target=example.labels[r];if target<0{continue;}let row=logits.row(r);let m=row.iter().copied().fold(f32::NEG_INFINITY,f32::max);let mut sum=0.0;for &x in row{sum+=(x-m).exp();}let logz=m+sum.ln();loss+=logz-row[target as usize];for c in 0..logits.ncols(){grad[[r,c]]=(row[c]-m).exp()/sum;if c==target as usize{grad[[r,c]]-=1.0;}}count+=1;}let inv=1.0/count.max(1)as f32;grad.mapv_inplace(|x|x*inv*scale);ModelTrainStep::from_logits_gradient(model,&example.input,logits,tape,state,loss*inv,grad)}
fn run()->Result<(),String>{let args=parse()?;if !matches!(args.mode.as_str(),"pretrain"|"sft"){return Err("--mode must be pretrain or sft".into());}if !matches!(args.optimizer.as_str(),"lion"|"adamw"){return Err("--optimizer must be lion or adamw".into());}if !matches!(args.rqt_bits,0|2|3|4|8){return Err("--rqt-bits must be 0 (off), 2, 3, 4 or 8".into());}if args.steps==0||args.ctx_len==0||args.batch_size==0||args.grad_accum==0{return Err("steps, ctx-len, batch-size and grad-accum must be > 0".into());}
let model_path=if args.resume&&args.output_dir.join("config.json").exists(){args.output_dir.clone()}else{args.model_dir.clone()};
if args.mode=="sft"{require_file(&args.dataset,"SFT dataset")?;}else if args.dataset.is_dir(){require_directory(&args.dataset,"dataset")?;}else{require_file(&args.dataset,"dataset")?;}
if args.mode=="pretrain"&&!args.resume&&!model_path.join("config.json").is_file(){initialize_pretrain_model(&model_path)?;}
require_directory(&model_path,"model")?;require_file(&model_path.join("config.json"),"model config")?;
let mut pretrained=PretrainedModel::load(&model_path).map_err(|e|format!("failed to load model from {}: {e}",model_path.display()))?;
let mut data=if args.mode=="sft"{Data::Sft(SftDataset::open(&args.dataset,pretrained.tokenizer.clone(),args.ctx_len)?)}else if args.dataset.is_dir(){Data::Multi(MultiFileDatasetStream::open_discovered(&args.dataset,pretrained.tokenizer.clone(),args.ctx_len)?)}else{Data::Single(DatasetStream::open(&args.dataset,pretrained.tokenizer.clone(),args.ctx_len)?)};
if let Data::Sft(sft)=&data{if sft.is_empty(){return Err(format!("SFT dataset {} contains no usable examples",args.dataset.display()));}}
let state_path=args.output_dir.join("training_state.json");let mut state=if args.resume{load_state(&state_path)}else{State::default()};let kind=if args.optimizer=="adamw"{OptimizerKind::AdamW}else{OptimizerKind::Lion};let mut train=TrainStep::new_with_optimizer(&pretrained.model,args.lr,kind);let opt_path=args.output_dir.join("optimizer.bin");if args.resume&&opt_path.exists(){let bytes=fs::read(&opt_path).map_err(|e|format!("failed to read optimizer state {}: {e}",opt_path.display()))?;train.load_optimizer_state_bytes(&bytes)?;}
println!("[MODEL] {} parameters",pretrained.model.parameter_count());println!("[TRAIN] mode={} optimizer={} steps={} batch={} grad_accum={} lr={}",args.mode,args.optimizer,args.steps,args.batch_size,args.grad_accum,args.lr);if args.rqt_bits>0{println!("[RQT] {}-bit quantized forward/backward, full-precision masters",args.rqt_bits);}
while state.step<args.steps{let lr=cosine_lr(args.lr,args.min_lr,state.step,args.steps,args.warmup);train.set_lr(lr);let mut accumulated:Option<ModelGradients>=None;let mut loss_sum=0.0;let mut samples=0usize;let window=(args.grad_accum*args.batch_size)as f32;let scale=1.0/window;for _ in 0..args.grad_accum{for _ in 0..args.batch_size{let masters=if args.rqt_bits>0{Some(qat::quantize_in_place(&mut pretrained.model,args.rqt_bits)?)}else{None};let step=if args.mode=="sft"{let example=loop{match &mut data{Data::Sft(sft)=>match sft.next_example()?{Some(ex)=>break ex,None=>sft.reset()},_=>unreachable!()}};state.tokens+=example.input.len();sft_step(&pretrained.model,&example,scale)}else{let batch=match data.next()?{Some(batch)=>batch,None=>return Err("dataset ended before training reached requested steps".into())};state.tokens+=batch.input.len();let targets=Array1::from_vec(batch.target);ModelTrainStep::run_scaled(&pretrained.model,&batch.input,&targets,scale)};if let Some(m)=masters{qat::restore_masters(&mut pretrained.model,m);}loss_sum+=step.loss;samples+=1;if let Some(g)=&mut accumulated{g.add_in_place(&step.gradients);}else{accumulated=Some(step.gradients);}}}let gradients=accumulated.ok_or("no gradients accumulated")?;train.step(&mut pretrained.model,&gradients);state.step+=1;if state.step%args.log_every==0{println!("step {} | loss {:.5} | lr {:.3e} | tokens {}",state.step,loss_sum/samples.max(1)as f32,lr,state.tokens);}if state.step%args.save_every==0||state.step==args.steps{fs::create_dir_all(&args.output_dir).map_err(|e|format!("failed to create output directory {}: {e}",args.output_dir.display()))?;pretrained.save(&args.output_dir)?;fs::write(&opt_path,train.optimizer_state_bytes()).map_err(|e|format!("failed to save optimizer state {}: {e}",opt_path.display()))?;save_state(&state_path,&state);println!("[SAVE] {}",args.output_dir.display());}}
Ok(())}
fn main(){if let Err(error)=run(){eprintln!("[WORKFLOW ERROR] {error}");std::process::exit(1);}}
