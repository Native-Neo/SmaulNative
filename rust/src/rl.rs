use crate::inference::sample;
use crate::model_io::PretrainedModel;
use crate::training::TrainStep;
use rand::{Rng,SeedableRng};
use serde::{Deserialize,Serialize};
use std::fs::{self,OpenOptions};
use std::io::Write;
use std::path::{Path,PathBuf};

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Candidate{pub id:usize,pub text:String,pub tokens:Vec<usize>,pub old_logprobs:Vec<f32>}
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Preference{pub prompt:String,pub responses:Vec<String>,pub chosen:usize,pub source:String}

pub struct SmaulRl{pub model:PretrainedModel,pub work_dir:PathBuf,pub eos_id:usize}
impl SmaulRl{
 pub fn load(model_dir:impl AsRef<Path>,work_dir:impl AsRef<Path>)->Result<Self,String>{let model=PretrainedModel::load(model_dir)?;let work_dir=work_dir.as_ref().to_path_buf();fs::create_dir_all(&work_dir).map_err(|e|e.to_string())?;let eos_id=model.tokenizer.eos_id();Ok(Self{model,work_dir,eos_id})}
 pub fn generate(&self,prompt:&str,max_new:usize,temp:f32,top_k:usize,top_p:f32,seed:u64)->Candidate{let mut ids=self.model.tokenizer.encode(prompt);if ids.is_empty(){ids.push(self.model.tokenizer.bos_id());}let mut rng=rand::rngs::StdRng::seed_from_u64(seed);let(mut logits,mut state)=self.model.model.forward(&ids,None);let mut response=Vec::new();let mut old=Vec::new();for _ in 0..max_new{let row=logits.row(logits.nrows()-1);let values=row.as_slice().unwrap();let token=sample(values,temp,top_k,top_p,1.0,&response,&mut rng);let max=values.iter().copied().fold(f32::NEG_INFINITY,f32::max);let z: f32=values.iter().map(|x|(*x/temp.max(1e-8)-max).exp()).sum();old.push(values[token]/temp.max(1e-8)-max-z.ln());if token==self.eos_id{break;}response.push(token);(logits,state)=self.model.model.forward(&[token],Some(&state));}Candidate{id:0,text:self.model.tokenizer.decode(&response),tokens:response,old_logprobs:old}}
 pub fn candidates(&self,prompt:&str,count:usize,max_new:usize,temp:f32,top_k:usize,top_p:f32)->Vec<Candidate>{let mut rng=rand::rng();(0..count).map(|id|{let mut c=self.generate(prompt,max_new,temp,top_k,top_p,rng.random());c.id=id;c}).collect()}
 pub fn save_preference(&self,prompt:&str,candidates:&[Candidate],chosen:usize,source:&str)->Result<(),String>{if chosen>=candidates.len(){return Err("chosen response is out of range".into())}let r=Preference{prompt:prompt.into(),responses:candidates.iter().map(|x|x.text.clone()).collect(),chosen,source:source.into()};let path=self.work_dir.join("preferences.jsonl");let mut f=OpenOptions::new().create(true).append(true).open(path).map_err(|e|e.to_string())?;writeln!(f,"{}",serde_json::to_string(&r).map_err(|e|e.to_string())?).map_err(|e|e.to_string())}
 pub fn grpo_loss(candidates:&[Candidate],chosen:usize,clip:f32,kl_coef:f32)->f32{assert!(chosen<candidates.len());let n=candidates.len()as f32;let mean=0.0;let mut rewards=vec![-1.0;candidates.len()];rewards[chosen]=1.0;let m=rewards.iter().sum::<f32>()/n;let s=(rewards.iter().map(|x|(x-m)*(x-m)).sum::<f32>()/(n-1.0).max(1.0)).sqrt().max(1e-6);let mut total=0.0;for(i,c)in candidates.iter().enumerate(){let a=(rewards[i]-mean)/s;if c.old_logprobs.is_empty(){continue}for&old in &c.old_logprobs{let ratio=1.0f32;let cr=ratio.clamp(1.0-clip,1.0+clip);total+=-(ratio*a).min(cr*a)+kl_coef*(old-old);}}total/(candidates.iter().filter(|x|!x.tokens.is_empty()).map(|x|x.tokens.len()).sum::<usize>().max(1)as f32)}
 pub fn train_response(&mut self,tokens:&[usize],lr:f32)->Result<f32,String>{if tokens.len()<2{return Ok(0.0)}let targets=ndarray::Array1::from_vec(tokens[1..].to_vec());let input=&tokens[..tokens.len()-1];let step=crate::model_train_step::ModelTrainStep::run(&self.model.model,input,&targets);let loss=step.loss;let mut train=TrainStep::new(&self.model.model,lr);train.step(&mut self.model.model,&step.gradients);Ok(loss)}
 pub fn run_human(&mut self,prompts:&[String],count:usize,max_new:usize,temp:f32,top_k:usize,top_p:f32,lr:f32)->Result<(),String>{if count<2{return Err("responses must be at least 2".into())}for prompt in prompts{let c=self.candidates(prompt,count,max_new,temp,top_k,top_p);for(x,i)in c.iter().enumerate(){println!("\n[{}]\n{}",x+1,i.text)}let chosen=loop{print!("Choose best response [1-{}]: ",count);std::io::stdout().flush().ok();let mut s=String::new();std::io::stdin().read_line(&mut s).map_err(|e|e.to_string())?;if let Ok(n)=s.trim().parse::<usize>(){if (1..=count).contains(&n){break n-1}}println!("Invalid choice.")};self.save_preference(prompt,&c,chosen,"human")?;let loss=self.train_response(&c[chosen].tokens,lr)?;println!("[RL] loss={loss:.5}");}Ok(())}
}
