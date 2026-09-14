use crate::rl::{Candidate,Preference,SmaulRl};
use std::fs::{self,OpenOptions};
use std::io::Write;
use std::path::{Path,PathBuf};
use rand::Rng;

#[derive(Clone,Debug)]
pub struct PreferenceModel{pub vocab:usize,pub dim:usize,pub embedding:Vec<f32>,pub w1:Vec<f32>,pub b1:Vec<f32>,pub w2:Vec<f32>,pub b2:f32}
impl PreferenceModel{
 pub fn new(vocab:usize,dim:usize,seed:u64)->Self{let mut r=rand::rngs::StdRng::seed_from_u64(seed);let mut randv=|n| (0..n).map(|_|r.random_range(-0.02..0.02)).collect();Self{vocab,dim,embedding:randv(vocab*dim),w1:randv(dim*dim),b1:vec![0.0;dim],w2:randv(dim),b2:0.0}}
 fn score(&self,tokens:&[usize])->f32{let mut h=vec![0.0;self.dim];if tokens.is_empty(){return self.b2}for &id in tokens{if id>=self.vocab{continue}for d in 0..self.dim{h[d]+=self.embedding[id*self.dim+d]}}for x in &mut h{*x/=tokens.len()as f32;*x=x.tanh()}self.b2+h.iter().zip(&self.w2).map(|(a,b)|a*b).sum::<f32>()}
 pub fn scores(&self,token_lists:&[Vec<usize>])->Vec<f32>{token_lists.iter().map(|x|self.score(x)).collect()}
 pub fn train_pair(&mut self,a:&[usize],b:&[usize],chosen_a:bool,lr:f32)->f32{let sa=self.score(a);let sb=self.score(b);let diff=if chosen_a{sa-sb}else{sb-sa};let sig=1.0/(1.0+(-diff).exp());let grad=-(1.0-sig);let target=if chosen_a{1.0}else{-1.0};for&ids in [a,b].iter(){let sign=if ids as *const _==&a as *const _{target}else{-target};for&id in *ids{if id>=self.vocab{continue}for d in 0..self.dim{self.embedding[id*self.dim+d]-=lr*grad*sign/self.dim as f32;}}} -diff.max(-50.0).min(50.0).exp().ln_1p()}
 pub fn save(&self,path:impl AsRef<Path>)->Result<(),String>{let mut s=String::new();s.push_str(&format!("{} {}\n",self.vocab,self.dim));for x in self.embedding.iter().chain(&self.w1).chain(&self.b1).chain(&self.w2){s.push_str(&format!("{x:.9}\n"));}s.push_str(&format!("{:.9}\n",self.b2));fs::write(path,s).map_err(|e|e.to_string())}
}

pub struct AutoRl{pub rl:SmaulRl,pub scorer:PreferenceModel,pub scorer_path:PathBuf}
impl AutoRl{
 pub fn load(model_dir:impl AsRef<Path>,work_dir:impl AsRef<Path>)->Result<Self,String>{let rl=SmaulRl::load(model_dir,&work_dir)?;let path=work_dir.as_ref().join("preference_model.rust");let scorer=PreferenceModel::new(rl.model.tokenizer.vocab_size(),32,1);Ok(Self{rl,scorer,scorer_path:path})}
 pub fn preference_count(&self)->usize{let p=self.rl.work_dir.join("preferences.jsonl");fs::read_to_string(p).map(|s|s.lines().filter(|x|!x.trim().is_empty()).count()).unwrap_or(0)}
 pub fn train_preferences(&mut self,epochs:usize,lr:f32)->Result<(),String>{let p=self.rl.work_dir.join("preferences.jsonl");let text=match fs::read_to_string(p){Ok(x)=>x,Err(_)=>return Ok(())};let records:Vec<Preference>=text.lines().filter_map(|x|serde_json::from_str(x).ok()).collect();for _ in 0..epochs{for r in &records{if r.chosen>=r.responses.len(){continue}let chosen=self.rl.model.tokenizer.encode(&r.responses[r.chosen]);for(i,response)in r.responses.iter().enumerate(){if i==r.chosen{continue}let rejected=self.rl.model.tokenizer.encode(response);let _=self.scorer.train_pair(&chosen,&rejected,true,lr);}}}self.scorer.save(&self.scorer_path)}
 pub fn choose(&self,prompt:&str,candidates:&[Candidate])->usize{let prompt_ids=self.rl.model.tokenizer.encode(prompt);candidates.iter().enumerate().map(|(i,c)|(i,self.scorer.score(&[prompt_ids.clone(),c.tokens.clone()].concat()))).max_by(|a,b|a.1.partial_cmp(&b.1).unwrap()).map(|x|x.0).unwrap_or(0)}
 pub fn run(&mut self,prompts:&[String],count:usize,max_new:usize,temp:f32,top_k:usize,top_p:f32,verify:bool,preference_epochs:usize,preference_lr:f32,rl_lr:f32)->Result<(),String>{self.train_preferences(preference_epochs,preference_lr)?;for prompt in prompts{let candidates=self.rl.candidates(prompt,count,max_new,temp,top_k,top_p);let predicted=self.choose(prompt,&candidates);let chosen=if verify{loop{println!("Automated choice: {}",predicted+1);print!("Correct? [Y/n]: ");std::io::stdout().flush().ok();let mut s=String::new();std::io::stdin().read_line(&mut s).map_err(|e|e.to_string())?;match s.trim().to_lowercase().as_str(){""|"y"|"yes"=>break predicted,"n"|"no"=>{for(i,c)in candidates.iter().enumerate(){println!("\n[{}]\n{}",i+1,c.text)}print!("Better [1-{}]: ",count);std::io::stdout().flush().ok();let mut x=String::new();std::io::stdin().read_line(&mut x).ok();if let Ok(n)=x.trim().parse::<usize>(){if (1..=count).contains(&n){break n-1}}},_=>println!("Please answer yes or no.")}}}else{predicted};self.rl.save_preference(prompt,&candidates,chosen,if chosen==predicted{"auto_confirmed"}else{"human_correction"})?;self.train_preferences(preference_epochs,preference_lr)?;let loss=self.rl.train_response(&candidates[chosen].tokens,rl_lr)?;println!("[RL] loss={loss:.5}");}Ok(())}
}
