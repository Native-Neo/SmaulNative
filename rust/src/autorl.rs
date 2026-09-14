use crate::rl::{Candidate,Preference,SmaulRl};
use rand::{Rng,SeedableRng};
use std::fs;
use std::io::Write;
use std::path::{Path,PathBuf};

#[derive(Clone,Debug)]
pub struct PreferenceModel{pub vocab:usize,pub dim:usize,pub embedding:Vec<f32>,pub w2:Vec<f32>,pub b2:f32}
impl PreferenceModel{
 pub fn new(vocab:usize,dim:usize,seed:u64)->Self{let mut r=rand::rngs::StdRng::seed_from_u64(seed);let mut v=||r.random_range(-0.02..0.02);Self{vocab,dim,embedding:(0..vocab*dim).map(|_|v()).collect(),w2:(0..dim).map(|_|v()).collect(),b2:0.0}}
 pub fn score(&self,tokens:&[usize])->f32{if tokens.is_empty(){return self.b2}let mut h=vec![0.0;self.dim];for&id in tokens{if id<self.vocab{for d in 0..self.dim{h[d]+=self.embedding[id*self.dim+d]}}}for x in &mut h{*x=(*x/tokens.len()as f32).tanh()}self.b2+h.iter().zip(&self.w2).map(|(a,b)|a*b).sum::<f32>()}
 pub fn train_pair(&mut self,a:&[usize],b:&[usize],lr:f32){let diff=self.score(a)-self.score(b);let grad=-(1.0-1.0/(1.0+(-diff).exp()));for(ids,sign)in[(a,1.0f32),(b,-1.0)]{for&id in ids{if id<self.vocab{for d in 0..self.dim{self.embedding[id*self.dim+d]-=lr*grad*sign/self.dim as f32;}}}}}
 pub fn save(&self,path:impl AsRef<Path>)->Result<(),String>{let mut s=format!("{} {}\n",self.vocab,self.dim);for x in self.embedding.iter().chain(&self.w2){s.push_str(&format!("{x:.9}\n"));}s.push_str(&format!("{:.9}\n",self.b2));fs::write(path,s).map_err(|e|e.to_string())}
}

pub struct AutoRl{pub rl:SmaulRl,pub scorer:PreferenceModel,pub scorer_path:PathBuf}
impl AutoRl{
 pub fn load(model_dir:impl AsRef<Path>,work_dir:impl AsRef<Path>)->Result<Self,String>{let rl=SmaulRl::load(model_dir,&work_dir)?;let scorer_path=work_dir.as_ref().join("preference_model.rust");Ok(Self{scorer:PreferenceModel::new(rl.model.tokenizer.vocab_size(),32,1),scorer_path,rl})}
 pub fn preference_count(&self)->usize{fs::read_to_string(self.rl.work_dir.join("preferences.jsonl")).map(|s|s.lines().filter(|x|!x.trim().is_empty()).count()).unwrap_or(0)}
 pub fn train_preferences(&mut self,epochs:usize,lr:f32)->Result<(),String>{let text=match fs::read_to_string(self.rl.work_dir.join("preferences.jsonl")){Ok(x)=>x,Err(_)=>return Ok(())};let records:Vec<Preference>=text.lines().filter_map(|x|serde_json::from_str(x).ok()).collect();for _ in 0..epochs{for r in &records{if r.chosen>=r.responses.len(){continue}let a=self.rl.model.tokenizer.encode(&r.responses[r.chosen]);for(i,x)in r.responses.iter().enumerate(){if i!=r.chosen{let b=self.rl.model.tokenizer.encode(x);self.scorer.train_pair(&a,&b,lr)}}}}self.scorer.save(&self.scorer_path)}
 pub fn choose(&self,prompt:&str,candidates:&[Candidate])->usize{let p=self.rl.model.tokenizer.encode(prompt);candidates.iter().enumerate().max_by(|(_,a),(_,b)|self.scorer.score(&[p.clone(),a.tokens.clone()].concat()).partial_cmp(&self.scorer.score(&[p.clone(),b.tokens.clone()].concat())).unwrap()).map(|x|x.0).unwrap_or(0)}
 pub fn run(&mut self,prompts:&[String],count:usize,max_new:usize,temp:f32,top_k:usize,top_p:f32,verify:bool,epochs:usize,pref_lr:f32,rl_lr:f32)->Result<(),String>{if count<2{return Err("responses must be at least 2".into())}self.train_preferences(epochs,pref_lr)?;for prompt in prompts{let c=self.rl.candidates(prompt,count,max_new,temp,top_k,top_p);let predicted=self.choose(prompt,&c);let chosen=if verify{loop{println!("Automated choice: {}",predicted+1);print!("Correct? [Y/n]: ");std::io::stdout().flush().ok();let mut s=String::new();std::io::stdin().read_line(&mut s).map_err(|e|e.to_string())?;match s.trim().to_lowercase().as_str(){""|"y"|"yes"=>break predicted,"n"|"no"=>{for(i,x)in c.iter().enumerate(){println!("\n[{}]\n{}",i+1,x.text)}print!("Better [1-{}]: ",count);std::io::stdout().flush().ok();let mut x=String::new();std::io::stdin().read_line(&mut x).ok();if let Ok(n)=x.trim().parse::<usize>(){if (1..=count).contains(&n){break n-1}}},_=>println!("Please answer yes or no.")}}}else{predicted};self.rl.save_preference(prompt,&c,chosen,if chosen==predicted{"auto_confirmed"}else{"human_correction"})?;self.train_preferences(epochs,pref_lr)?;println!("[RL] loss={:.5}",self.rl.train_response(&c[chosen].tokens,rl_lr)?);}Ok(())}
}
