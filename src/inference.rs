use crate::checkpoint::PretrainedModel;
use crate::gguf::load_gguf;
use crate::rwkv_model::RwkvModel;
use crate::tokenizer::Tokenizer;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use serde_json::json;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn sample(logits: &[f32], temperature: f32, top_k: usize, top_p: f32, repetition_penalty: f32, recent: &[usize], rng: &mut impl Rng) -> usize {
    assert!(temperature.is_finite() && temperature >= 0.0 && top_p.is_finite() && top_p > 0.0 && top_p <= 1.0 && repetition_penalty.is_finite() && repetition_penalty > 0.0);
    let mut scores = logits.to_vec();
    if repetition_penalty != 1.0 { for &id in recent { if id < scores.len() { scores[id] = if scores[id] > 0.0 { scores[id] / repetition_penalty } else { scores[id] * repetition_penalty }; } } }
    if temperature == 0.0 { return argmax(&scores); }
    for x in &mut scores { *x /= temperature; }
    let mut ids: Vec<usize> = (0..scores.len()).collect();
    ids.sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
    if top_k > 0 && top_k < ids.len() { ids.truncate(top_k); }
    let max = ids.iter().map(|&i| scores[i]).fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = ids.into_iter().map(|i| (i, (scores[i] - max).exp())).collect();
    let total: f32 = probs.iter().map(|(_, p)| *p).sum();
    if total == 0.0 || !total.is_finite() { return argmax(&scores); }
    for (_, p) in &mut probs { *p /= total; }
    if top_p < 1.0 {
        let mut cumulative = 0.0;
        let mut keep = probs.len();
        for i in 0..probs.len() { cumulative += probs[i].1; if cumulative > top_p { keep = i + 1; break; } }
        probs.truncate(keep.max(1));
        let total: f32 = probs.iter().map(|(_, p)| *p).sum();
        for (_, p) in &mut probs { *p /= total; }
    }
    let mut r = rng.random::<f32>();
    for (id, p) in probs { if r <= p { return id; } r -= p; }
    argmax(logits)
}

fn argmax(values: &[f32]) -> usize { values.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(i,_)| i).unwrap_or(0) }

pub struct Inference<'a> { pub model: &'a RwkvModel, pub tokenizer: &'a Tokenizer, pub eos_id: usize }

impl<'a> Inference<'a> {
    fn validate(max_new_tokens: usize, temperature: f32, top_k: usize, top_p: f32, repetition_penalty: f32) -> Result<(), String> {
        let _ = (max_new_tokens, top_k);
        if !temperature.is_finite() || temperature < 0.0 { return Err("temperature must be non-negative".into()); }
        if !top_p.is_finite() || !(0.0 < top_p && top_p <= 1.0) { return Err("top_p must be in (0, 1]".into()); }
        if !repetition_penalty.is_finite() || repetition_penalty <= 0.0 { return Err("repetition_penalty must be positive".into()); }
        Ok(())
    }

    pub fn generate(&self, prompt: &str, max_new_tokens: usize, temperature: f32, top_k: usize, top_p: f32, repetition_penalty: f32, seed: u64) -> String {
        self.stream(prompt,max_new_tokens,temperature,top_k,top_p,repetition_penalty,&[],seed).collect()
    }

    pub fn stream<'b>(&'b self, prompt: &str, max_new_tokens: usize, temperature: f32, top_k: usize, top_p: f32, repetition_penalty: f32, stop: &'b [&'b str], seed: u64) -> impl Iterator<Item=String> + 'b {
        Self::validate(max_new_tokens,temperature,top_k,top_p,repetition_penalty).expect("invalid generation arguments");
        let mut tokens=self.tokenizer.encode(prompt); if tokens.is_empty(){tokens.push(self.tokenizer.bos_id());}
        let mut rng=rand::rngs::StdRng::seed_from_u64(seed); let (mut logits,mut state)=self.model.forward(&tokens,None);
        let mut recent=tokens.iter().copied().rev().take(128).collect::<Vec<_>>(); recent.reverse();
        let mut generated=Vec::new(); let mut emitted=String::new(); let mut chunks=Vec::new();
        for _ in 0..max_new_tokens {
            let token=sample(logits.row(logits.nrows()-1).as_slice().unwrap(),temperature,top_k,top_p,repetition_penalty,&recent,&mut rng); if token==self.eos_id{break;}
            generated.push(token); recent.push(token); if recent.len()>128{recent.remove(0);}
            let current=self.tokenizer.decode(&generated); let mut end=current.len();
            for &marker in stop { if !marker.is_empty(){if let Some(pos)=current[emitted.len()..].find(marker){end=end.min(emitted.len()+pos);}} }
            if end>emitted.len(){chunks.push(current[emitted.len()..end].to_owned());} emitted=current[..end].to_owned(); if end<current.len(){break;}
            (logits,state)=self.model.forward(&[token],Some(&state));
        }
        chunks.into_iter()
    }

    pub fn chat_prompt(&self, messages:&[(&str,&str)], system:Option<&str>)->String { let mut out=String::new(); if let Some(system)=system{out.push_str("System:\n");out.push_str(system);out.push_str("\n\n");} for &(role,content) in messages{let role=if role.is_empty(){"user"}else{role};let mut chars=role.chars();let label=chars.next().map(|c|c.to_uppercase().collect::<String>()+chars.as_str()).unwrap_or_default();out.push_str(&label);out.push_str(":\n");out.push_str(content);out.push_str("\n\n");}out.push_str("Assistant:\n");out }
    pub fn chat_generate(&self,messages:&[(&str,&str)],system:Option<&str>,max_new_tokens:usize,temperature:f32,top_k:usize,top_p:f32,repetition_penalty:f32,seed:u64)->String{self.generate(&self.chat_prompt(messages,system),max_new_tokens,temperature,top_k,top_p,repetition_penalty,seed)}
}

#[cfg(test)]
mod tests_inference { use super::*; use crate::rwkv_model::RwkvModelConfig;
    #[test] fn greedy_sampling_selects_maximum(){let mut rng=rand::rngs::StdRng::seed_from_u64(1);assert_eq!(sample(&[1.0,4.0,2.0],0.0,0,1.0,1.0,&[],&mut rng),1);}
    #[test] fn invalid_generation_arguments_are_rejected(){assert!(Inference::validate(0,-1.0,0,1.0,1.0).is_err());assert!(Inference::validate(0,0.7,0,0.0,1.0).is_err());assert!(Inference::validate(0,0.7,0,1.0,0.0).is_err());}
    #[test] fn top_p_keeps_at_least_one_token(){let mut rng=rand::rngs::StdRng::seed_from_u64(1);let id=sample(&[10.0,0.0,0.0],1.0,0,0.01,1.0,&[],&mut rng);assert_eq!(id,0);}
    #[test] fn inference_can_generate(){let tokenizer=Tokenizer::from_vocab(vec!["<pad>".into(),"<unk>".into(),"<bos>".into(),"<eos>".into(),"<cap>".into(),"<upper>".into(),"a".into()]);let model=RwkvModel::new(RwkvModelConfig::new(7,8,1,4),1);let engine=Inference{model:&model,tokenizer:&tokenizer,eos_id:tokenizer.eos_id()};let _=engine.generate("a",1,0.0,0,1.0,1.0,1);}
    #[test] fn chat_prompt_matches_format(){let tokenizer=Tokenizer::from_vocab(vec!["<pad>".into(),"<unk>".into(),"<bos>".into(),"<eos>".into(),"<cap>".into(),"<upper>".into(),"a".into()]);let model=RwkvModel::new(RwkvModelConfig::new(7,8,1,4),1);let engine=Inference{model:&model,tokenizer:&tokenizer,eos_id:3};assert_eq!(engine.chat_prompt(&[("user","hello")],Some("sys")),"System:\nsys\n\nUser:\nhello\n\nAssistant:\n");}
}


// ===== infer_server =====
const HTML:&str=r#"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>SmaulNative</title><style>body{margin:0;background:#101010;color:#eee;font:15px system-ui;display:flex;height:100vh}main{max-width:850px;width:100%;margin:auto;padding:24px;box-sizing:border-box}#chat{height:75vh;overflow:auto}.m{margin:20px 0;white-space:pre-wrap;line-height:1.6}.u{background:#222;padding:12px;border-radius:12px}.a{padding:12px}textarea{width:100%;box-sizing:border-box;background:#181818;color:#eee;border:1px solid #333;border-radius:12px;padding:12px;resize:none}button{margin-top:8px;padding:10px 18px;border:0;border-radius:9px;cursor:pointer}</style></head><body><main><h2>SmaulNative</h2><div id=chat></div><textarea id=i rows=3 placeholder='Message SmaulNative'></textarea><button onclick=send()>Send</button><script>let m=[];function draw(){chat.innerHTML=m.map(x=>`<div class=m><b>${x.role}</b><div class=${x.role==='You'?'u':'a'}>${x.text.replaceAll('&','&amp;').replaceAll('<','&lt;')}</div></div>`).join('');chat.scrollTop=chat.scrollHeight}async function send(){let x=i.value.trim();if(!x)return;i.value='';m.push({role:'You',text:x},{role:'SmaulNative',text:''});draw();let r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:[...m.slice(0,-1)].map(x=>({role:x.role==='You'?'user':'assistant',content:x.text})),stream:false,max_tokens:256,temperature:.7,top_p:.95})});let j=await r.json();m[m.length-1].text=j.choices?.[0]?.message?.content||j.error||'';draw()}i.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}})</script></main></body></html>"#;
#[derive(Deserialize)]struct Message{role:String,content:String}
#[derive(Deserialize)]struct Chat{messages:Vec<Message>,#[serde(default="dmax")]max_tokens:usize,#[serde(default="dtemp")]temperature:f32,#[serde(default="dp")]top_p:f32,#[serde(default)]top_k:usize,#[serde(default="drp")]repetition_penalty:f32,#[serde(default)]stream:bool,system:Option<String>}
fn dmax()->usize{256}fn dtemp()->f32{0.7}fn dp()->f32{0.95}fn drp()->f32{1.05}
pub struct Server{pub model:Arc<Mutex<PretrainedModel>>,pub max_prompt_tokens:usize}
impl Server{pub fn new(model_dir:impl AsRef<std::path::Path>,max_prompt_tokens:usize)->Result<Self,String>{Ok(Self{model:Arc::new(Mutex::new(PretrainedModel::load(model_dir)?)),max_prompt_tokens})}pub fn serve(&self,host:&str,port:u16)->Result<(),String>{let listener=TcpListener::bind((host,port)).map_err(|e|e.to_string())?;println!("SmaulNative listening on http://{host}:{port}");for stream in listener.incoming(){match stream{Ok(s)=>{let model=Arc::clone(&self.model);let max=self.max_prompt_tokens;std::thread::spawn(move||{if let Err(e)=handle(s,model,max){eprintln!("server: {e}")}});},Err(e)=>eprintln!("accept: {e}")}}Ok(())}}
fn response(stream:&mut TcpStream,status:&str,typ:&str,body:&str){let _=write!(stream,"HTTP/1.1 {status}\r\nContent-Type: {typ}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.as_bytes().len());}
const MAX_BODY_BYTES:usize=4*1024*1024;
fn handle(mut stream:TcpStream,model:Arc<Mutex<PretrainedModel>>,max_prompt:usize)->Result<(),String>{
    // Without a read timeout and a body-size cap, a client can send a huge (or
    // never-completing) Content-Length and keep this thread + its growing buffer
    // alive indefinitely -- an easy memory/thread-exhaustion DoS. Cap both.
    let _=stream.set_read_timeout(Some(std::time::Duration::from_secs(30)));
    let mut buf=Vec::new();let mut tmp=[0u8;8192];let header_end;loop{let n=stream.read(&mut tmp).map_err(|e|e.to_string())?;if n==0{return Ok(())}buf.extend_from_slice(&tmp[..n]);if let Some(p)=buf.windows(4).position(|x|x==b"\r\n\r\n"){header_end=p+4;break}if buf.len()>1024*1024{return Err("request headers too large".into())}}let headers=String::from_utf8_lossy(&buf[..header_end]).into_owned();let mut it=headers.lines();let first=it.next().unwrap_or("");let mut parts=first.split_whitespace();let method=parts.next().unwrap_or("");let path=parts.next().unwrap_or("");let len=it.find_map(|x|{let mut p=x.splitn(2,':');if p.next()?.eq_ignore_ascii_case("content-length"){p.next()?.trim().parse().ok()}else{None}}).unwrap_or(0);if len>MAX_BODY_BYTES{response(&mut stream,"413 Payload Too Large","application/json",&json!({"error":format!("request body exceeds {MAX_BODY_BYTES} bytes")}).to_string());return Ok(())}while buf.len()<header_end+len{let n=stream.read(&mut tmp).map_err(|e|e.to_string())?;if n==0{break}buf.extend_from_slice(&tmp[..n]);}if method=="GET"&&path=="/"{response(&mut stream,"200 OK","text/html; charset=utf-8",HTML);return Ok(())}if method=="GET"&&path=="/health"{let m=model.lock().map_err(|_|"model lock poisoned".to_string())?;let body=json!({"status":"ok","device":"cpu","parameters":format!("{:.1}M",m.model.parameter_count()as f32/1e6)}).to_string();response(&mut stream,"200 OK","application/json",&body);return Ok(())}if method=="GET"&&path=="/v1/models"{let body=json!({"object":"list","data":[{"id":"rwkv-x","object":"model","owned_by":"SmaulNative"}]}).to_string();response(&mut stream,"200 OK","application/json",&body);return Ok(())}if method!="POST"||path!="/v1/chat/completions"{response(&mut stream,"404 Not Found","application/json","{\"error\":\"not found\"}");return Ok(())}let body=&buf[header_end..header_end+len.min(buf.len()-header_end)];let req:Chat=serde_json::from_slice(body).map_err(|e|e.to_string())?;let m=model.lock().map_err(|_|"model lock poisoned".to_string())?;let engine=Inference{model:&m.model,tokenizer:&m.tokenizer,eos_id:m.tokenizer.eos_id()};let messages:Vec<(&str,&str)>=req.messages.iter().map(|x|(x.role.as_str(),x.content.as_str())).collect();let prompt=engine.chat_prompt(&messages,req.system.as_deref().or(Some("You are SmaulNative, a helpful local AI assistant.")));let count=engine.tokenizer.encode(&prompt).len();if count>max_prompt{response(&mut stream,"413 Payload Too Large","application/json",&json!({"error":format!("prompt exceeds {max_prompt} tokens ({count})")}).to_string());return Ok(())}let created=SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();let text=engine.generate(&prompt,req.max_tokens,req.temperature,req.top_k,req.top_p,req.repetition_penalty,created);let id=format!("chatcmpl-{created}");if req.stream{let mut out=String::new();out.push_str(&format!("data: {}\n\n",json!({"id":id,"object":"chat.completion.chunk","created":created,"model":"rwkv-x","choices":[{"index":0,"delta":{"content":text},"finish_reason":null}]})));out.push_str(&format!("data: {}\n\ndata: [DONE]\n\n",json!({"id":id,"object":"chat.completion.chunk","created":created,"model":"rwkv-x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]})));response(&mut stream,"200 OK","text/event-stream; charset=utf-8",&out)}else{response(&mut stream,"200 OK","application/json",&json!({"id":id,"object":"chat.completion","created":created,"model":"rwkv-x","choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}]}).to_string())}Ok(())}


// ===== gguf_inference =====

pub struct GgufInference {
    pub model: RwkvModel,
    pub tokenizer: Tokenizer,
}

impl GgufInference {
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let (model, tokenizer) = load_gguf(path)?;
        Ok(Self { model, tokenizer })
    }

    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        seed: u64,
    ) -> Result<String, String> {
        if !temperature.is_finite() || temperature < 0.0 { return Err("temperature must be non-negative".into()); }
        if !top_p.is_finite() || !(0.0 < top_p && top_p <= 1.0) { return Err("top_p must be in (0, 1]".into()); }
        if !repetition_penalty.is_finite() || repetition_penalty <= 0.0 { return Err("repetition_penalty must be positive".into()); }

        let mut tokens = self.tokenizer.encode(prompt);
        if tokens.is_empty() { tokens.push(self.tokenizer.bos_id()); }
        let mut rng = StdRng::seed_from_u64(seed);
        let (mut logits, mut state) = self.model.forward(&tokens, None);
        let mut recent = tokens.iter().copied().rev().take(128).collect::<Vec<_>>();
        recent.reverse();
        let mut generated = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            let row = logits.row(logits.nrows() - 1);
            let token = sample(row.as_slice().ok_or("model logits are not contiguous")?, temperature, top_k, top_p, repetition_penalty, &recent, &mut rng);
            if token == self.tokenizer.eos_id() { break; }
            generated.push(token);
            recent.push(token);
            if recent.len() > 128 { recent.remove(0); }
            (logits, state) = self.model.forward(&[token], Some(&state));
        }
        Ok(self.tokenizer.decode(&generated))
    }

    pub fn chat_prompt(&self, messages: &[(&str, &str)], system: Option<&str>) -> String {
        let mut out = String::new();
        if let Some(system) = system {
            out.push_str("System:\n");
            out.push_str(system);
            out.push_str("\n\n");
        }
        for &(role, content) in messages {
            let role = if role.is_empty() { "user" } else { role };
            let mut chars = role.chars();
            let label = chars.next().map(|c| c.to_uppercase().collect::<String>() + chars.as_str()).unwrap_or_default();
            out.push_str(&label);
            out.push_str(":\n");
            out.push_str(content);
            out.push_str("\n\n");
        }
        out.push_str("Assistant:\n");
        out
    }

    pub fn chat_generate(
        &self,
        messages: &[(&str, &str)],
        system: Option<&str>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        seed: u64,
    ) -> Result<String, String> {
        let prompt = self.chat_prompt(messages, system);
        self.generate(&prompt, max_new_tokens, temperature, top_k, top_p, repetition_penalty, seed)
    }
}

#[cfg(test)]
mod tests_gguf_inference {
    use super::*;

    #[test]
    fn chat_prompt_is_stable() {
        let tokenizer = Tokenizer::from_vocab(vec![
            "<pad>".into(), "<unk>".into(), "<bos>".into(), "<eos>".into(),
            "<cap>".into(), "<upper>".into(), "a".into(),
        ]);
        let model = RwkvModel::new(crate::rwkv_model::RwkvModelConfig::new(7, 8, 1, 4), 1);
        let engine = GgufInference { model, tokenizer };
        assert_eq!(engine.chat_prompt(&[("user", "hello")], Some("sys")), "System:\nsys\n\nUser:\nhello\n\nAssistant:\n");
    }
}
