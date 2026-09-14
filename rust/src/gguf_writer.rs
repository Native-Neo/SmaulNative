use std::fs::File;
use std::io::Write;

pub struct Tensor<'a>{pub name:&'a str,pub shape:&'a [u64],pub dtype:u32,pub data:&'a [u8]}
pub struct Writer{metadata:Vec<u8>,metadata_count:u64,tensors:Vec<Vec<u8>>,infos:Vec<(String,Vec<u64>,u32)>}
impl Writer{
 pub fn new()->Self{Self{metadata:Vec::new(),metadata_count:0,tensors:Vec::new(),infos:Vec::new()}}
 fn u32(v:&mut Vec<u8>,x:u32){v.extend(x.to_le_bytes())}
 fn u64(v:&mut Vec<u8>,x:u64){v.extend(x.to_le_bytes())}
 fn s(v:&mut Vec<u8>,x:&str){Self::u64(v,x.len() as u64);v.extend(x.as_bytes())}
 pub fn add_meta_str(&mut self,k:&str,v:&str){Self::s(&mut self.metadata,k);Self::u32(&mut self.metadata,8);Self::s(&mut self.metadata,v);self.metadata_count+=1}
 pub fn add_meta_u32(&mut self,k:&str,v:u32){Self::s(&mut self.metadata,k);Self::u32(&mut self.metadata,4);Self::u32(&mut self.metadata,v);self.metadata_count+=1}
 pub fn add_meta_bool(&mut self,k:&str,v:bool){Self::s(&mut self.metadata,k);Self::u32(&mut self.metadata,7);self.metadata.push(v as u8);self.metadata_count+=1}
 pub fn add_tokens(&mut self,k:&str,tokens:&[String]){Self::s(&mut self.metadata,k);Self::u32(&mut self.metadata,9);Self::u32(&mut self.metadata,8);Self::u64(&mut self.metadata,tokens.len() as u64);for t in tokens{Self::s(&mut self.metadata,t)}self.metadata_count+=1}
 pub fn tensor(&mut self,t:Tensor<'_>){self.tensors.push(t.data.to_vec());self.infos.push((t.name.into(),t.shape.into(),t.dtype));}
 pub fn finish(self,path:&str)->Result<(),String>{let mut head=Vec::new();head.extend(b"GGUF");Self::u32(&mut head,3);Self::u64(&mut head,self.tensors.len() as u64);Self::u64(&mut head,self.metadata_count);head.extend(self.metadata);let mut offs=0u64;for(i,(n,s,d))in self.infos.iter().enumerate(){offs=(offs+31)&!31;Self::s(&mut head,n);Self::u32(&mut head,s.len() as u32);for x in s{Self::u64(&mut head,*x)}Self::u32(&mut head,*d);Self::u64(&mut head,offs);offs+=self.tensors[i].len() as u64;}
 let data_start=(head.len()+31)&!31;let mut f=File::create(path).map_err(|e|e.to_string())?;f.write_all(&head).map_err(|e|e.to_string())?;if data_start>head.len(){f.write_all(&vec![0;data_start-head.len()]).map_err(|e|e.to_string())?}let mut pos=data_start as u64;for d in self.tensors{let a=(pos+31)&!31;if a>pos{f.write_all(&vec![0;(a-pos)as usize]).map_err(|e|e.to_string())?}f.write_all(&d).map_err(|e|e.to_string())?;pos=a+d.len() as u64;}Ok(())}
}
