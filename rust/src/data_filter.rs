use serde_json::Value;

pub fn normalize_text(text:&str)->String{text.replace("\r\n","\n").replace('\r','\n').trim().to_owned()}

pub fn filter_text(text:&str,dataset:&str,min_chars:usize,max_chars:usize)->Option<String>{
 let text=normalize_text(text);
 if text.len()<min_chars||text.len()>max_chars||text.contains('\u{fffd}') { return None; }
 let letters=text.chars().filter(|c|c.is_alphabetic()).count().max(1) as f32;
 if dataset=="hindi" { let d=text.chars().filter(|c|matches!(*c,'\u{0900}'..='\u{097f}')).count(); if d<8||d as f32/letters<0.20{return None;} }
 if dataset=="english" { let l=text.chars().filter(|c|c.is_ascii_alphabetic()).count(); if l<12||l as f32/letters<0.50{return None;} }
 Some(text)
}

pub fn filter_record(record:&Value,dataset:&str,min_chars:usize,max_chars:usize)->Option<String>{
 let object=record.as_object()?;
 let prompt=object.get("prompt").and_then(Value::as_str);
 let completion=object.get("completion").and_then(Value::as_str);
 let text=match(prompt.zip(completion)){Some((a,b))=>format!("{a}\n{b}"),None=>prompt.or(completion).or_else(||object.get("text").and_then(Value::as_str))?};
 filter_text(text,dataset,min_chars,max_chars)
}

#[cfg(test)]
mod tests{use super::*;#[test]fn filters_language(){assert!(filter_text("This is a sufficiently long English training document with many letters.","english",20,1000).is_some());assert!(filter_text("short","english",20,1000).is_none());}}
