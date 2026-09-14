#[derive(Clone,Debug)]
pub struct TokenWindow{capacity:usize,tokens:Vec<usize>}
impl TokenWindow{pub fn new(capacity:usize)->Self{assert!(capacity>0);Self{capacity,tokens:Vec::with_capacity(capacity)}}pub fn push(&mut self,token:usize){if self.tokens.len()==self.capacity{self.tokens.remove(0);}self.tokens.push(token)}pub fn as_slice(&self)->&[usize]{&self.tokens}pub fn len(&self)->usize{self.tokens.len()}}
#[cfg(test)]
mod tests{use super::*;#[test]fn keeps_latest(){let mut w=TokenWindow::new(2);w.push(1);w.push(2);w.push(3);assert_eq!(w.as_slice(),&[2,3]);}}
