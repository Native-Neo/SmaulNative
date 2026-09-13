#[derive(Clone,Debug)]
pub struct SequenceCursor{start:usize,length:usize}
impl SequenceCursor{
 pub fn new(start:usize,length:usize)->Self{Self{start,length}}
 pub fn start(&self)->usize{self.start}
 pub fn end(&self)->usize{self.start.saturating_add(self.length)}
 pub fn length(&self)->usize{self.length}
 pub fn advance(&mut self,count:usize){self.start=self.end();self.length=count;}
}
#[cfg(test)]
mod tests{use super::*;#[test]fn tracks_range(){let mut c=SequenceCursor::new(4,8);assert_eq!(c.end(),12);c.advance(3);assert_eq!(c.start(),12);assert_eq!(c.end(),15);}}
