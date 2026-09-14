#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct MicroBatch{pub start:usize,pub end:usize}
impl MicroBatch{pub fn len(&self)->usize{self.end.saturating_sub(self.start)}pub fn new(start:usize,end:usize)->Self{assert!(end>=start);Self{start,end}}}
#[cfg(test)]
mod tests{use super::*;#[test]fn measures(){let b=MicroBatch::new(4,12);assert_eq!(b.len(),8);}}
