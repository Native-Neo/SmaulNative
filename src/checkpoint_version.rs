pub const CHECKPOINT_FORMAT_VERSION:u32=2;
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct CheckpointVersion(pub u32);
impl CheckpointVersion{pub fn current()->Self{Self(CHECKPOINT_FORMAT_VERSION)}pub fn is_supported(self)->bool{self.0==CHECKPOINT_FORMAT_VERSION}}
#[cfg(test)]
mod tests{use super::*;#[test]fn current_is_supported(){assert!(CheckpointVersion::current().is_supported());assert_eq!(CheckpointVersion::current().0,2);}}
