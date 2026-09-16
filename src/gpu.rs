use crate::qat::{unpack_lowbit,Bits};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBackend { Cpu, Cuda, Hip }
pub fn backend() -> GpuBackend { if cfg!(feature="cuda"){GpuBackend::Cuda}else if cfg!(feature="hip"){GpuBackend::Hip}else{GpuBackend::Cpu} }
pub fn packed_row_bytes(in_features:usize,bits:u8)->Result<usize,String>{let b=Bits::from_bits(bits)?;if b==Bits::Fp8{return Err("low-bit backend supports 2-, 3- and 4-bit weights".into())}Ok(b.row_bytes(in_features))}
fn decode(code:u8,b:Bits)->f32{b.levels().get(code as usize).copied().unwrap_or(0.0)}
pub fn decode_fp2(code:u8)->f32{decode(code&3,Bits::Fp2)}
pub fn decode_fp3(code:u8)->f32{decode(code&7,Bits::Fp3)}
pub fn decode_fp4(code:u8)->f32{decode(code&15,Bits::Fp4)}

pub fn packed_linear_cpu(x:&[f32],packed:&[u8],scale:&[f32],batch:usize,out_features:usize,in_features:usize,bits:u8)->Result<Vec<f32>,String>{
 let b=Bits::from_bits(bits)?;if b==Bits::Fp8{return Err("use packed_linear_fp8_cpu for FP8 weights".into())}
 if x.len()!=batch*in_features||scale.len()!=out_features{return Err("low-bit linear shape mismatch".into())}
 let rb=b.row_bytes(in_features);if packed.len()<out_features*rb{return Err("packed weight is too small".into())}
 let levels=b.levels();let mut out=vec![0.0;batch*out_features];let mut row=vec![0.0f32;in_features];
 for o in 0..out_features{
  // decode the output row once instead of once per batch element
  for (k,c) in unpack_lowbit(&packed[o*rb..(o+1)*rb],in_features,b).into_iter().enumerate(){row[k]=levels[c as usize]*scale[o];}
  for n in 0..batch{let xr=&x[n*in_features..(n+1)*in_features];let mut sum=0.0;for k in 0..in_features{sum+=xr[k]*row[k];}out[n*out_features+o]=sum}
 }
 Ok(out)}

pub fn fp8_e4m3(code:u8)->f32{crate::qat::fp8_value(code)}
pub fn packed_linear_fp8_cpu(x:&[f32],packed:&[u8],batch:usize,out_features:usize,in_features:usize)->Result<Vec<f32>,String>{if x.len()!=batch*in_features||packed.len()<out_features*in_features{return Err("FP8 linear shape mismatch".into())}let mut out=vec![0.0;batch*out_features];for n in 0..batch{for o in 0..out_features{let mut sum=0.0;for k in 0..in_features{sum+=x[n*in_features+k]*fp8_e4m3(packed[o*in_features+k]);}out[n*out_features+o]=sum}}Ok(out)}

#[cfg(feature="cuda")] extern "C"{fn smaul_cuda_packed_linear(x:*const f32,w:*const u8,s:*const f32,y:*mut f32,batch:i32,out_features:i32,in_features:i32,bits:i32)->i32;}
#[cfg(feature="hip")] extern "C"{fn smaul_hip_packed_linear(x:*const f32,w:*const u8,s:*const f32,y:*mut f32,batch:i32,out_features:i32,in_features:i32,bits:i32)->i32;}
#[cfg(feature="cuda")] pub fn packed_linear_cuda(x:&[f32],w:&[u8],s:&[f32],batch:usize,out_features:usize,in_features:usize,bits:u8)->Result<Vec<f32>,String>{if bits!=2&&bits!=4{return Err("CUDA packed kernel supports FP2/FP4".into())}let mut y=vec![0.0;batch*out_features];let e=unsafe{smaul_cuda_packed_linear(x.as_ptr(),w.as_ptr(),s.as_ptr(),y.as_mut_ptr(),batch as i32,out_features as i32,in_features as i32,bits as i32)};if e!=0{Err(format!("CUDA low-bit kernel failed: {e}"))}else{Ok(y)}}
#[cfg(feature="hip")] pub fn packed_linear_hip(x:&[f32],w:&[u8],s:&[f32],batch:usize,out_features:usize,in_features:usize,bits:u8)->Result<Vec<f32>,String>{if bits!=2&&bits!=4{return Err("HIP packed kernel supports FP2/FP4".into())}let mut y=vec![0.0;batch*out_features];let e=unsafe{smaul_hip_packed_linear(x.as_ptr(),w.as_ptr(),s.as_ptr(),y.as_mut_ptr(),batch as i32,out_features as i32,in_features as i32,bits as i32)};if e!=0{Err(format!("HIP low-bit kernel failed: {e}"))}else{Ok(y)}}

pub fn packed_linear(x:&[f32],packed:&[u8],scale:&[f32],batch:usize,out_features:usize,in_features:usize,bits:u8)->Result<Vec<f32>,String>{match bits{2|3|4=>{
#[cfg(feature="cuda")]
{match packed_linear_cuda(x,packed,scale,batch,out_features,in_features,bits){Ok(y)=>return Ok(y),Err(_)=>{}}}
#[cfg(feature="hip")]
{match packed_linear_hip(x,packed,scale,batch,out_features,in_features,bits){Ok(y)=>return Ok(y),Err(_)=>{}}}
packed_linear_cpu(x,packed,scale,batch,out_features,in_features,bits)},8=>packed_linear_fp8_cpu(x,packed,batch,out_features,in_features),_=>Err("bits must be 2, 3, 4, or 8".into())}}

#[cfg(test)]mod tests{use super::*;use crate::qat::{dequantize_weight,fake_quantize,quantize_weight};use ndarray::array;
#[test]fn cpu_fp4_matches_shape(){let y=packed_linear_cpu(&[1.0,2.0],&[0x18],&[1.0],1,1,2,4).unwrap();assert_eq!(y.len(),1);assert!(y[0].is_finite())}
#[test]fn fp8_packed_is_direct(){let y=packed_linear_fp8_cpu(&[1.0,2.0],&[0,0],1,1,2).unwrap();assert_eq!(y,vec![0.0]);assert!(fp8_e4m3(0x7f).is_nan());assert!(fp8_e4m3(0xff).is_nan());assert!((fp8_e4m3(0x01)-2.0_f32.powi(-9)).abs()<1e-12);assert!((fp8_e4m3(0x09)-2.0_f32.powi(-6)*(1.0+1.0/8.0)).abs()<1e-7)}
#[test]fn packed_fp2_and_fp4_match_qat_dequantization(){let x=array![[1.0,-2.0,0.25],[0.5,1.5,-1.0]];for bits in [Bits::Fp2,Bits::Fp3,Bits::Fp4]{let(packed,scales)=quantize_weight(&x,bits);let expected=dequantize_weight(&packed,&scales,x.dim(),bits);let y=packed_linear_cpu(&[1.0,2.0,3.0,4.0,5.0,6.0],&packed,&scales,2,2,3,bits.bits()).unwrap();let input=array![[1.0,2.0,3.0],[4.0,5.0,6.0]];let reference=input.dot(&expected.t());for(a,b)in y.iter().zip(reference.iter()){assert!((a-b).abs()<1e-6)}}}
#[test]fn all_qat_modes_have_consistent_weight_shapes(){let x=array![[1.0,-2.0,0.25],[0.5,1.5,-1.0]];for bits in [Bits::Fp2,Bits::Fp3,Bits::Fp4,Bits::Fp8]{let q=fake_quantize(&x,bits);assert_eq!(q.dim(),x.dim());}}
}
