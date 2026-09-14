use ndarray::Array2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bits { Fp2, Fp4 }
impl Bits { pub fn from_bits(bits:u8)->Result<Self,String>{match bits{2=>Ok(Self::Fp2),4=>Ok(Self::Fp4),_=>(Err("bits must be 2 or 4".into()))}} fn levels(self)->&'static[f32]{match self{Self::Fp2=>&[-1.0,0.0,1.0],Self::Fp4=>&[-2.0,-1.0,-0.5,-0.25,0.0,0.25,0.5,1.0,2.0]}} }

pub fn quantize_codes(x:&Array2<f32>,bits:Bits)->(Array2<u8>,Vec<f32>){let levels=bits.levels();let mut codes=Array2::zeros(x.raw_dim());let mut scales=vec![0.0;x.ncols()];for col in 0..x.ncols(){let mut scale=x.column(col).iter().map(|v|v.abs()).fold(0.0,f32::max);if scale<f32::EPSILON{scale=1.0}scales[col]=scale;for row in 0..x.nrows(){let value=x[[row,col]]/scale;let(mut best,mut distance)=(0,f32::INFINITY);for(i,level)in levels.iter().enumerate(){let d=(value-level).abs();if d<distance{distance=d;best=i}}codes[[row,col]]=best as u8;}}(codes,scales)}
pub fn dequantize(codes:&Array2<u8>,scales:&[f32],bits:Bits)->Array2<f32>{assert_eq!(codes.ncols(),scales.len());let levels=bits.levels();Array2::from_shape_fn(codes.raw_dim(),|(row,col)|levels[codes[[row,col]]as usize]*scales[col])}
pub fn fake_quantize(x:&Array2<f32>,bits:Bits)->Array2<f32>{let(codes,scales)=quantize_codes(x,bits);dequantize(&codes,&scales,bits)}
pub fn quantized_linear(input:&Array2<f32>,weight:&Array2<f32>,bits:Bits)->Array2<f32>{assert_eq!(input.ncols(),weight.ncols());input.dot(&fake_quantize(weight,bits).t())}

pub const QAT_3BIT_MIN:i8=-4;
pub const QAT_3BIT_MAX:i8=3;

pub fn pack_3bit(codes:&[u8])->Vec<u8>{let mut out=Vec::with_capacity((codes.len()*3+7)/8);let mut acc=0u8;let mut bits=0u8;for&code in codes{let code=code&7;for shift in[2u8,1,0]{acc|=((code>>shift)&1)<<(7-bits);bits+=1;if bits==8{out.push(acc);acc=0;bits=0;}}}if bits>0{out.push(acc)}out}
pub fn unpack_3bit(packed:&[u8],numel:usize)->Vec<u8>{let mut out=Vec::with_capacity(numel);for i in 0..numel*3{let byte=packed[i/8];let bit=(byte>>(7-(i%8)))&1;if i%3==0{out.push(bit<<2)}else if i%3==1{*out.last_mut().unwrap()|=bit<<1}else{*out.last_mut().unwrap()|=bit}}out}

pub fn quantize_weight_3bit(weight:&Array2<f32>)->(Vec<u8>,Vec<f32>){let mut codes=Vec::with_capacity(weight.len());let mut scales=Vec::with_capacity(weight.nrows());for row in weight.rows(){let mut scale=row.iter().map(|v|v.abs()).fold(0.0,f32::max);if scale<f32::EPSILON{scale=1.0}scales.push(scale);for&v in row{let q=(v/scale).round().clamp(QAT_3BIT_MIN as f32,QAT_3BIT_MAX as f32)as i8;codes.push((q-QAT_3BIT_MIN)as u8);}}(pack_3bit(&codes),scales)}
pub fn dequantize_weight_3bit(packed:&[u8],scales:&[f32],shape:(usize,usize))->Array2<f32>{let(num_rows,num_cols)=shape;assert_eq!(scales.len(),num_rows);let codes=unpack_3bit(packed,num_rows*num_cols);Array2::from_shape_fn(shape,|(r,c)|((codes[r*num_cols+c]as i8+QAT_3BIT_MIN)as f32)*scales[r])}
pub fn quantized_linear_3bit(input:&Array2<f32>,weight:&Array2<f32>)->Array2<f32>{let(packed,scales)=quantize_weight_3bit(weight);input.dot(&dequantize_weight_3bit(&packed,&scales,weight.dim()).t())}

#[cfg(test)]
mod tests{use super::*;use ndarray::array;
#[test]fn fp2_round_trip_uses_three_levels(){let x=array![[-2.0,0.0],[0.5,2.0]];let q=fake_quantize(&x,Bits::Fp2);assert_eq!(q[[0,0]],-2.0);assert_eq!(q[[0,1]],0.0);assert_eq!(q[[1,0]],1.0);assert_eq!(q[[1,1]],2.0)}
#[test]fn quantized_linear_has_expected_shape(){let x=array![[1.0,2.0,3.0]];let w=array![[1.0,0.0,0.0],[0.0,1.0,0.0]];assert_eq!(quantized_linear(&x,&w,Bits::Fp4).shape(),&[1,2])}
#[test]fn pack_3bit_round_trips(){let codes=vec![0,1,2,3,4,5,6,7,3,1,6];assert_eq!(unpack_3bit(&pack_3bit(&codes),codes.len()),codes)}
#[test]fn qat_3bit_weight_preserves_shape(){let w=array![[0.2,-0.4,0.7],[1.0,-0.5,0.25]];let(p,s)=quantize_weight_3bit(&w);let q=dequantize_weight_3bit(&p,&s,w.dim());assert_eq!(q.dim(),w.dim());assert!(q.iter().all(|v|v.is_finite()))}
}