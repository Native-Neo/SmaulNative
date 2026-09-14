use ndarray::Array2;

pub const QAT_3BIT_MIN: i8 = -4;
pub const QAT_3BIT_MAX: i8 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bits { Fp2, Fp4, Fp8 }

impl Bits {
    pub fn from_bits(bits: u8) -> Result<Self, String> { match bits { 2 => Ok(Self::Fp2), 4 => Ok(Self::Fp4), 8 => Ok(Self::Fp8), _ => Err("bits must be 2, 4, or 8".into()) } }
    pub fn bits(self) -> u8 { match self { Self::Fp2 => 2, Self::Fp4 => 4, Self::Fp8 => 8 } }
    fn levels(self) -> &'static [f32] { match self { Self::Fp2 => &[-1.0, 0.0, 1.0], Self::Fp4 => &[-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0], Self::Fp8 => &[] } }
}

fn fp8_e4m3(value: f32) -> f32 {
    if !value.is_finite() || value == 0.0 { return value; }
    let sign = value.is_sign_negative();
    let x = value.abs();
    let exponent = x.log2().floor().clamp(-6.0, 7.0);
    let base = 2.0_f32.powf(exponent);
    let mantissa = ((x / base - 1.0) * 8.0).round().clamp(0.0, 7.0);
    let quantized = base * (1.0 + mantissa / 8.0);
    if sign { -quantized } else { quantized }
}

pub fn quantize_codes(x: &Array2<f32>, bits: Bits) -> (Array2<u8>, Vec<f32>) {
    assert_ne!(bits, Bits::Fp8);
    let levels = bits.levels(); let mut codes = Array2::zeros(x.raw_dim()); let mut scales = vec![0.0; x.ncols()];
    for col in 0..x.ncols() { let mut scale=x.column(col).iter().map(|v|v.abs()).fold(0.0,f32::max); if scale<f32::EPSILON {scale=1.0;} scales[col]=scale; for row in 0..x.nrows(){let value=x[[row,col]]/scale;let mut best=0;let mut distance=f32::INFINITY;for(index,level)in levels.iter().enumerate(){let current=(value-level).abs();if current<distance{distance=current;best=index;}}codes[[row,col]]=best as u8;}}
    (codes,scales)
}

pub fn dequantize(codes:&Array2<u8>,scales:&[f32],bits:Bits)->Array2<f32>{assert_ne!(bits,Bits::Fp8);assert_eq!(codes.ncols(),scales.len());let levels=bits.levels();Array2::from_shape_fn(codes.raw_dim(),|(row,col)|levels[codes[[row,col]]as usize]*scales[col])}
pub fn fake_quantize(x:&Array2<f32>,bits:Bits)->Array2<f32>{match bits{Bits::Fp8=>x.mapv(fp8_e4m3),Bits::Fp2|Bits::Fp4=>{let(codes,scales)=quantize_codes(x,bits);dequantize(&codes,&scales,bits)}}}
pub fn quantized_linear(input:&Array2<f32>,weight:&Array2<f32>,bits:Bits)->Array2<f32>{assert_eq!(input.ncols(),weight.ncols());input.dot(&fake_quantize(weight,bits).t())}

pub fn pack_3bit(codes:&[u8])->Vec<u8>{let mut output=Vec::with_capacity((codes.len()*3+7)/8);let mut accumulator=0u8;let mut bit_count=0u8;for&code in codes{let code=code&7;for shift in[2u8,1,0]{accumulator|=((code>>shift)&1)<<(7-bit_count);bit_count+=1;if bit_count==8{output.push(accumulator);accumulator=0;bit_count=0;}}}if bit_count!=0{output.push(accumulator);}output}
pub fn unpack_3bit(packed:&[u8],numel:usize)->Vec<u8>{assert!(packed.len()*8>=numel*3);let mut output=Vec::with_capacity(numel);for i in 0..numel*3{let byte=packed[i/8];let bit=(byte>>(7-(i%8)))&1;match i%3{0=>output.push(bit<<2),1=>*output.last_mut().unwrap()|=bit<<1,_=>*output.last_mut().unwrap()|=bit}}output}

pub fn pack_lowbit(codes:&[u8],bits:Bits)->Vec<u8>{match bits{Bits::Fp2=>{let mut out=Vec::with_capacity((codes.len()*2+7)/8);let mut acc=0u8;let mut n=0;for&c in codes{acc|=(c&3)<<(6-n);n+=2;if n==8{out.push(acc);acc=0;n=0;}}if n!=0{out.push(acc);}out},Bits::Fp4=>{let mut out=Vec::with_capacity((codes.len()+1)/2);for chunk in codes.chunks(2){out.push((chunk[0]&15)<<4|chunk.get(1).copied().unwrap_or(0)&15);}out},Bits::Fp8=>codes.to_vec()}}
pub fn unpack_lowbit(packed:&[u8],numel:usize,bits:Bits)->Vec<u8>{match bits{Bits::Fp2=>{assert!(packed.len()*4>=numel);(0..numel).map(|i|(packed[i/4]>>(6-2*(i%4)))&3).collect()},Bits::Fp4=>{assert!(packed.len()*2>=numel);(0..numel).map(|i|if i%2==0{(packed[i/2]>>4)&15}else{packed[i/2]&15}).collect()},Bits::Fp8=>packed[..numel].to_vec()}}

pub fn quantize_weight(weight:&Array2<f32>,bits:Bits)->(Vec<u8>,Vec<f32>){match bits{Bits::Fp8=>{let mut bytes=Vec::with_capacity(weight.len()*4);for&v in weight.iter(){bytes.extend_from_slice(&fp8_e4m3(v).to_le_bytes());}(bytes,vec![1.0])},Bits::Fp2|Bits::Fp4=>{let(codes,scales)=quantize_codes(weight,bits);(pack_lowbit(codes.as_slice().unwrap(),bits),scales)}}}
pub fn dequantize_weight(packed:&[u8],scales:&[f32],shape:(usize,usize),bits:Bits)->Array2<f32>{match bits{Bits::Fp8=>{assert!(packed.len()>=shape.0*shape.1*4);let values=(0..shape.0*shape.1).map(|i|f32::from_le_bytes([packed[i*4],packed[i*4+1],packed[i*4+2],packed[i*4+3]])).collect();Array2::from_shape_vec(shape,values).unwrap()},Bits::Fp2|Bits::Fp4=>{let codes=unpack_lowbit(packed,shape.0*shape.1,bits);let levels=bits.levels();Array2::from_shape_fn(shape,|(r,c)|levels[codes[r*shape.1+c]as usize]*scales[c])}}}

pub fn quantize_weight_3bit(weight:&Array2<f32>)->(Vec<u8>,Vec<f32>){let mut codes=Vec::with_capacity(weight.len());let mut scales=Vec::with_capacity(weight.nrows());for row in weight.rows(){let max_abs=row.iter().map(|v|v.abs()).fold(0.0,f32::max);let scale=(max_abs/QAT_3BIT_MAX as f32).max(f32::EPSILON);scales.push(scale);for&value in row{let q=(value/scale).round().clamp(QAT_3BIT_MIN as f32,QAT_3BIT_MAX as f32)as i8;codes.push((q-QAT_3BIT_MIN)as u8);}}(pack_3bit(&codes),scales)}
pub fn dequantize_weight_3bit(packed:&[u8],scales:&[f32],shape:(usize,usize))->Array2<f32>{let(rows,cols)=shape;assert_eq!(scales.len(),rows);let codes=unpack_3bit(packed,rows*cols);Array2::from_shape_fn(shape,|(row,col)|((codes[row*cols+col]as i8+QAT_3BIT_MIN)as f32)*scales[row])}

#[derive(Clone,Debug)]pub struct QatLinear{pub weight:Array2<f32>,pub bits:Bits}
impl QatLinear{pub fn new(weight:Array2<f32>,bits:Bits)->Self{assert!(!weight.is_empty());Self{weight,bits}}pub fn forward(&self,input:&Array2<f32>)->Array2<f32>{input.dot(&fake_quantize(&self.weight,self.bits).t())}pub fn convert(&self)->PackedLinear{let(packed,scales)=quantize_weight(&self.weight,self.bits);PackedLinear{packed,scales,shape:self.weight.dim(),bits:self.bits}}}
#[derive(Clone,Debug)]pub struct PackedLinear{pub packed:Vec<u8>,pub scales:Vec<f32>,pub shape:(usize,usize),pub bits:Bits}
impl PackedLinear{pub fn forward(&self,input:&Array2<f32>)->Array2<f32>{input.dot(&dequantize_weight(&self.packed,&self.scales,self.shape,self.bits).t())}}

#[derive(Clone,Debug)]pub struct QatLinear3Bit{pub weight:Array2<f32>,pub signed_activation:bool,pub activation_min:f32,pub activation_max:f32,pub momentum:f32}
impl QatLinear3Bit{pub fn new(weight:Array2<f32>,signed_activation:bool)->Self{assert!(weight.nrows()>0&&weight.ncols()>0);Self{weight,signed_activation,activation_min:0.0,activation_max:0.0,momentum:0.1}}pub fn observe(&mut self,input:&Array2<f32>){if input.is_empty(){return;}let min=input.iter().copied().fold(f32::INFINITY,f32::min);let max=input.iter().copied().fold(f32::NEG_INFINITY,f32::max);if self.activation_min==0.0&&self.activation_max==0.0{self.activation_min=min;self.activation_max=max}else{self.activation_min=(1.0-self.momentum)*self.activation_min+self.momentum*min;self.activation_max=(1.0-self.momentum)*self.activation_max+self.momentum*max}}pub fn fake_quantize_activation(&self,input:&Array2<f32>)->Array2<f32>{if self.signed_activation{let max_abs=self.activation_min.abs().max(self.activation_max.abs());let scale=(max_abs/QAT_3BIT_MAX as f32).max(f32::EPSILON);return Array2::from_shape_fn(input.raw_dim(),|i|(input[i]/scale).round().clamp(QAT_3BIT_MIN as f32,QAT_3BIT_MAX as f32)*scale)}let range=(self.activation_max-self.activation_min).max(f32::EPSILON);Array2::from_shape_fn(input.raw_dim(),|i|{let normalized=((input[i]-self.activation_min)/range).clamp(0.0,1.0);let q=(normalized*7.0).round();self.activation_min+q*range/7.0})}pub fn forward(&mut self,input:&Array2<f32>)->Array2<f32>{self.observe(input);let activation=self.fake_quantize_activation(input);let(packed,scales)=quantize_weight_3bit(&self.weight);activation.dot(&dequantize_weight_3bit(&packed,&scales,self.weight.dim()).t())}pub fn convert(&self)->QuantizedLinear3Bit{let(packed,scales)=quantize_weight_3bit(&self.weight);QuantizedLinear3Bit{packed,scales,shape:self.weight.dim()}}}
#[derive(Clone,Debug)]pub struct QuantizedLinear3Bit{pub packed:Vec<u8>,pub scales:Vec<f32>,pub shape:(usize,usize)}
impl QuantizedLinear3Bit{pub fn forward(&self,input:&Array2<f32>)->Array2<f32>{input.dot(&dequantize_weight_3bit(&self.packed,&self.scales,self.shape).t())}}

#[cfg(test)]mod tests{use super::*;use ndarray::array;#[test]fn bit_modes_are_supported(){assert_eq!(Bits::from_bits(2).unwrap().bits(),2);assert_eq!(Bits::from_bits(4).unwrap().bits(),4);assert_eq!(Bits::from_bits(8).unwrap().bits(),8);assert!(Bits::from_bits(3).is_err());}#[test]fn fp2_round_trip_uses_three_levels(){let x=array![[-2.0,0.0],[0.5,2.0]];let q=fake_quantize(&x,Bits::Fp2);assert_eq!(q[[0,0]],-2.0);assert_eq!(q[[0,1]],0.0);assert_eq!(q[[1,0]],1.0);assert_eq!(q[[1,1]],2.0);}#[test]fn fp4_quantization_has_expected_shape(){let x=array![[1.0,-0.5,0.2]];assert_eq!(fake_quantize(&x,Bits::Fp4).dim(),x.dim());}#[test]fn fp8_quantization_is_finite(){let x=array![[1.0,-0.5,0.2]];let q=fake_quantize(&x,Bits::Fp8);assert!(q.iter().all(|v|v.is_finite()));assert!((q[[0,0]]-1.0).abs()<0.01);}#[test]fn quantized_linear_has_expected_shape(){let x=array![[1.0,2.0,3.0]];let w=array![[1.0,0.0,0.0],[0.0,1.0,0.0]];assert_eq!(quantized_linear(&x,&w,Bits::Fp4).shape(),&[1,2]);}#[test]fn packed_lowbit_round_trips(){let codes=vec![0,1,2,3,0,1,2,3,2,1,0];for bits in[Bits::Fp2,Bits::Fp4]{let packed=pack_lowbit(&codes,bits);let expected=codes.iter().map(|c|if bits==Bits::Fp2{c&3}else{c&15}).collect::<Vec<_>>();assert_eq!(unpack_lowbit(&packed,codes.len(),bits),expected);}}#[test]fn pack_3bit_round_trips(){let codes=vec![0,1,2,3,4,5,6,7,3,1,6];assert_eq!(unpack_3bit(&pack_3bit(&codes),codes.len()),codes);}#[test]fn qat_3bit_weight_preserves_shape(){let weight=array![[0.2,-0.4,0.7],[1.0,-0.5,0.25]];let(packed,scales)=quantize_weight_3bit(&weight);let quantized=dequantize_weight_3bit(&packed,&scales,weight.dim());assert_eq!(quantized.dim(),weight.dim());assert!(quantized.iter().all(|value|value.is_finite()));assert!(quantized[[0,2]]>0.6);}#[test]fn qat_linear_observes_and_converts(){let weight=array![[1.0,0.0],[0.0,1.0]];let mut qat=QatLinear3Bit::new(weight,true);let input=array![[1.0,-0.5]];let output=qat.forward(&input);assert_eq!(output.dim(),(1,2));assert!(qat.activation_min<=-0.5);assert!(qat.activation_max>=1.0);assert_eq!(qat.convert().shape,(2,2));}#[test]fn fp4_packed_linear_round_trip(){let weight=array![[1.0,-0.5],[0.25,2.0]];let qat=QatLinear::new(weight,Bits::Fp4);let packed=qat.convert();let x=array![[1.0,2.0]];assert_eq!(packed.forward(&x).dim(),(1,2));}#[test]fn fp8_packed_linear_round_trip(){let weight=array![[1.0,-0.5],[0.25,2.0]];let qat=QatLinear::new(weight,Bits::Fp8);let packed=qat.convert();let x=array![[1.0,2.0]];let y=packed.forward(&x);assert_eq!(y.dim(),(1,2));assert!(y.iter().all(|v|v.is_finite()));}}
