# RQT

SmaulNative RQT (Real Quantized Training) trains directly from physically quantized weights. It is separate from QAT: there is no FP32 master weight for an RQT linear layer.

## Modes

```bash
# Pure FP4
python train.py --cpu --mode pretrain --dataset_dir ./datasets --rqt --rqt_bits 4

# Pure FP6
python train.py --cpu --mode pretrain --dataset_dir ./datasets --rqt --rqt_bits 6

# Pure FP8
python train.py --cpu --mode pretrain --dataset_dir ./datasets --rqt --rqt_bits 8

# Mixed FP4/FP6/FP8
python train.py --cpu --mode pretrain --dataset_dir ./datasets --mixed_rqt
```

`--rqt` selects one precision for every `nn.Linear`. `--mixed_rqt` uses FP8 for RWKV TimeMix/MOBA attention projections and the model head, FP4 for Channel-Mix/FFN projections, and FP6 for the remaining linear layers.

## How it differs from QAT

QAT keeps an ordinary floating-point parameter and fake-quantizes it during the forward pass. RQT replaces that parameter with packed low-bit storage. The forward pass dequantizes the packed weight for the matrix operation, gradients are captured from that actual quantized forward weight, and the optimizer immediately requantizes the updated weight after every step.

RQT therefore does not maintain a hidden FP32 master copy of an RQT linear weight. Lion's optimizer averages remain FP32 because optimizer state needs more numerical resolution than the stored model weight.

## Storage

- FP4 packs two 4-bit codes into each byte.
- FP6 packs four 6-bit codes into three bytes.
- FP8 uses PyTorch `float8_e4m3fn` storage.
- FP4 and FP6 use custom E2M1 and E3M2-style finite floating-point codebooks with a per-output-row scale.
- RQT checkpoints store the packed weights, scales, and per-layer bit-width metadata.

The CPU implementation currently decodes weights to FP32 for the matrix multiplication. Physical packing reduces stored weight memory, but it is not yet a native FP4/FP6 GEMM kernel.

## Resume

RQT uses `RQTLion`, which stores optimizer averages separately for packed linear layers and normal floating-point parameters. State is keyed by stable module/parameter names and validates tensor shapes when loaded.

## Limitations

RQT is experimental. Extremely low precision can make optimization unstable, especially FP4. Gradient clipping is applied before the Lion update. FP8 requires a PyTorch build/device that supports `torch.float8_e4m3fn` conversion.
