# train.py

`train.py` is the main training entry point for pretraining and SFT.

## Pretraining

```bash
python train.py --cpu --mode pretrain \
    --dataset_dir ./datasets \
    --output_dir ./SmaulNative \
    --ctx_len 512
```

The default new-model configuration is:

- hidden size: `832`
- layers: `17`
- head size: `64`
- MOBA layers: `3`
- batch size: `1`
- learning rate: `1e-4`
- optimizer: Lion

`--cpu` enables the native CPU WKV backend and CPU thread configuration.

## SFT

```bash
python train.py --cpu --mode sft \
    --dataset_dir ./sft_data \
    --output_dir ./SmaulNative-SFT \
    --tokenizer_path ./SmaulNative/tokenizer.json
```

SFT uses `SFTDataset` and masks every non-assistant token from the loss.

## RQT

RQT is enabled separately from ordinary floating-point training:

```bash
python train.py --cpu --mode sft --dataset_dir ./sft_data \
    --output_dir ./SmaulNative-RQT --rqt --rqt_bits 6
```

Use `--rqt_bits 4`, `6`, or `8` for pure FP4, FP6, or FP8 RQT. Use `--mixed_rqt` for the mixed FP4/FP6/FP8 layout. RQT uses `RQTLion`; ordinary training uses the standard Lion optimizer.

Unlike QAT, RQT does not retain an FP32 master weight for RQT linear layers. The packed weight is used for the forward pass and requantized after every optimizer step.

See [rqt.md](rqt.md) for storage, checkpoint, and precision details.

## Streaming

`--stream_dataset` accepts `none`, `hindi`, `english`, `openthoughts`, or `all`. Streaming is available for pretraining only.

## Resume

The checkpoint directory contains model state, optimizer state when scheduled, RNG state, and `resume_state.json`. Resume state includes the dataset position and the partially consumed token buffer. RQT optimizer state is keyed by stable module/parameter names and validates tensor shapes on restore.

## Training controls

Useful flags include:

- `--batch_size`
- `--epochs` (SFT)
- `--lr`
- `--weight_decay`
- `--save_every`
- `--optimizer_save_every`
- `--precision fp32|fp16|bf16`
- `--save_dtype fp32|fp16|bf16`
- `--rqt`
- `--rqt_bits 4|6|8`
- `--mixed_rqt`
- `--router_only`
- `--compile`
- `--cpu`

Final partial pretraining batches are flushed at end-of-stream instead of being silently discarded.
