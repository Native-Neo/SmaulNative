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
- context window used for training: `1024`
- batch size: `2`
- optimizer: `adafactor`
- learning rate: `1e-4`

`--cpu` enables the native CPU WKV backend and CPU thread configuration.

## SFT

```bash
python train.py --cpu --mode sft \
    --dataset_dir ./sft_data \
    --output_dir ./SmaulNative-SFT \
    --tokenizer_path ./SmaulNative/tokenizer.json
```

SFT uses `SFTDataset` and masks every non-assistant token from the loss.

## Streaming

`--stream_dataset` accepts `none`, `hindi`, `english`, `openthoughts`, or `all`. Streaming is available
for pretraining only.

## Resume

The checkpoint directory contains model state, optimizer state when scheduled, RNG state, and
`resume_state.json`. Resume state includes the dataset position and the partially consumed token buffer.
Use `--new_data` to reset the data position while keeping the model output directory.

## Training controls

Useful flags include:

- `--batch_size`
- `--epochs` (SFT)
- `--learning_rate`
- `--optimizer adafactor|lion|adamw`
- `--save_every`
- `--optimizer_save_every`
- `--precision fp32|fp16|bf16`
- `--save_dtype fp32|fp16|bf16`
- `--train_router_only`
- `--qat`
- `--qat_calib_batches`
- `--qat_export_dir`
- `--compile`
- `--cpu`

Final partial pretraining batches are flushed at end-of-stream instead of being silently discarded.
