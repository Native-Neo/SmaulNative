# train.py

The one entrypoint for both pretraining and SFT -- a plain PyTorch loop (no Lightning/DeepSpeed),
with Ctrl-C checkpointing, full resume, native CPU WKV, and Hugging Face-style export.

## Run it

```bash
# pretrain from scratch (auto-trains a tokenizer if --tokenizer_path doesn't exist)
python train.py --mode pretrain --dataset_dir ./datasets --output_dir ./RWKV-X-256M

# CPU pretraining with the native C++ WKV backend
python train.py --cpu --mode pretrain --dataset_dir ./datasets --output_dir ./RWKV-X-256M

# SFT on top of a pretrained checkpoint
python train.py --mode sft --dataset_dir ./sft_data --output_dir ./RWKV-X-SFT --tokenizer_path ./RWKV-X-256M/tokenizer.json
```

## Key flags

| Flag | Default | What it does |
|---|---|---|
| `--mode` | *(required)* | `pretrain` or `sft` |
| `--cpu` | off | use the native C++ CPU WKV backend and CPU thread configuration |
| `--dataset_dir` | `./datasets` | training data (pretrain: any text; SFT: conversation JSON/JSONL) |
| `--output_dir` | `./RWKV-X-256M` | model checkpoint dir (`config.json`, `model.safetensors`, bundled `tokenizer.json`) |
| `--checkpoint_dir` | `./checkpoints` | optimizer state + resume position |
| `--tokenizer_path` | `./tokenizer.json` | auto-trained here if missing |
| `--tokenizer_vocab_size` | `32768` | only used if auto-training |
| `--n_embd` | `768` | hidden size (new model only) -- model size targets 256M parameters |
| `--head_size` | `64` | must divide `--n_embd` |
| `--n_moba_layer` | `3` | how many blocks use MOBA sparse attention |
| `--ctx_len` | `512` | train-time BPTT window -- keep this small on CPU |
| `--batch_size` | `1` | training batch size |
| `--epochs` | `3` | SFT only |
| `--learning_rate` | `1e-4` | optimizer learning rate |
| `--optimizer` | `lion` | `lion` or `adamw` |
| `--log_every` | `10` | steps between log lines |
| `--save_every` | `200` | steps between checkpoint saves |
| `--new_data` | off | reset dataset position while keeping model/optimizer weights |
| `--qat` | off | enable 3-bit QAT on Channel-Mix linears (see [qat.md](qat.md)) |
| `--qat_calib_batches` | `64` | batches used to settle fake-quant ranges before training |
| `--qat_export_dir` | none | after training, convert and save an int3 checkpoint |

## Notes

- **Resuming**: re-run the same command with the same `--output_dir`/`--checkpoint_dir` to restore model
  weights, optimizer state, training counters, and pretraining dataset position.
- **Ctrl-C is safe**: `SIGINT` finishes the current step and the checkpoint save path preserves the
  truncated-BPTT buffer before exiting.
- **New dataset, same model**: `--new_data` resets the dataset read-position without losing trained weights.
- **SFT after pretrain**: use the pretrained checkpoint's bundled `tokenizer.json` so the vocabulary stays
  consistent. A separate output/checkpoint directory is recommended for SFT.
- **Model size is fixed**: new models target `TARGET_PARAMS = 268_435_456` (256M binary/Mebi convention).
  `n_layer` is solved automatically by `config_for_target_params`; it is not a required size knob.
- **Configuration validation**: `--n_layer` must be positive when supplied, `--n_moba_layer` must be
  non-negative, and MOBA layers must leave at least one RWKV layer.
- **Optimizer**: Lion is the default and keeps one momentum buffer instead of AdamW's two. Gradients are
  clipped to norm 1.0, and a non-finite loss skips the optimizer step.
- **QAT**: `--qat` calibrates fake-quant ranges before training and `--qat_export_dir` can save a separate
  packed int3 checkpoint after training.
- **CPU WKV**: `--cpu` installs the native C++ WKV forward/backward implementation instead of the old
  TorchScript fallback. The kernel parallelizes independent batch/head work and is compiled for the host CPU.
  The native path requires a head size of at most 128.
- **CPU MOBA**: MOBA still uses PyTorch causal attention on CPU and remains O(T²), so reducing `--ctx_len`
  is important for CPU training speed.
