# Quickstart

Run commands from the repository root.

## Install

```bash
python -m pip install -r requirements.txt
```

The native CPU FP8 extension needs `ninja` and a working C++ compiler; without them
training still works via the torch fallback (see [cpu.md](cpu.md)).

## Data

Download pre-tokenized shards or generate synthetic data into `./datasets`:

```bash
python download.py --languages hindi english --max_rows 100000
# or: python syntheticdata.py --count 250000 --format both --output-dir ./datasets/synthetic
```

## Tokenizer

Train one explicitly (or let `train.py` build it automatically):

```bash
python tokenizer.py train --fromdataset ./datasets \
    --vocab-size 8000 --output ./runs/linear/tokenizer.json
```

`--vocab-size` must equal the `--vocab` you pass to `train.py`.

## Train

```bash
python train.py --data ./datasets --out ./runs/linear \
    --tokenizer ./runs/linear/tokenizer.json \
    --d 512 --layers 8 --heads 8 --vocab 8000 \
    --ctx 256 --batch 2 --steps 1000 --threads 2
```

See [train.md](train.md) for all flags, the Lion optimizer, and the checkpoint format.

## Inference

Single prompt, interactive chat, or local server -- all load `./runs/linear` by default:

```bash
python infer_linear.py --model ./runs/linear --prompt "Hello world" --max 64
python infer_cli.py --model ./runs/linear
python infer_server.py --model ./runs/linear --port 8080
```

See [inference.md](inference.md), [infer_cli.md](infer_cli.md), and
[infer_server.md](infer_server.md) for sampling flags, history controls, and auth.

## Preference RL

Collect human preferences and GRPO-train, or run verified automatic RL:

```bash
python rl.py --model_dir ./runs/linear --prompt "Explain gravity" --responses 8
python autorl.py --model_dir ./runs/linear --prompt "Explain gravity" --responses 8
```

See [rl.md](rl.md) and [autorl.md](autorl.md). Auto-labeling without `--no-verify`
requires at least 4 human preference records first.

## MoE + export

Merge dense checkpoints into a sparse MoE, and export a checkpoint to GGUF:

```bash
python merge_moe.py --base ./runs/base --branches ./runs/b1 ./runs/b2 --out ./runs/moe
python convert_linear_to_gguf.py ./runs/linear ./runs/linear.gguf --dtype f16
```

MoE merge copies the base `tokenizer.json` into the output (branch tokenizers must match)
and refuses non-empty `--out` without `--force`. GGUF export refuses to overwrite without
`--overwrite`.

## Benchmark

```bash
python benchmark.py --mode full --d 512 --layers 4 --ctx 256 --batch 2 --iters 10
```
