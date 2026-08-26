#!/usr/bin/env python3
########################################################################################################
# train_tokenizer.py
#
# Trains a byte-level BPE tokenizer directly on --dataset_dir (same recursive txt/jsonl/json/csv/
# parquet/source-file walk as train.py's pretrain mode, reused from universal_dataset.py so the exact
# same text extraction logic is used everywhere -- no duplicated/divergent parsing).
#
# Output is ONE file, tokenizer.json: HuggingFace `tokenizers`' native save format, which folds vocab,
# merge rules, special tokens, and the byte-level pre/post-processing config into a single JSON
# document. That's what "merged tokenizer" means here -- it replaces the old two-piece setup this
# project used to depend on (rwkv_tokenizer.py's TRIE_TOKENIZER class + the separate fixed-vocab
# rwkv_vocab_v20230424.txt file). Neither of those upstream files is imported anywhere anymore; once
# you've trained your own tokenizer.json you can delete tokenizer/rwkv_tokenizer.py and
# tokenizer/rwkv_vocab_v20230424.txt.
#
# Usage:
#   python train_tokenizer.py --dataset_dir ./datasets --output ./tokenizer.json --vocab_size 32768
#
# train.py auto-invokes this (via train_tokenizer()) the first time it doesn't find a tokenizer.json,
# so running it by hand is optional -- but useful if you want to train the tokenizer once on your full
# corpus before kicking off a long pretraining run, or want a bigger/smaller vocab than the default.
########################################################################################################

import argparse
from pathlib import Path
from typing import Iterator, List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer

from dataset import discover_files, iter_texts

SPECIAL_TOKENS = ["<pad>", "<eos>"]


def text_iterator(dataset_dir: Path) -> Iterator[str]:
    files = discover_files(dataset_dir)
    if not files:
        raise RuntimeError(f"No supported files found under {dataset_dir}")
    for text, _path, _idx in iter_texts(files):
        yield text


def train_tokenizer(dataset_dir: Path, output_path: Path, vocab_size: int = 32768,
                     min_frequency: int = 2, special_tokens: List[str] = None) -> Tokenizer:
    special_tokens = special_tokens or SPECIAL_TOKENS

    tok = Tokenizer(BPE(unk_token=None))
    tok.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    tok.post_processor = ByteLevelProcessor(trim_offsets=True)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    print(f"[TOKENIZER] training BPE (target vocab_size={vocab_size}) on {dataset_dir} ...")
    tok.train_from_iterator(text_iterator(dataset_dir), trainer=trainer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(output_path))
    print(f"[TOKENIZER] saved merged tokenizer -> {output_path} "
          f"(actual vocab_size={tok.get_vocab_size()})")
    return tok


def parse_args():
    p = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer on --dataset_dir")
    p.add_argument("--dataset_dir", type=str, default="./datasets")
    p.add_argument("--output", type=str, default="./tokenizer.json")
    p.add_argument("--vocab_size", type=int, default=32768)
    p.add_argument("--min_frequency", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    train_tokenizer(Path(args.dataset_dir), Path(args.output), args.vocab_size, args.min_frequency)


if __name__ == "__main__":
    main()
