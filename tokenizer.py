#!/usr/bin/env python3
"""
tokenizer.py -- trains a byte-level BPE tokenizer.json on --dataset_dir,
reusing dataset.py's file walk/text extraction (same logic as train.py).

Usage:
  python tokenizer.py --dataset_dir ./datasets --output ./tokenizer.json --vocab_size 131072
"""
import argparse
from pathlib import Path
from typing import Iterator, List, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer

from dataset import discover_files, iter_texts  # bug: was `dataset`, module doesn't exist

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<cls>", "<mask>"] + \
                 [f"<extra_{i}>" for i in range(8)]  # reserved tokens for future use


def text_iterator(dataset_dir: Path) -> Iterator[str]:
    files = discover_files(dataset_dir)
    if not files:
        raise RuntimeError(f"No supported files found under {dataset_dir}")
    for text, _path, _idx in iter_texts(files):
        yield text


def train_tokenizer(dataset_dir: Path, output_path: Path, vocab_size: int = 131072,
                     min_frequency: int = 2, special_tokens: Optional[List[str]] = None) -> Tokenizer:
    special_tokens = special_tokens or SPECIAL_TOKENS
    tok = Tokenizer(BPE(unk_token="<unk>"))  # bug: unk_token=None broke unknown-byte fallback
    tok.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    tok.post_processor = ByteLevelProcessor(trim_offsets=True)

    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                          special_tokens=special_tokens, show_progress=True)

    print(f"[TOKENIZER] training BPE (vocab_size={vocab_size}) on {dataset_dir} ...")
    tok.train_from_iterator(text_iterator(dataset_dir), trainer=trainer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(output_path))
    print(f"[TOKENIZER] saved -> {output_path} (actual vocab_size={tok.get_vocab_size()})")
    return tok


def parse_args():
    p = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer on --dataset_dir")
    p.add_argument("--dataset_dir", type=str, default="./datasets")
    p.add_argument("--output", type=str, default="./tokenizer.json")
    p.add_argument("--vocab_size", type=int, default=131072)  # 128K vocab
    p.add_argument("--min_frequency", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_tokenizer(Path(args.dataset_dir), Path(args.output), args.vocab_size, args.min_frequency)
