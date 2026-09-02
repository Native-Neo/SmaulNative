#!/usr/bin/env python3
# tokenizer.py -- trains a byte-level BPE tokenizer on --dataset_dir, saves one tokenizer.json
# (vocab+merges+special tokens in one file). train.py auto-runs this if tokenizer.json is missing.
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

# <pad>/<eos> are required by dataset.py's TokenizerWrapper. <bos>/<unk> are reserved here
# so they exist in the vocab for future use (e.g. prepending <bos>, or switching pre-tokenizers
# later); byte-level BPE itself never needs <unk> since every byte is representable.
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def text_iterator(dataset_dir: Path) -> Iterator[str]:
    files = discover_files(dataset_dir)
    if not files:
        raise RuntimeError(f"No supported files found under {dataset_dir}")
    for text, _path, _idx in iter_texts(files):
        yield text


def train_tokenizer(dataset_dir: Path, output_path: Path, vocab_size: int = 65536,
                     min_frequency: int = 2, special_tokens: List[str] = None) -> Tokenizer:
    special_tokens = special_tokens or SPECIAL_TOKENS
    tok = Tokenizer(BPE(unk_token=None))
    pre_tok = ByteLevelPreTokenizer(add_prefix_space=False)
    tok.pre_tokenizer = pre_tok
    tok.decoder = ByteLevelDecoder()
    tok.post_processor = ByteLevelProcessor(trim_offsets=True)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tok.alphabet(),  # seed all 256 byte tokens so none are ever missing
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
    p.add_argument("--output", type=str, default="./SmaulNative/tokenizer.json")
    p.add_argument("--vocab_size", type=int, default=65536)
    p.add_argument("--min_frequency", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    train_tokenizer(Path(args.dataset_dir), Path(args.output), args.vocab_size, args.min_frequency)


if __name__ == "__main__":
    main()
