#!/usr/bin/env python3
# tokenizer.py -- trains byte-level BPE tokenizer.
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
from stream_data import stream_dataset

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def text_iterator(dataset_dir: Path) -> Iterator[str]:
    files = discover_files(dataset_dir)
    if not files:
        raise RuntimeError(f"No supported files found under {dataset_dir}")
    for text, _path, _idx in iter_texts(files):
        yield text


def remote_text_iterator(dataset_name: str, max_records: int = 0) -> Iterator[str]:
    if dataset_name == "all":
        for name in ("hindi", "english", "openthoughts"):
            iterator = stream_dataset(name)
            if max_records:
                from itertools import islice
                iterator = islice(iterator, max_records)
            yield from iterator
        return
    iterator = stream_dataset(dataset_name)
    if max_records:
        from itertools import islice
        iterator = islice(iterator, max_records)
    yield from iterator


def train_tokenizer(dataset_dir: Path, output_path: Path, vocab_size: int = 65536,
                     min_frequency: int = 2, special_tokens: List[str] = None,
                     stream_name: str = "none", max_records: int = 0) -> Tokenizer:
    special_tokens = special_tokens or SPECIAL_TOKENS
    tok = Tokenizer(BPE(unk_token="<unk>"))
    pre_tok = ByteLevelPreTokenizer(add_prefix_space=False)
    tok.pre_tokenizer = pre_tok
    tok.decoder = ByteLevelDecoder()
    tok.post_processor = ByteLevelProcessor(trim_offsets=True)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tok.alphabet(),
    )
    if stream_name != "none":
        if max_records:
            limit = f", {max_records:,} records per dataset" if stream_name == "all" else f", max {max_records:,} records"
        else:
            limit = ""
        print(f"[TOKENIZER] streaming {stream_name} from Hugging Face{limit} ...")
        iterator = remote_text_iterator(stream_name, max_records)
    else:
        print(f"[TOKENIZER] training BPE on {dataset_dir} ...")
        iterator = text_iterator(dataset_dir)
    tok.train_from_iterator(iterator, trainer=trainer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(output_path))
    print(f"[TOKENIZER] saved -> {output_path} (actual vocab_size={tok.get_vocab_size()})")
    return tok


def parse_args():
    p = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer")
    p.add_argument("--dataset_dir", type=str, default="./datasets")
    p.add_argument("--stream_dataset", choices=["none", "hindi", "english", "openthoughts", "all"], default="none",
                    help="Stream training text directly from Hugging Face")
    p.add_argument("--max_records", type=int, default=5_000_000,
                    help="Maximum streamed records per dataset when using all; 0 means unlimited")
    p.add_argument("--output", type=str, default="./SmaulNative/tokenizer.json")
    p.add_argument("--vocab_size", type=int, default=65536)
    p.add_argument("--min_frequency", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    train_tokenizer(Path(args.dataset_dir), Path(args.output), args.vocab_size, args.min_frequency,
                    stream_name=args.stream_dataset, max_records=args.max_records)


if __name__ == "__main__":
    main()
