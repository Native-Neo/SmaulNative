# dataset.py -- multi-format dataset loader, pretrain + SFT.

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset

IGNORE_INDEX = -100


class TokenizerWrapper:
    def __init__(self, tokenizer):
        self._tok = tokenizer
        self.pad_token_id = tokenizer.token_to_id("<pad>")
        self.eos_token_id = tokenizer.token_to_id("<eos>")
        if self.pad_token_id is None or self.eos_token_id is None:
            raise ValueError("tokenizer.json is missing <pad>/<eos> special tokens")

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)

    def get_vocab_size(self) -> int:
        return self._tok.get_vocab_size()


def load_tokenizer(path: Path) -> TokenizerWrapper:
    if not Path(path).exists():
        raise FileNotFoundError(f"No tokenizer found at {path}. Run tokenizer.py first")
    from tokenizer import SmaulTokenizer
    return TokenizerWrapper(SmaulTokenizer.from_file(path))


def tokenizer_vocab_size(tok) -> int:
    if isinstance(tok, (str, Path)):
        tok = load_tokenizer(Path(tok))
    return tok.get_vocab_size()


TEXT_KEYS = (
    "text", "content", "document", "body", "code", "prompt", "completion",
)
SUPPORTED_SUFFIXES = {
    ".txt", ".text", ".jsonl", ".json", ".csv", ".parquet", ".py", ".cpp", ".c", ".h", ".hpp",
    ".cc", ".cxx", ".rs", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".cs", ".php", ".rb",
    ".swift", ".kt", ".kts", ".scala", ".sh", ".bash", ".zsh", ".html", ".css", ".scss", ".sql",
    ".md", ".rst", ".yaml", ".yml", ".toml", ".xml",
}
PLAIN_TEXT_SUFFIXES = SUPPORTED_SUFFIXES - {".jsonl", ".json", ".csv", ".parquet"}
