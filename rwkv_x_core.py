
@dataclass
class RWKVXConfig:
    vocab_size: int = 65530
    n_embd: int = 832
    n_layer: int = 17
    head_size: int = 64
    n_moba_layer: int = 5
    moba_chunk_size: int = 512
    moba_topk: int = 4
    dropout: float = 0.0
    head_size_divisor: int = 8
    ctx_len_hint: int = 2048
    wkv_chunk_size: int = 64
    checkpoint_ffn: bool = True
    is_moe: bool = False
    num_experts: int = 1
    num_experts_per_tok: int = 1
    qat_bits: int = 0
    rqt_bits: int = 0
    fp8_training: bool = False
    quantization_bits: int = 0
    rqt_mixed: bool = False
    tokenizer_sha256: str = ""
    dataset_fingerprint: str = ""

    def save(self, path: Path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path):
        return cls(**json.loads(Path(path).read_text()))