# RWKV-X 256M - Microsoft StableQAT (3-Bit) Training Script
import argparse, datetime, glob, logging, os, warnings
from dataclasses import dataclass
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)


# =============================================================================
# Microsoft StableQAT Implementation (Learnable Step Size & Grad Scale)
# =============================================================================

class GradScaler(torch.autograd.Function):
    """Scales gradients during backward pass to stabilize step-size learning in low-bit QAT."""
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def grad_scale(x, scale):
    return GradScaler.apply(x, scale)


class Stable3BitQuantizerFunction(torch.autograd.Function):
    """Stable 3-Bit Quantizer with Straight-Through Estimator (STE)."""
    @staticmethod
    def forward(ctx, weight, step_size, qmin, qmax):
        ctx.qmin = qmin
        ctx.qmax = qmax

        # Normalized weight representation
        w_scaled = weight / step_size
        w_clipped = torch.clamp(w_scaled, qmin, qmax)
        w_quant = torch.round(w_clipped)
        w_dequant = w_quant * step_size

        ctx.save_for_backward(w_scaled, step_size)
        return w_dequant

    @staticmethod
    def backward(ctx, grad_output):
        w_scaled, step_size = ctx.saved_tensors
        qmin, qmax = ctx.qmin, ctx.qmax

        # Mask elements that exceeded quantization bounds
        zero_mask = (w_scaled < qmin) | (w_scaled > qmax)

        # Gradient w.r.t input weight (STE pass-through)
        grad_weight = grad_output.clone()
        grad_weight[zero_mask] = 0.0

        # Gradient w.r.t learnable step size (StableQAT formulation)
        w_clipped = torch.clamp(w_scaled, qmin, qmax)
        w_round = torch.round(w_clipped)

        grad_s_elem = torch.where(zero_mask, w_clipped, w_round - w_scaled)
        grad_step_size = (grad_output * grad_s_elem).sum().view_as(step_size)

        return grad_weight, grad_step_size, None, None


class StableQATLinear(nn.Module):
    """Wrapper that converts a standard Linear layer into a Microsoft StableQAT 3-bit layer."""

    def __init__(self, original_linear: nn.Linear, num_bits: int = 3):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.weight = original_linear.weight
        self.bias = original_linear.bias

        # 3-bit signed quantization bounds [-4, 3]
        self.qmin = -(2 ** (num_bits - 1))
        self.qmax = (2 ** (num_bits - 1)) - 1

        # Initialize learnable step size (s) based on weight norm
        init_step_size = (2.0 * self.weight.abs().mean()) / (self.qmax ** 0.5)
        self.step_size = nn.Parameter(torch.tensor([init_step_size], dtype=self.weight.dtype))

        # Gradient scale factor to prevent step-size gradient explosion
        self.grad_scale_factor = 1.0 / math_sqrt(self.weight.numel() * self.qmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply gradient scaling to step-size for training stability
        s_scaled = grad_scale(self.step_size, self.grad_scale_factor)
        s_clamped = torch.clamp(s_scaled, min=1e-8)

        # Quantize weights using StableQAT
        q_weight = Stable3BitQuantizerFunction.apply(
            self.weight, s_clamped, self.qmin, self.qmax
        )
        return F.linear(x, q_weight, self.bias)


def math_sqrt(val: float) -> float:
    return val ** 0.5


def apply_microsoft_stable_qat(module: nn.Module):
    """Recursively replaces all nn.Linear layers with 3-bit StableQATLinear modules."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, StableQATLinear(child, num_bits=3))
        else:
            apply_microsoft_stable_qat(child)


# =============================================================================
# Dataset Auto-Discovery
# =============================================================================

def find_dataset(datasets_dir: str = "./datasets") -> tuple[str, str]:
    """Scans the ./datasets directory and automatically picks up the dataset file."""
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir, exist_ok=True)

    files = [f for f in glob.glob(os.path.join(datasets_dir, "*")) if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(
            f"No dataset file found in '{datasets_dir}'. Please place your dataset file inside ./datasets"
        )

    selected_file = files[0]
    ext = os.path.splitext(selected_file)[1].lower()

    data_type_map = {
        ".bin": "binidx",
        ".idx": "binidx",
        ".npy": "numpy",
        ".txt": "utf-8",
        ".uint16": "uint16"
    }
    data_type = data_type_map.get(ext, "binidx")
    return selected_file, data_type


# =============================================================================
# Minimal CLI & Runner
# =============================================================================

def parse_minimal_args():
    parser = argparse.ArgumentParser(description="RWKV-X 256M Minimal Microsoft StableQAT (3-Bit) Trainer")
    parser.add_argument("--data_file", default="", type=str, help="Dataset path (auto-detected from ./datasets if left blank)")
    parser.add_argument("--load_model", default="", type=str, help="Path to baseline pretrained weights (.pth)")
    parser.add_argument("--proj_dir", default="out_256m_stableqat3bit", type=str, help="Output directory")
    parser.add_argument("--micro_bsz", default=8, type=int, help="Micro batch size per GPU")
    parser.add_argument("--devices", default=1, type=int, help="Number of GPUs")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_minimal_args()

    # Auto-detect dataset if not explicitly specified
    data_file, data_type = find_dataset("./datasets") if not cli_args.data_file else (cli_args.data_file, "binidx")

    @dataclass
    class RWKVConfig:
        # Fixed 256M Parameter Model Spec
        n_layer: int = 18
        n_embd: int = 1024
        ctx_len: int = 2048
        dim_att: int = 1024
        dim_ffn: int = 3584
        head_size_a: int = 64
        head_size_divisor: int = 8
        vocab_size: int = 0

        # Fixed Training Hyperparameters
        lr_init: float = 3e-4
        lr_final: float = 1e-5
        beta1: float = 0.9
        beta2: float = 0.99
        adam_eps: float = 1e-8
        grad_clip: float = 1.0
        weight_decay: float = 0.01
        epoch_steps: int = 1000
        epoch_count: int = 500
        epoch_begin: int = 0
        epoch_save: int = 5
        warmup_steps: int = -1

        # Minimal Hardware & System Options
        micro_bsz: int = cli_args.micro_bsz
        devices: int = cli_args.devices
        num_nodes: int = 1
        accelerator: str = "gpu"
        strategy: str = "auto"
        precision: str = "bf16"
        proj_dir: str = cli_args.proj_dir
        load_model: str = cli_args.load_model
        load_pretrain: str = ""
        data_file: str = data_file
        data_type: str = data_type

        # RWKV Specific Defaults
        train_type: str = ""
        my_testing: str = "x070"
        grad_cp: int = 0
        head_qk: int = 0
        pre_ffn: int = 0
        tiny_att_dim: int = 0
        tiny_att_layer: int = -999
        my_pos_emb: int = 0
        my_pile_version: int = 1
        my_pile_stage: int = 0
        my_pile_shift: int = -1
        my_pile_edecay: int = 0
        my_sample_len: int = 0
        my_ffn_shift: int = 1
        my_att_shift: int = 1
        magic_prime: int = 0
        my_qa_mask: int = 0
        my_random_steps: int = 0
        my_exit: int = 99999999
        my_exit_tokens: int = 0
        n_moba_layer: int = 1
        moba_chunk_size: int = 16
        moba_topk: int = 24
        only_train_moba: bool = False
        use_longce: bool = False
        ds_bucket_mb: int = 200

    args = RWKVConfig()

    warnings.filterwarnings("ignore")
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = args.grad_clip
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = args.epoch_count
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = args.num_nodes * args.devices * args.micro_bsz

    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    os.environ["RWKV_TRAIN_TYPE"] = args.train_type
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    os.environ["RWKV_JIT_ON"] = "1"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    os.makedirs(args.proj_dir, exist_ok=True)

    tokens_per_epoch = args.epoch_steps * args.real_bsz * args.ctx_len

    rank_zero_info(f"""
================================================================================
  RWKV-X 256M - Microsoft StableQAT (3-Bit Precision)
================================================================================
  Data Source  : {args.data_file} ({args.data_type})
  Quantization : Microsoft StableQAT (3-Bit Learnable Step-Size + Grad Scale)
  Architecture : 18 Layers | 1024 Embedding Dim | 2048 Context Length
  Hardware     : {args.devices} GPU(s) [{args.precision.upper()}] | Micro Batch: {args.micro_bsz}
  Tokens/Epoch : {tokens_per_epoch:,}
  Output Dir   : {args.proj_dir}
================================================================================
""")

    from src.dataset import MyDataset
    from src.model import RWKV, RWKVHybrid
    from src.trainer import train_callback

    train_data = MyDataset(args)
    args.vocab_size = train_data.vocab_size

    rwkv = RWKV(args)
    args.n_head = args.dim_att // args.head_size_a

    @dataclass
    class MOBAConfig:
        n_moba_layer: int = args.n_moba_layer
        n_head: int = args.n_head
        n_embd: int = args.n_embd
        moba_chunk_size: int = args.moba_chunk_size
        moba_topk: int = args.moba_topk

    model = RWKVHybrid(rwkv, args, MOBAConfig())

    if args.load_model:
        rank_zero_info(f">> Loading Weights: {args.load_model}")
        load_dict = torch.load(args.load_model, map_location="cpu", weights_only=False)
        model.load_state_dict(load_dict)

    # Inject Microsoft StableQAT 3-bit modules
    rank_zero_info(">> Injecting Microsoft StableQAT (3-bit learnable step-size layers)...")
    apply_microsoft_stable_qat(model)

    trainer = Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        logger=args.logger,
        callbacks=[train_callback(args)],
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        num_sanity_val_steps=args.num_sanity_val_steps,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=args.enable_checkpointing,
        accumulate_grad_batches=1,
        gradient_clip_val=args.gradient_clip_val,
    )

    data_loader = DataLoader(
        train_data,
        shuffle=False,
        pin_memory=True,
        batch_size=args.micro_bsz,
        num_workers=1,
        persistent_workers=False,
        drop_last=True,
    )

    trainer.fit(model, data_loader)
