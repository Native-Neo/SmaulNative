#!/usr/bin/env python3

import argparse
import contextlib
import json
import os
import signal
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

if "--cpu" in sys.argv or "--rqt" in sys.argv:
    threads = str(os.environ.get("SMAUL_CPU_THREADS") or max(1, (os.cpu_count() or 2) // 2))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, threads)
    if "MKL_ENABLE_INSTRUCTIONS" not in os.environ:
        try:
            flags = Path("/proc/cpuinfo").read_text(errors="ignore")
            if " avx" in flags or "\navx " in flags:
                os.environ["MKL_ENABLE_INSTRUCTIONS"] = "AVX"
        except OSError:
            pass
    os.environ.setdefault("TORCHINDUCTOR_CPP_WRAPPER", "1")
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "1")
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN,CPP")

import torch
from torch.optim import Optimizer

from dataset import PretrainStream, SFTDataset, discover_files, iter_texts, load_tokenizer, tokenizer_vocab_size
from rwkv_x_core import RWKVXModel, RWKV_CMix_MoE
from stream_data import stream_dataset
from tokenizer import ensure_tokenizer
import qat

STOP_REQUESTED = False


def _sigint_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[Ctrl-C] Stop requested. Finishing current step, then saving checkpoint.")


signal.signal(signal.SIGINT, _sigint_handler)


def _format_size(num_bytes):
    units = ("B", "KiB", "MiB", "GiB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024


def _print_model_size(model):
    parameters = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    fp32_bytes = parameters * 4
    print(f"[MODEL] {parameters:,} parameters | trainable={trainable:,} | FP32={_format_size(fp32_bytes)}")
    return parameters


def _print_rqt_storage(model):
    packed_bytes = 0
    scale_bytes = 0
    for module in model.modules():
        quant = getattr(module, "quant", None)
        if quant is None or not hasattr(quant, "packed"):
            continue
        packed_bytes += quant.packed.numel() * quant.packed.element_size()
        if hasattr(quant, "scale"):
            scale_bytes += quant.scale.numel() * quant.scale.element_size()
    master_bytes = sum(param.numel() for param in model.parameters()) * 4
    print(f"[RQT] packed storage={_format_size(packed_bytes + scale_bytes)} | master FP32={_format_size(master_bytes)}")


class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if not state:
                    state["exp_avg"] = torch.zeros_like(param)
                exp_avg = state["exp_avg"]
                if weight_decay:
                    param.mul_(1 - lr * weight_decay)
                param.add_(exp_avg.mul(b1).add(param.grad, alpha=1 - b1).sign(), alpha=-lr)
                exp_avg.lerp_(param.grad, 1 - b2)
        return loss


def set_router_only_training(model, router_only):
    if not model.cfg.is_moe:
        raise ValueError("set_router_only_training requires cfg.is_moe=True")
    gates = {
        id(param)
        for module in model.modules()
        if isinstance(module, RWKV_CMix_MoE)
        for param in module.gate.parameters()
    }
    trainable_params = 0
    for param in model.parameters():
        param.requires_grad_(id(param) in gates if router_only else True)
        if param.requires_grad:
            trainable_params += param.numel()
    return trainable_params


class ResumeState:
    def __init__(self):
        self.global_step = 0
        self.total_tokens = 0
        self.file_path: Optional[str] = None
        self.record_index = 0
        self.epoch = 0
        self.buffer_tokens = []

    @classmethod
    def load(cls, path):
        state = cls()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if not isinstance(data, dict):
                    raise ValueError("resume state must be a JSON object")
                state.global_step = data.get("global_step", 0)
                state.total_tokens = data.get("total_tokens", 0)
                state.file_path = data.get("file_path")
                state.record_index = data.get("record_index", 0)
                state.epoch = data.get("epoch", 0)
                state.buffer_tokens = data.get("buffer_tokens", [])
            except Exception as exc:
                raise RuntimeError(f"could not load resume state {path}: {exc}") from exc
        return state

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.__dict__, indent=2))
        os.replace(tmp, path)


def _save_rng_state(path):
    state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _load_rng_state(path):
    if not path.exists():
        return
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "torch" not in state:
            raise ValueError("missing torch RNG state")
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as exc:
        raise RuntimeError(f"could not restore RNG state {path}: {exc}") from exc


def save_checkpoint(model, optimizer, resume, output_dir, checkpoint_dir, tokenizer_path, save_dtype="fp32", save_optimizer=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = getattr(model, "_orig_mod", model)
    model.save_pretrained(output_dir, dtype=save_dtype, include_upstream=False)
    bundled = output_dir / "tokenizer.json"
    if tokenizer_path.resolve() != bundled.resolve():
        shutil.copy2(tokenizer_path, bundled)
    if save_optimizer:
        tmp = checkpoint_dir / "optimizer.pt.tmp"
        torch.save(optimizer.state_dict(), tmp)
        os.replace(tmp, checkpoint_dir / "optimizer.pt")
        _save_rng_state(checkpoint_dir / "rng_state.pt")
    resume.save(checkpoint_dir / "resume_state.json")
    print(f"[SAVE COMPLETE] {output_dir}")


def _autocast(args, device):
    if device.type == "cuda":
        dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    if args.precision == "bf16":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _optimizer_step(args, model, optimizer, xb, yb, device, scaler):
    optimizer.zero_grad(set_to_none=True)
    with _autocast(args, device):
        _, loss, _ = model(xb, labels=yb)
    if not torch.isfinite(loss):
        print(f"[WARN] non-finite loss {loss.item()}, skipping step")
        return None
    if scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
    if scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    if args.rqt:
        import rqt
        rqt.refresh_rqt(model)
    return loss


def _remote_token_stream(name, tokenizer, ctx_len, resume):
    dataset = file_path = None
    record = 0
    if resume.file_path:
        if "::" not in resume.file_path:
            raise ValueError(f"invalid remote resume position: {resume.file_path!r}")
        dataset, file_path = resume.file_path.split("::", 1)
        record = resume.record_index
    buffer_tokens = list(resume.buffer_tokens)
    for text, position in stream_dataset(name, start_dataset=dataset, start_file=file_path, start_record=record, with_position=True):
        buffer_tokens.extend(tokenizer.encode(text) + [tokenizer.eos_token_id])
        while len(buffer_tokens) >= ctx_len + 1:
            chunk = buffer_tokens[:ctx_len + 1]
            del buffer_tokens[:ctx_len]
            yield (torch.tensor(chunk[:-1]), torch.tensor(chunk[1:]), f"{position[0]}::{position[1]}", position[2], list(buffer_tokens))


def _train_pretrain_batch(args, model, optimizer, resume, device, scaler, batch_x, batch_y, path, record, buffer_tokens):
    xb = torch.stack(batch_x).to(device)
    yb = torch.stack(batch_y).to(device)
    loss = _optimizer_step(args, model, optimizer, xb, yb, device, scaler)
    if loss is None:
        return None
    resume.global_step += 1
    resume.total_tokens += xb.numel()
    resume.file_path = path
    resume.record_index = record
    resume.buffer_tokens = buffer_tokens
    return loss


def train_pretrain(args, model, optimizer, resume, device, tokenizer, scaler):
    if args.stream_dataset != "none":
        stream = _remote_token_stream(args.stream_dataset, tokenizer, args.ctx_len, resume)
        remote = True
    else:
        if resume.file_path and not Path(resume.file_path).is_file():
            raise FileNotFoundError(f"resume dataset file no longer exists: {resume.file_path}")
        stream = PretrainStream(Path(args.dataset_dir), tokenizer, args.ctx_len, resume_file=resume.file_path, resume_record=resume.record_index, buffer_tokens=resume.buffer_tokens)
        remote = False
    model.train()
    batch_x, batch_y = [], []
    last_path = None
    last_record = 0
    last_buffer = []
    t0 = time.perf_counter()
    tokens_since_log = 0
    for item in stream:
        if remote:
            x, y, path, record, buffer_tokens = item
        else:
            x, y, position = item
            path, record = position
            buffer_tokens = stream.buffer_tokens
        last_path, last_record, last_buffer = path, record, buffer_tokens
        batch_x.append(x)
        batch_y.append(y)
        if len(batch_x) < args.batch_size:
            continue
        loss = _train_pretrain_batch(args, model, optimizer, resume, device, scaler, batch_x, batch_y, path, record, buffer_tokens)
        batch_x, batch_y = [], []
        if loss is None:
            continue
        tokens_since_log += args.ctx_len * args.batch_size
        if resume.global_step % args.log_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"step {resume.global_step} | loss {loss.item():.4f} | {tokens_since_log / max(elapsed, 1e-9):.1f} tok/s | tokens {resume.total_tokens:,}")
            t0 = time.perf_counter()
            tokens_since_log = 0
        if resume.global_step % args.save_every == 0 and resume.global_step % args.optimizer_save_every == 0:
            save_checkpoint(model, optimizer, resume, Path(args.output_dir), Path(args.checkpoint_dir), Path(args.tokenizer_path), args.save_dtype, True)
        if STOP_REQUESTED:
            break
    if batch_x and not STOP_REQUESTED:
        loss = _train_pretrain_batch(args, model, optimizer, resume, device, scaler, batch_x, batch_y, last_path, last_record, last_buffer)
        if loss is not None:
            print(f"[FLUSH] final partial batch size={len(batch_x)} | loss={loss.item():.4f}")


def train_sft(args, model, optimizer, resume, device, tokenizer, scaler):
    dataset = SFTDataset(Path(args.dataset_dir), tokenizer, args.ctx_len)
    model.train()
    for epoch in range(resume.epoch, args.epochs):
        generator = torch.Generator().manual_seed(epoch)
        permutation = torch.randperm(len(dataset), generator=generator).tolist()
        start = resume.record_index if epoch == resume.epoch else 0
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=permutation[start:])
        consumed = start
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = _optimizer_step(args, model, optimizer, xb, yb, device, scaler)
            consumed += len(xb)
            if loss is None:
                continue
            resume.global_step += 1
            resume.total_tokens += xb.numel()
            resume.epoch = epoch
            resume.record_index = consumed
            if resume.global_step % args.log_every == 0:
                print(f"epoch {epoch} step {resume.global_step} | loss {loss.item():.4f}")
            if resume.global_step % args.save_every == 0 and resume.global_step % args.optimizer_save_every == 0:
                save_checkpoint(model, optimizer, resume, Path(args.output_dir), Path(args.checkpoint_dir), Path(args.tokenizer_path), args.save_dtype, True)
            if STOP_REQUESTED:
                break
        if STOP_REQUESTED:
            break
        resume.epoch = epoch + 1
        resume.record_index = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pretrain", "sft"], required=True)
    parser.add_argument("--dataset_dir", default="./datasets")
    parser.add_argument("--stream_dataset", choices=["none", "hindi", "english", "openthoughts", "all"], default="none")
    parser.add_argument("--output_dir", default="./SmaulNative")
    parser.add_argument("--checkpoint_dir", default="./SmaulNative")
    parser.add_argument("--tokenizer_path", default="./SmaulNative/tokenizer.json")
    parser.add_argument("--tokenizer_vocab_size", type=int, default=65536)
    parser.add_argument("--tokenizer_max_records", type=int, default=5_000_000)
    parser.add_argument("--n_embd", type=int, default=832)
    parser.add_argument("--n_layer", type=int, default=17)
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--n_moba_layer", type=int, default=3)
    parser.add_argument("--ctx_len", type=int, default=1024)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--save_dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adafactor", "lion", "adamw"], default="adafactor")
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--optimizer_save_every", type=int, default=None)
    parser.add_argument("--new_data", action="store_true")
    parser.add_argument("--train_router_only", action="store_true")
    parser.add_argument("--qat", type=int, choices=(2, 4, 8), nargs="?", const=8, default=None)
    parser.add_argument("--rqt", type=int, choices=(2, 4, 8), nargs="?", const=8, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    args.precision = args.precision or ("fp16" if torch.cuda.is_available() and not args.cpu else "fp32")
    args.optimizer_save_every = args.optimizer_save_every or args.save_every
    if args.qat and args.rqt:
        parser.error("--qat and --rqt cannot be used together")
    if args.rqt and not args.cpu:
        parser.error("--rqt requires --cpu")
    if (args.tokenizer_max_records < 0 or args.n_embd <= 0 or args.head_size <= 0 or args.n_layer <= 0 or args.n_moba_layer < 0 or args.n_moba_layer >= args.n_layer or args.tokenizer_vocab_size <= 0 or args.batch_size <= 0 or args.ctx_len <= 0 or args.epochs <= 0 or args.learning_rate <= 0 or args.log_every <= 0 or args.save_every <= 0 or args.optimizer_save_every <= 0):
        parser.error("invalid model/training parameters")
    if args.n_embd % args.head_size:
        parser.error("--n_embd must be divisible by --head_size")
    if args.cpu and args.precision == "fp16":
        parser.error("--precision fp16 requires CUDA")
    if args.mode == "sft" and args.stream_dataset != "none":
        parser.error("--stream_dataset is supported for pretraining only")
    return args


def _requested_config(args, tokenizer):
    from rwkv_x_core import RWKVXConfig
    return RWKVXConfig(vocab_size=tokenizer_vocab_size(tokenizer), n_embd=args.n_embd, n_layer=args.n_layer, n_moba_layer=args.n_moba_layer, head_size=args.head_size, ctx_len_hint=args.ctx_len)


def _checkpoint_matches(config, requested):
    fields = ("vocab_size", "n_embd", "n_layer", "n_moba_layer", "head_size", "ctx_len_hint")
    return all(getattr(config, field) == getattr(requested, field) for field in fields)


def build_model(args, tokenizer):
    output_dir = Path(args.output_dir)
    requested = _requested_config(args, tokenizer)
    config_path = output_dir / "config.json"
    model_path = output_dir / "model.safetensors"
    if config_path.exists() and model_path.exists():
        from rwkv_x_core import RWKVXConfig
        try:
            existing = RWKVXConfig.load(config_path)
        except Exception as exc:
            print(f"[MODEL] ignoring invalid checkpoint config: {exc}")
        else:
            if _checkpoint_matches(existing, requested):
                print("[MODEL] loading compatible checkpoint")
                return RWKVXModel.from_pretrained(output_dir), True
            print(
                "[MODEL] checkpoint architecture mismatch; starting fresh "
                f"(requested {args.n_embd}/{args.n_layer}/{args.n_moba_layer}/{args.head_size}, "
                f"checkpoint {existing.n_embd}/{existing.n_layer}/{existing.n_moba_layer}/{existing.head_size})"
            )
    return RWKVXModel(requested), False


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[DEVICE] {device} | precision={args.precision}")
    if args.cpu:
        from cpu import configure
        print(f"[CPU] {configure()} threads, native WKV, compile={args.compile}")
    tokenizer_path = Path(args.tokenizer_path)
    tokenizer, tokenizer_rebuilt = ensure_tokenizer(Path(args.dataset_dir), tokenizer_path, args.tokenizer_vocab_size, args.stream_dataset, args.tokenizer_max_records)
    if tokenizer_rebuilt:
        print(f"[TOKENIZER] using requested vocabulary={tokenizer.get_vocab_size()}")
    model, loaded_checkpoint = build_model(args, tokenizer)
    model = model.to(device)
    print(f"[MODEL CONFIG] vocab={model.cfg.vocab_size} n_embd={model.cfg.n_embd} n_layer={model.cfg.n_layer} n_moba_layer={model.cfg.n_moba_layer} head_size={model.cfg.head_size} ctx_len={args.ctx_len} batch_size={args.batch_size}")
    _print_model_size(model)
    if args.train_router_only:
        trainable = set_router_only_training(model, True)
        print(f"[ROUTER-ONLY] {trainable:,} trainable params")
    if args.qat:
        n = qat.prepare_qat(model, args.qat)
        print(f"[QAT] fake-quantizing {n} linears")
    if args.rqt:
        import rqt
        n = rqt.prepare_rqt(model, args.rqt)
        print(f"[RQT] real-quantizing {n} linears")
        _print_rqt_storage(model)
    if args.compile:
        model = torch.compile(model, mode="max-autotune")
    checkpoint_dir = Path(args.checkpoint_dir)
    resume = ResumeState.load(checkpoint_dir / "resume_state.json") if loaded_checkpoint else ResumeState()
    if args.new_data:
        resume = ResumeState()
    elif loaded_checkpoint:
        _load_rng_state(checkpoint_dir / "rng_state.pt")
    optimizer_classes = {"lion": Lion, "adamw": torch.optim.AdamW, "adafactor": torch.optim.Adafactor}
    if args.cpu and args.optimizer == "lion":
        from cpu import NativeLion
        optimizer_classes["lion"] = NativeLion
    optimizer = optimizer_classes[args.optimizer](model.parameters(), lr=args.learning_rate)
    optimizer_path = checkpoint_dir / "optimizer.pt"
    if optimizer_path.exists() and loaded_checkpoint and not args.new_data:
        try:
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=False))
        except Exception as exc:
            raise RuntimeError(f"could not restore optimizer {optimizer_path}: {exc}") from exc
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and args.precision == "fp16" else None
    try:
        if args.mode == "pretrain":
            train_pretrain(args, model, optimizer, resume, device, tokenizer, scaler)
        else:
            train_sft(args, model, optimizer, resume, device, tokenizer, scaler)
    finally:
        save_checkpoint(model, optimizer, resume, Path(args.output_dir), checkpoint_dir, tokenizer_path, args.save_dtype, True)


if __name__ == "__main__":
    main()
