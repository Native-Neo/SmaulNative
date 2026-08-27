#!/usr/bin/env python3
# train.py
import argparse, json, os, shutil, signal, time
from pathlib import Path
from typing import Optional

import torch
from torch.optim import Optimizer

from rwkv_x_core import RWKVXModel, config_for_target_params
from dataset import load_tokenizer, tokenizer_vocab_size, PretrainStream, SFTDataset
from tokenizer import train_tokenizer

STOP_REQUESTED = False


def _sigint_handler(signum, frame):
    global STOP_REQUESTED
    if not STOP_REQUESTED:
        print("\n[Ctrl-C] Stop requested. Finishing the current step before saving.")
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, _sigint_handler)


class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01):
        if lr <= 0:
            raise ValueError("lr must be greater than zero")
        if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError("Invalid beta")
        super().__init__(params, {"lr": lr, "betas": betas, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, (beta1, beta2), wd = group["lr"], group["betas"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avg = state["exp_avg"]

                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                update = exp_avg.mul(beta1).add(p.grad, alpha=1.0 - beta1)
                p.add_(torch.sign(update), alpha=-lr)
                exp_avg.mul_(beta2).add_(p.grad, alpha=1.0 - beta2)

        return loss


class ResumeState:
    def __init__(self):
        self.global_step = 0
        self.total_tokens = 0
        self.file_path: Optional[str] = None
        self.byte_offset = 0
        self.record_index = 0

    @classmethod
    def load(cls, path: Path):
        state = cls()
        if not path.exists():
            return state
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            state.global_step = int(data.get("global_step", 0))
            state.total_tokens = int(data.get("total_tokens", 0))
            state.file_path = data.get("file_path")
            state.byte_offset = int(data.get("byte_offset", 0))
            state.record_index = int(data.get("record_index", 0))
        except Exception as e:
            print(f"[WARN] Could not load resume state: {e}")
        return state

    def reset_dataset_position(self):
        self.global_step = self.total_tokens = self.byte_offset = self.record_index = 0
        self.file_path = None

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "global_step": self.global_step, "total_tokens": self.total_tokens,
            "file_path": self.file_path, "byte_offset": self.byte_offset,
            "record_index": self.record_index,
        }
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)


def save_checkpoint(model, optimizer, resume, output_dir: Path, checkpoint_dir: Path, tokenizer_path: Path):
    print("\n[SAVE] Saving checkpoint...")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)

    bundled = output_dir / "tokenizer.json"
    if tokenizer_path.exists() and tokenizer_path.resolve() != bundled.resolve():
        tmp = output_dir / "tokenizer.json.tmp"
        shutil.copy2(tokenizer_path, tmp)
        os.replace(tmp, bundled)

    tmp_opt = checkpoint_dir / "optimizer.pt.tmp"
    torch.save(optimizer.state_dict(), tmp_opt)
    os.replace(tmp_opt, checkpoint_dir / "optimizer.pt")

    resume.save(checkpoint_dir / "resume_state.json")
    print(f"[SAVE COMPLETE]\n  Model:      {output_dir}\n  Checkpoint: {checkpoint_dir}")


def parse_stream_position(position):
    """Accept (file_path, byte_offset, record_index), (file_path, record_index), a dict, or a bare str."""
    if isinstance(position, dict):
        return (position.get("file_path"), int(position.get("byte_offset", 0)), int(position.get("record_index", 0)))
    if isinstance(position, (tuple, list)):
        if len(position) >= 3:
            return position[0], int(position[1]), int(position[2])
        if len(position) == 2:
            return position[0], 0, int(position[1])
        if len(position) == 1:
            return position[0], 0, 0
    if isinstance(position, str):
        return position, 0, 0
    return None, 0, 0


def train_pretrain(args, model, optimizer, resume, device, tokenizer):
    stream = PretrainStream(
        Path(args.dataset_dir), tokenizer, args.ctx_len,
        resume_file=resume.file_path, resume_byte_offset=resume.byte_offset, resume_record=resume.record_index,
    )
    model.train()
    batch_x, batch_y, batch_positions = [], [], []
    log_start, tokens_since_log = time.perf_counter(), 0

    for x, y, position in stream:
        batch_x.append(x)
        batch_y.append(y)
        batch_positions.append(position)
        if len(batch_x) < args.batch_size:
            continue

        xb, yb = torch.stack(batch_x).to(device), torch.stack(batch_y).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, loss, _ = model(xb, labels=yb)

        if not torch.isfinite(loss):
            print(f"[WARN] Non-finite loss: {loss.item()}")
            batch_x.clear(); batch_y.clear(); batch_positions.clear()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        file_path, byte_offset, record_index = parse_stream_position(batch_positions[-1])
        resume.global_step += 1
        resume.total_tokens += xb.numel()
        resume.file_path = str(file_path) if file_path is not None else None
        resume.byte_offset = byte_offset
        resume.record_index = record_index
        tokens_since_log += xb.numel()

        if resume.global_step % args.log_every == 0:
            elapsed = time.perf_counter() - log_start
            tps = tokens_since_log / max(elapsed, 1e-9)
            print(f"step {resume.global_step} | loss {loss.item():.4f} | {tps:.1f} tok/s | "
                  f"tokens {resume.total_tokens:,} | byte {resume.byte_offset:,}")
            log_start, tokens_since_log = time.perf_counter(), 0

        batch_x.clear(); batch_y.clear(); batch_positions.clear()
        if STOP_REQUESTED:
            break


def train_sft(args, model, optimizer, resume, device, tokenizer):
    dataset = SFTDataset(Path(args.dataset_dir), tokenizer, args.ctx_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model.train()

    for epoch in range(resume.record_index, args.epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, loss, _ = model(xb, labels=yb)

            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss: {loss.item()}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            resume.global_step += 1
            resume.total_tokens += xb.numel()
            if resume.global_step % args.log_every == 0:
                print(f"epoch {epoch + 1}/{args.epochs} | step {resume.global_step} | "
                      f"loss {loss.item():.4f} | tokens {resume.total_tokens:,}")
            if STOP_REQUESTED:
                break

        resume.record_index = epoch + 1
        if STOP_REQUESTED:
            break


def parse_args():
    p = argparse.ArgumentParser(description="RWKV-X training script")
    p.add_argument("--mode", choices=["pretrain", "sft"], required=True)
    p.add_argument("--dataset_dir", type=str, default="./datasets")
    p.add_argument("--output_dir", type=str, default="./RWKV-X-256M")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--tokenizer_path", type=str, default="./tokenizer.json")
    p.add_argument("--tokenizer_vocab_size", type=int, default=131072)  # sync'd with train_tokenizer.py's 128K default
    p.add_argument("--target_params", type=int, default=256_000_000)
    p.add_argument("--n_embd", type=int, default=768)
    p.add_argument("--head_size", type=int, default=64)
    p.add_argument("--n_moba_layer", type=int, default=3)
    p.add_argument("--ctx_len", type=int, default=4096, help="Training BPTT window")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=["lion", "adamw"], default="lion")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--new-data", "--new_data", action="store_true", dest="new_data",
                    help="Reset dataset position and counters while keeping model and optimizer state.")
    return p.parse_args()


def resolve_tokenizer(args, output_dir: Path):
    """Priority: 1) tokenizer bundled with checkpoint, 2) --tokenizer_path, 3) train one."""
    checkpoint_tokenizer = output_dir / "tokenizer.json"
    requested = Path(args.tokenizer_path)

    if checkpoint_tokenizer.exists():
        print(f"[TOKENIZER] Using checkpoint tokenizer: {checkpoint_tokenizer}")
        return checkpoint_tokenizer
    if requested.exists():
        print(f"[TOKENIZER] Using tokenizer: {requested}")
        return requested

    print(f"[TOKENIZER] No tokenizer found. Training one at {requested}")
    requested.parent.mkdir(parents=True, exist_ok=True)
    train_tokenizer(Path(args.dataset_dir), requested, args.tokenizer_vocab_size)

    if not requested.exists():
        raise RuntimeError("Tokenizer training finished but tokenizer.json was not created.")
    return requested


def build_model(args, tokenizer):
    output_dir = Path(args.output_dir)
    config_path, model_path = output_dir / "config.json", output_dir / "model.safetensors"

    if config_path.exists() and model_path.exists():
        print(f"[RESUME] Loading model from {output_dir}")
        return RWKVXModel.from_pretrained(output_dir)

    print("[INIT] Creating new RWKV-X model")
    vocab_size = tokenizer_vocab_size(tokenizer)
    config = config_for_target_params(
        args.target_params, vocab_size=vocab_size, n_embd=args.n_embd,
        n_moba_layer=args.n_moba_layer, head_size=args.head_size,
    )
    config.ctx_len_hint = args.ctx_len

    model = RWKVXModel(config)
    print(f"[MODEL] n_layer={config.n_layer} | n_embd={config.n_embd} | "
          f"n_moba_layer={config.n_moba_layer} | vocab_size={config.vocab_size} | "
          f"params={model.num_parameters() / 1e6:.1f}M")
    return model


def load_optimizer(args, model, checkpoint_dir: Path, device):
    if args.optimizer == "lion":
        optimizer = Lion(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    optimizer_path = checkpoint_dir / "optimizer.pt"
    if optimizer_path.exists():
        try:
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
            print("[RESUME] Loaded optimizer state")
        except Exception as e:
            print(f"[WARN] Could not restore optimizer: {e}")
    return optimizer


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")
    if args.ctx_len < 2:
        raise ValueError("--ctx_len must be at least 2")
    if args.log_every < 1:
        raise ValueError("--log_every must be at least 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    output_dir, checkpoint_dir = Path(args.output_dir), Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = resolve_tokenizer(args, output_dir)
    tokenizer = load_tokenizer(tokenizer_path)
    model = build_model(args, tokenizer).to(device)
    optimizer = load_optimizer(args, model, checkpoint_dir, device)

    resume = ResumeState.load(checkpoint_dir / "resume_state.json")
    if args.new_data:
        print("[NEW DATA] Resetting dataset position and counters.")
        resume.reset_dataset_position()

    print(f"[RESUME] step={resume.global_step:,} | tokens={resume.total_tokens:,} | "
          f"file={resume.file_path} | byte={resume.byte_offset:,} | record={resume.record_index}")

    try:
        (train_pretrain if args.mode == "pretrain" else train_sft)(args, model, optimizer, resume, device, tokenizer)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted.")
    finally:
        save_checkpoint(model, optimizer, resume, output_dir, checkpoint_dir, tokenizer_path)


if __name__ == "__main__":
    main()
