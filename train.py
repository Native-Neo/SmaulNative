#!/usr/bin/env python3

import argparse, json, os, shutil, signal, time
from pathlib import Path
from typing import Optional

import torch
from torch.optim import Optimizer

from rwkv_x_core import (
    RWKVXModel,
    config_for_target_params,
)
from dataset import (
    load_tokenizer,
    tokenizer_vocab_size,
    PretrainStream,
    SFTDataset,
)
from tokenizer import train_tokenizer


STOP_REQUESTED = False


def _sigint_handler(signum, frame):
    global STOP_REQUESTED

    if not STOP_REQUESTED:
        print(
            "\n[Ctrl-C] Stop requested. "
            "Finishing the current step before saving."
        )

    STOP_REQUESTED = True


signal.signal(signal.SIGINT, _sigint_handler)


class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01,
    ):
        if lr <= 0:
            raise ValueError("lr must be greater than zero")

        if not 0 <= betas[0] < 1:
            raise ValueError("Invalid beta1")

        if not 0 <= betas[1] < 1:
            raise ValueError("Invalid beta2")

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                gradient = parameter.grad

                if gradient.is_sparse:
                    raise RuntimeError(
                        "Lion does not support sparse gradients"
                    )

                state = self.state[parameter]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )

                exp_avg = state["exp_avg"]

                if weight_decay != 0:
                    parameter.mul_(
                        1.0 - lr * weight_decay
                    )

                update = (
                    exp_avg.mul(beta1)
                    .add(
                        gradient,
                        alpha=1.0 - beta1,
                    )
                )

                parameter.add_(
                    torch.sign(update),
                    alpha=-lr,
                )

                exp_avg.mul_(beta2).add_(
                    gradient,
                    alpha=1.0 - beta2,
                )

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
            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            state.global_step = int(
                data.get(
                    "global_step",
                    0,
                )
            )

            state.total_tokens = int(
                data.get(
                    "total_tokens",
                    0,
                )
            )

            state.file_path = data.get(
                "file_path"
            )

            state.byte_offset = int(
                data.get(
                    "byte_offset",
                    0,
                )
            )

            state.record_index = int(
                data.get(
                    "record_index",
                    0,
                )
            )

        except Exception as error:
            print(
                f"[WARN] Could not load resume state: "
                f"{error}"
            )

        return state

    def reset_dataset_position(self):
        self.global_step = 0
        self.total_tokens = 0

        self.file_path = None
        self.byte_offset = 0
        self.record_index = 0

    def save(self, path: Path):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "global_step": self.global_step,
            "total_tokens": self.total_tokens,
            "file_path": self.file_path,
            "byte_offset": self.byte_offset,
            "record_index": self.record_index,
        }

        temporary_path = path.with_suffix(
            ".json.tmp"
        )

        with open(
            temporary_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
            )

            file.flush()
            os.fsync(file.fileno())

        os.replace(
            temporary_path,
            path,
        )


def save_checkpoint(
    model: RWKVXModel,
    optimizer: Optimizer,
    resume: ResumeState,
    output_dir: Path,
    checkpoint_dir: Path,
    tokenizer_path: Path,
):
    print("\n[SAVE] Saving checkpoint...")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(output_dir)

    bundled_tokenizer = (
        output_dir / "tokenizer.json"
    )

    if (
        tokenizer_path.exists()
        and tokenizer_path.resolve()
        != bundled_tokenizer.resolve()
    ):
        temporary_tokenizer = (
            output_dir / "tokenizer.json.tmp"
        )

        shutil.copy2(
            tokenizer_path,
            temporary_tokenizer,
        )

        os.replace(
            temporary_tokenizer,
            bundled_tokenizer,
        )

    temporary_optimizer = (
        checkpoint_dir / "optimizer.pt.tmp"
    )

    torch.save(
        optimizer.state_dict(),
        temporary_optimizer,
    )

    os.replace(
        temporary_optimizer,
        checkpoint_dir / "optimizer.pt",
    )

    resume.save(
        checkpoint_dir
        / "resume_state.json"
    )

    print("[SAVE COMPLETE]")
    print(f"  Model:      {output_dir}")
    print(
        f"  Checkpoint: {checkpoint_dir}"
    )


def parse_stream_position(position):
    """
    Accept several PretrainStream position formats.

    Preferred format:

        (
            file_path,
            byte_offset,
            record_index,
        )

    Older dataset implementations may return:

        (
            file_path,
            record_index,
        )

    or a dictionary.
    """

    file_path = None
    byte_offset = 0
    record_index = 0

    if isinstance(position, dict):
        file_path = position.get(
            "file_path"
        )

        byte_offset = int(
            position.get(
                "byte_offset",
                0,
            )
        )

        record_index = int(
            position.get(
                "record_index",
                0,
            )
        )

        return (
            file_path,
            byte_offset,
            record_index,
        )

    if isinstance(
        position,
        (tuple, list),
    ):
        if len(position) >= 3:
            return (
                position[0],
                int(position[1]),
                int(position[2]),
            )

        if len(position) == 2:
            return (
                position[0],
                0,
                int(position[1]),
            )

        if len(position) == 1:
            return (
                position[0],
                0,
                0,
            )

    if isinstance(position, str):
        return (
            position,
            0,
            0,
        )

    return (
        file_path,
        byte_offset,
        record_index,
    )


def train_pretrain(
    args,
    model,
    optimizer,
    resume,
    device,
    tokenizer,
):
    stream = PretrainStream(
        Path(args.dataset_dir),
        tokenizer,
        args.ctx_len,
        resume_file=resume.file_path,
        resume_byte_offset=resume.byte_offset,
        resume_record=resume.record_index,
    )

    model.train()

    batch_x = []
    batch_y = []
    batch_positions = []

    log_start_time = time.perf_counter()
    tokens_since_log = 0

    for x, y, position in stream:
        batch_x.append(x)
        batch_y.append(y)
        batch_positions.append(position)

        if len(batch_x) < args.batch_size:
            continue

        xb = torch.stack(
            batch_x
        ).to(device)

        yb = torch.stack(
            batch_y
        ).to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits, loss, _ = model(
            xb,
            labels=yb,
        )

        if not torch.isfinite(loss):
            print(
                f"[WARN] Non-finite loss: "
                f"{loss.item()}"
            )

            batch_x.clear()
            batch_y.clear()
            batch_positions.clear()

            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        (
            file_path,
            byte_offset,
            record_index,
        ) = parse_stream_position(
            batch_positions[-1]
        )

        resume.global_step += 1
        resume.total_tokens += xb.numel()

        resume.file_path = (
            str(file_path)
            if file_path is not None
            else None
        )

        resume.byte_offset = byte_offset
        resume.record_index = record_index

        tokens_since_log += xb.numel()

        if (
            resume.global_step
            % args.log_every
            == 0
        ):
            elapsed = (
                time.perf_counter()
                - log_start_time
            )

            tokens_per_second = (
                tokens_since_log
                / max(elapsed, 1e-9)
            )

            print(
                f"step {resume.global_step} | "
                f"loss {loss.item():.4f} | "
                f"{tokens_per_second:.1f} tok/s | "
                f"tokens {resume.total_tokens:,} | "
                f"byte {resume.byte_offset:,}"
            )

            log_start_time = time.perf_counter()
            tokens_since_log = 0

        batch_x.clear()
        batch_y.clear()
        batch_positions.clear()

        if STOP_REQUESTED:
            break


def train_sft(
    args,
    model,
    optimizer,
    resume,
    device,
    tokenizer,
):
    dataset = SFTDataset(
        Path(args.dataset_dir),
        tokenizer,
        args.ctx_len,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    model.train()

    start_epoch = resume.record_index

    for epoch in range(
        start_epoch,
        args.epochs,
    ):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits, loss, _ = model(
                xb,
                labels=yb,
            )

            if not torch.isfinite(loss):
                print(
                    f"[WARN] Non-finite loss: "
                    f"{loss.item()}"
                )

                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            resume.global_step += 1
            resume.total_tokens += xb.numel()

            if (
                resume.global_step
                % args.log_every
                == 0
            ):
                print(
                    f"epoch {epoch + 1}/{args.epochs} | "
                    f"step {resume.global_step} | "
                    f"loss {loss.item():.4f} | "
                    f"tokens {resume.total_tokens:,}"
                )

            if STOP_REQUESTED:
                break

        resume.record_index = epoch + 1

        if STOP_REQUESTED:
            break


def parse_args():
    parser = argparse.ArgumentParser(
        description="RWKV-X training script"
    )

    parser.add_argument(
        "--mode",
        choices=[
            "pretrain",
            "sft",
        ],
        required=True,
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./datasets",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./RWKV-X-256M",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./tokenizer.json",
    )

    parser.add_argument(
        "--tokenizer_vocab_size",
        type=int,
        default=32768,
    )

    parser.add_argument(
        "--target_params",
        type=int,
        default=256_000_000,
    )

    parser.add_argument(
        "--n_embd",
        type=int,
        default=768,
    )

    parser.add_argument(
        "--head_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--n_moba_layer",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--ctx_len",
        type=int,
        default=512,
        help="Training BPTT window",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--optimizer",
        choices=[
            "lion",
            "adamw",
        ],
        default="lion",
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--new-data",
        "--new_data",
        action="store_true",
        dest="new_data",
        help=(
            "Reset dataset position and counters while "
            "keeping model and optimizer state."
        ),
    )

    return parser.parse_args()


def resolve_tokenizer(
    args,
    output_dir: Path,
):
    """
    Priority:

    1. tokenizer.json already bundled with the model checkpoint.
    2. tokenizer_path supplied through CLI.
    3. Train a tokenizer only if neither exists.
    """

    checkpoint_tokenizer = (
        output_dir / "tokenizer.json"
    )

    requested_tokenizer = Path(
        args.tokenizer_path
    )

    if checkpoint_tokenizer.exists():
        print(
            f"[TOKENIZER] Using checkpoint tokenizer: "
            f"{checkpoint_tokenizer}"
        )

        return checkpoint_tokenizer

    if requested_tokenizer.exists():
        print(
            f"[TOKENIZER] Using tokenizer: "
            f"{requested_tokenizer}"
        )

        return requested_tokenizer

    print(
        f"[TOKENIZER] No tokenizer found. "
        f"Training one at {requested_tokenizer}"
    )

    requested_tokenizer.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_tokenizer(
        Path(args.dataset_dir),
        requested_tokenizer,
        args.tokenizer_vocab_size,
    )

    if not requested_tokenizer.exists():
        raise RuntimeError(
            "Tokenizer training finished but tokenizer.json "
            "was not created."
        )

    return requested_tokenizer


def build_model(
    args,
    tokenizer,
):
    output_dir = Path(
        args.output_dir
    )

    config_path = (
        output_dir / "config.json"
    )

    model_path = (
        output_dir / "model.safetensors"
    )

    if (
        config_path.exists()
        and model_path.exists()
    ):
        print(
            f"[RESUME] Loading model from "
            f"{output_dir}"
        )

        return RWKVXModel.from_pretrained(
            output_dir
        )

    print(
        "[INIT] Creating new RWKV-X model"
    )

    vocab_size = tokenizer_vocab_size(
        tokenizer
    )

    config = config_for_target_params(
        args.target_params,
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        n_moba_layer=args.n_moba_layer,
        head_size=args.head_size,
    )

    config.ctx_len_hint = args.ctx_len

    model = RWKVXModel(
        config
    )

    print(
        f"[MODEL] "
        f"n_layer={config.n_layer} | "
        f"n_embd={config.n_embd} | "
        f"n_moba_layer={config.n_moba_layer} | "
        f"vocab_size={config.vocab_size} | "
        f"params={model.num_parameters() / 1e6:.1f}M"
    )

    return model


def load_optimizer(
    args,
    model,
    checkpoint_dir: Path,
    device,
):
    if args.optimizer == "lion":
        optimizer = Lion(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.99),
            weight_decay=0.01,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.01,
        )

    optimizer_path = (
        checkpoint_dir / "optimizer.pt"
    )

    if optimizer_path.exists():
        try:
            optimizer.load_state_dict(
                torch.load(
                    optimizer_path,
                    map_location=device,
                )
            )

            print(
                "[RESUME] Loaded optimizer state"
            )

        except Exception as error:
            print(
                f"[WARN] Could not restore optimizer: "
                f"{error}"
            )

    return optimizer


def main():
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError(
            "--batch_size must be at least 1"
        )

    if args.ctx_len < 2:
        raise ValueError(
            "--ctx_len must be at least 2"
        )

    if args.log_every < 1:
        raise ValueError(
            "--log_every must be at least 1"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[DEVICE] {device}"
    )

    output_dir = Path(
        args.output_dir
    )

    checkpoint_dir = Path(
        args.checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer_path = resolve_tokenizer(
        args,
        output_dir,
    )

    tokenizer = load_tokenizer(
        tokenizer_path
    )

    model = build_model(
        args,
        tokenizer,
    ).to(device)

    optimizer = load_optimizer(
        args,
        model,
        checkpoint_dir,
        device,
    )

    resume_path = (
        checkpoint_dir
        / "resume_state.json"
    )

    resume = ResumeState.load(
        resume_path
    )

    if args.new_data:
        print(
            "[NEW DATA] Resetting dataset position and counters."
        )

        resume.reset_dataset_position()

    print(
        f"[RESUME] "
        f"step={resume.global_step:,} | "
        f"tokens={resume.total_tokens:,} | "
        f"file={resume.file_path} | "
        f"byte={resume.byte_offset:,} | "
        f"record={resume.record_index}"
    )

    try:
        if args.mode == "pretrain":
            train_pretrain(
                args,
                model,
                optimizer,
                resume,
                device,
                tokenizer,
            )

        else:
            train_sft(
                args,
                model,
                optimizer,
                resume,
                device,
                tokenizer,
            )

    except KeyboardInterrupt:
        print(
            "\n[INTERRUPT] Training interrupted."
        )

    finally:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            resume=resume,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            tokenizer_path=tokenizer_path,
        )


if __name__ == "__main__":
    main()
