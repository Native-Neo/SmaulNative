#!/usr/bin/env python3
# router_utils.py -- utilities for training an MoE RWKV-X model's router in isolation, typically
# right after merge_moe.py has upcycled dense checkpoints into experts.
from rwkv_x_core import RWKVXModel, RWKV_CMix_MoE


def set_router_only_training(model: RWKVXModel, router_only: bool) -> int:
    """When router_only=True, freezes every parameter except each RWKV_CMix_MoE's router.
    Since frozen params (requires_grad=False) never get a gradient tensor allocated, this also
    meaningfully cuts training memory versus fine-tuning the whole model."""
    if not model.cfg.is_moe:
        raise ValueError("set_router_only_training requires an MoE model (cfg.is_moe=True); "
                         "this checkpoint has no router -- did you mean to point --output_dir "
                         "at a merge_moe.py output instead?")

    router_params = {id(p) for module in model.modules() if isinstance(module, RWKV_CMix_MoE)
                     for p in module.gate.parameters()}
    n_trainable = 0
    for p in model.parameters():
        p.requires_grad_(id(p) in router_params if router_only else True)
        if p.requires_grad:
            n_trainable += p.numel()
    return n_trainable
