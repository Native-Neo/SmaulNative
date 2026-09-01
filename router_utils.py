#!/usr/bin/env python3
# router_utils.py -- utilities for training an MoE RWKV-X model's router in isolation, typically
# right after merge_moe.py has upcycled dense checkpoints into experts.
from rwkv_x_core import RWKVXModel


def set_router_only_training(model: RWKVXModel, router_only: bool) -> int:
    """When router_only=True, freezes every parameter except each RWKV_CMix_MoE's router (the
    `.ffn.gate.` weight in each MoE Channel-Mix) -- lets you fine-tune how tokens get routed
    across already-trained experts (typically right after merge_moe.py) without disturbing the
    experts themselves. Since frozen params (requires_grad=False) never get a gradient tensor
    allocated, this also meaningfully cuts training memory versus fine-tuning the whole model.
    Pass router_only=False to unfreeze everything again (e.g. for a later joint fine-tuning
    phase once routing has settled). Returns the resulting trainable parameter count. Raises if
    the model isn't an MoE model -- there's no router to train."""
    if not model.cfg.is_moe:
        raise ValueError("set_router_only_training requires an MoE model (cfg.is_moe=True); "
                          "this checkpoint has no router -- did you mean to point --output_dir "
                          "at a merge_moe.py output instead?")
    n_trainable = 0
    for name, p in model.named_parameters():
        is_router_param = ".gate." in name
        p.requires_grad_(is_router_param if router_only else True)
        if p.requires_grad:
            n_trainable += p.numel()
    return n_trainable
