# Run this on your machine (where torch is installed) before training with the
# block-sparse MOBA attention. Rebuilds the exact same chunk selection as a
# single full-T x T dense mask and checks the chunked loop matches it exactly.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from rwkv_x_core import RWKVXConfig, CausalSelfAttention

torch.manual_seed(0)
cfg = RWKVXConfig(n_embd=64, head_size=16, moba_chunk_size=8, moba_topk=2, n_layer=1)
att = CausalSelfAttention(cfg)
B, T, C = 2, 37, cfg.n_embd  # T not a multiple of chunk_size on purpose
x = torch.randn(B, T, C, requires_grad=True)


def dense_ref(att, x):
    B, T, C = x.shape
    H, N, cs, k_top = att.n_head, C // att.n_head, att.chunk_size, att.top_k
    q, k, v = att.receptance(x), att.key(x), att.value(x)
    q = q.view(B, T, H, N).transpose(1, 2)
    k = k.view(B, T, H, N).transpose(1, 2)
    v = v.view(B, T, H, N).transpose(1, 2)
    n_chunks = (T + cs - 1) // cs
    pad = n_chunks * cs - T
    k_c = (F.pad(k, (0, 0, 0, pad)) if pad else k).view(B, H, n_chunks, cs, N)
    k_mean = k_c.mean(dim=3)

    full_mask = torch.zeros(B, H, T, T, dtype=torch.bool)
    for i in range(n_chunks):
        lo, hi = i * cs, min((i + 1) * cs, T)
        if i == 0:
            for r in range(lo, hi):
                full_mask[:, :, r, lo:r + 1] = True
        else:
            n_pick = min(k_top, i)
            scores = torch.einsum("bhn,bhcn->bhc", q[:, :, lo:hi].mean(dim=2), k_mean[:, :, :i])
            top_c = scores.topk(n_pick, dim=-1).indices  # (B,H,n_pick)
            for b in range(B):
                for h in range(H):
                    for c_idx in top_c[b, h].tolist():
                        full_mask[b, h, lo:hi, c_idx * cs:(c_idx + 1) * cs] = True
            for r in range(lo, hi):
                full_mask[:, :, r, lo:r + 1] = True
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=full_mask)
    return att.output(y.transpose(1, 2).contiguous().view(B, T, C))


chunked_out = att(x)
ref_out = dense_ref(att, x)
diff = (chunked_out - ref_out).abs().max().item()
print("max abs diff:", diff, "(should be ~0, fp rounding only)")
assert diff < 1e-4, "MOBA chunked implementation does not match dense reference!"

chunked_out.sum().backward()
print("backward ok, x.grad is not None:", x.grad is not None)
print("PASS" if diff < 1e-4 else "FAIL")
