import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn.functional as F
from rwkv_x_core import RWKVXConfig, CausalSelfAttention

torch.manual_seed(0)
cfg = RWKVXConfig(n_embd=64, head_size=16, moba_chunk_size=8, moba_topk=2, n_layer=1)
att = CausalSelfAttention(cfg)
B, T, C = 2, 37, cfg.n_embd
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
    kc = F.pad(k, (0, 0, 0, pad)).view(B, H, n_chunks, cs, N)
    km = kc.mean(3)
    mask = torch.zeros(B, H, T, T, dtype=torch.bool)
    for i in range(n_chunks):
        lo, hi = i * cs, min((i + 1) * cs, T)
        if i: top = torch.einsum('bhn,bhcn->bhc', q[:, :, lo:hi].mean(2), km[:, :, :i]).topk(min(k_top, i), -1).indices
        if i:
            for b in range(B):
                for h in range(H):
                    for c in top[b, h].tolist():
                        mask[b, h, lo:hi, c * cs:(c + 1) * cs] = True
        for r in range(lo, hi):
            mask[:, :, r, lo:r + 1] = True
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return att.output(y.transpose(1, 2).contiguous().view(B, T, C))


chunked_out, _ = att(x)
ref_out = dense_ref(att, x)
diff = (chunked_out - ref_out).abs().max().item()
print('dense max abs diff:', diff)
assert diff < 1e-4
chunked_out.sum().backward()
assert x.grad is not None

with torch.no_grad():
    prompt = x.detach()[:1, :13].clone()
    full, _ = att(prompt)
    cache = att(prompt, use_cache=True)[1]
    step = torch.randn(1, 1, C)
    cached, _ = att(step, cache=cache, use_cache=True)
    reference, _ = att(torch.cat([prompt, step], 1))
    # Cached decoding should preserve the current partial chunk and selected past chunks.
    assert torch.allclose(cached, reference[:, -1:], rtol=1e-4, atol=1e-5)
print('PASS')
