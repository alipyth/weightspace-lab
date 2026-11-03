"""Harraf 2: a ~70M-parameter Persian LM built to learn from little data on a laptop CPU.

Compared with ``farsitree.model.LM`` (Harraf 1):

* the latent tree is kept: a causal parser reads the output of block ``parse_layer`` and
  every later attention layer subtracts ``softplus(s_h) * height`` from its logits;
* attention is grouped-query (n_kv_head key/value heads shared by n_head query heads), with
  RMSNorm and a SwiGLU MLP;
* **memory layers** (product-key memory, Lample et al. 2019): in the blocks listed in
  ``mem_layers`` a large table of n_keys**2 slots is added beside the MLP. Each token builds a
  query, finds its mem_topk best slots through two small sets of sub-keys (2 * n_keys scores
  instead of n_keys**2) and adds their values, weighted by a softmax. The table holds most of
  the parameters but only mem_topk rows are read per token, so it adds knowledge capacity at
  almost no compute. Its values are an ``EmbeddingBag`` with sparse gradients, trained by
  SparseAdam (``make_optimizer``): learning a new text only touches the slots it reads.

``build_model`` restores either version from a saved config dict.
"""
from dataclasses import dataclass, fields
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import LM, LatentParser, ModelConfig, apply_rope, rope_cache, tree_heights


@dataclass
class Harraf2Config:
    vocab_size: int = 8000
    n_layer: int = 12
    n_head: int = 8
    n_kv_head: int = 2
    d_model: int = 512
    mlp_hidden: int = 1408
    max_seq_len: int = 256
    tree: bool = True
    parse_layer: int = 2
    tree_tau: float = 4.0
    mem_layers: tuple = (4, 9)
    mem_keys: int = 181            # slots per memory layer = mem_keys ** 2 (32,761)
    mem_topk: int = 16             # slots read per token
    dropout: float = 0.0
    version: int = 2


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, cfg, use_tree):
        super().__init__()
        self.nh, self.nkv, self.hd = cfg.n_head, cfg.n_kv_head, cfg.d_model // cfg.n_head
        self.q = nn.Linear(cfg.d_model, self.nh * self.hd, bias=False)
        self.kv = nn.Linear(cfg.d_model, 2 * self.nkv * self.hd, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.use_tree = use_tree
        if use_tree:
            self.tree_slope = nn.Parameter(torch.full((self.nh,), -0.5))

    def forward(self, x, cos, sin, heights=None):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, self.nkv, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin).to(v.dtype), apply_rope(k, cos, sin).to(v.dtype)   # same dtype under fp16 autocast
        rep = self.nh // self.nkv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        if self.use_tree and heights is not None:
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            bias = -F.softplus(self.tree_slope).view(1, -1, 1, 1) * heights.unsqueeze(1)
            bias = bias.masked_fill(~causal, float("-inf")).to(q.dtype)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Memory(nn.Module):
    """Product-key memory: n_keys**2 value slots, the topk best read per token."""

    def __init__(self, d, n_keys, topk):
        super().__init__()
        self.n_keys, self.topk, self.half = n_keys, topk, d // 2
        self.query = nn.Linear(d, d, bias=False)
        self.keys = nn.Parameter(torch.randn(2, n_keys, self.half) / math.sqrt(self.half))
        self.values = nn.EmbeddingBag(n_keys * n_keys, d, mode="sum", sparse=True)
        nn.init.normal_(self.values.weight, std=d ** -0.5)
        self.register_buffer("hits", torch.zeros(n_keys * n_keys))      # how often each slot was read

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).reshape(-1, 2, self.half)
        q = q / (q.float().norm(dim=-1, keepdim=True).to(q.dtype) + 1e-6)       # cosine scores: stable
        scores = torch.einsum("nph,pkh->npk", q, self.keys.to(q.dtype)).float() * math.sqrt(self.half)
        k = self.topk
        s1, i1 = scores[:, 0].topk(k, dim=-1)
        s2, i2 = scores[:, 1].topk(k, dim=-1)
        cand = (s1.unsqueeze(2) + s2.unsqueeze(1)).reshape(-1, k * k)
        cand_i = (i1.unsqueeze(2) * self.n_keys + i2.unsqueeze(1)).reshape(-1, k * k)
        best, pos = cand.topk(k, dim=-1)
        idx = cand_i.gather(1, pos)
        w = torch.softmax(best, dim=-1)
        if self.training:
            with torch.no_grad():
                self.hits.index_add_(0, idx.reshape(-1), torch.ones(idx.numel(), device=idx.device))
        out = self.values(idx, per_sample_weights=w.to(self.values.weight.dtype))
        return out.view(B, T, C).to(x.dtype)


class Block(nn.Module):
    def __init__(self, cfg, use_tree, memory):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, use_tree)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden)
        self.mem = Memory(cfg.d_model, cfg.mem_keys, cfg.mem_topk) if memory else None
        self.ln3 = RMSNorm(cfg.d_model) if memory else None

    def forward(self, x, cos, sin, heights=None):
        x = x + self.attn(self.ln1(x), cos, sin, heights)
        x = x + self.mlp(self.ln2(x))
        if self.mem is not None:
            x = x + self.mem(self.ln3(x))
        return x


class Harraf2(nn.Module):
    def __init__(self, cfg: Harraf2Config):
        super().__init__()
        self.cfg = cfg
        self.arch = "harraf2"
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, cfg.tree and i >= cfg.parse_layer, i in cfg.mem_layers)
                                     for i in range(cfg.n_layer)])
        self.parser = LatentParser(cfg.d_model) if cfg.tree else None
        self.ln_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight
        cos, sin = rope_cache(cfg.max_seq_len, cfg.d_model // cfg.n_head, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        for n, p in self.named_parameters():
            if p.dim() == 2 and "values" not in n and "keys" not in n:
                nn.init.normal_(p, std=0.02)
            if n.endswith("proj.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def memory_layers(self):
        return [b.mem for b in self.blocks if b.mem is not None]

    def sparse_parameters(self):
        return [m.values.weight for m in self.memory_layers()]

    def dense_parameters(self):
        sparse = {id(p) for p in self.sparse_parameters()}
        return [p for p in self.parameters() if id(p) not in sparse]

    def active_params(self):
        """Parameters used for one token: everything except the unread memory slots."""
        mem = self.memory_layers()
        return self.num_params() - sum(m.values.weight.numel() for m in mem) + sum(m.topk * self.cfg.d_model for m in mem)

    def forward(self, idx, targets=None, return_hidden=False, return_dist=False):
        x = self.wte(idx)
        heights = dist = None
        for i, blk in enumerate(self.blocks):
            if i == self.cfg.parse_layer and self.parser is not None:
                dist = self.parser(x)
                heights = tree_heights(dist, self.cfg.tree_tau)
            x = blk(x, self.rope_cos, self.rope_sin, heights)
        h = self.ln_f(x)
        logits = self.head(h)
        out = {"logits": logits}
        if targets is not None:
            out["loss"] = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-1)
        if return_hidden:
            out["hidden"] = h
        if return_dist:
            out["dist"] = dist
            out["boundary"] = None
        return out


def build_model(cfg):
    """A model from a saved config dict (Harraf 2 when it says version 2, otherwise Harraf 1)."""
    if cfg.get("version") == 2:
        keep = {f.name for f in fields(Harraf2Config)}
        c = {k: v for k, v in cfg.items() if k in keep}
        c["mem_layers"] = tuple(c.get("mem_layers", ()))
        return Harraf2(Harraf2Config(**c))
    keep = {f.name for f in fields(ModelConfig)}
    c = {k: v for k, v in cfg.items() if k in keep}
    if "cca_rates" in c:
        c["cca_rates"] = tuple(c["cca_rates"])
    return LM(ModelConfig(**c))


class Optimizers:
    """AdamW for the dense weights and SparseAdam for memory values, used as one optimizer.
    Each parameter group carries lr_scale: callers set lr = base * lr_scale."""

    def __init__(self, *opts):
        self.opts = opts

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        return {"opts": [o.state_dict() for o in self.opts]}

    def load_state_dict(self, sd):
        for o, s in zip(self.opts, sd["opts"]):
            o.load_state_dict(s)


def make_optimizer(model, lr, mem_lr_scale=2.0):
    if not hasattr(model, "sparse_parameters") or not model.sparse_parameters():
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
        for g in opt.param_groups:
            g["lr_scale"] = 1.0
        return opt
    dense = torch.optim.AdamW(model.dense_parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    sparse = torch.optim.SparseAdam(model.sparse_parameters(), lr=lr * mem_lr_scale, betas=(0.9, 0.95))
    dense.param_groups[0]["lr_scale"], sparse.param_groups[0]["lr_scale"] = 1.0, mem_lr_scale
    return Optimizers(dense, sparse)


def clip_gradients(model, max_norm=1.0):
    """Clip the dense gradients (memory values have sparse gradients and their own optimizer)."""
    params = model.dense_parameters() if hasattr(model, "dense_parameters") else model.parameters()
    return torch.nn.utils.clip_grad_norm_(params, max_norm)
