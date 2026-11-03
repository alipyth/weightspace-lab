"""Two decoder-only LMs that differ only in positional information.

* ``baseline``: a standard GPT-style transformer with RoPE (ordinal position only).
* ``tree``: the same transformer plus a *latent, self-learned* tree position.
  A small causal "parser" predicts a syntactic distance d_k between token k-1
  and token k.  For a query i and key j (j < i) the height of the smallest
  constituent that contains both is approximated by a soft maximum of
  d_{j+1..i}.  Each attention head subtracts slope_h * height from its logits,
  so heads can choose to stay inside a constituent or look across it.
  RoPE is kept, so the model has ordinal *and* hierarchical position.

The parser reads hidden states from layer ``parse_layer`` and the bias is
applied in every later layer.  Everything is causal: d_k only depends on
tokens <= k, so the model remains a valid left-to-right LM.

* ``cca`` (constituent-compressed attention): every token attends to the
  last ``window`` tokens directly, and to anything further back only through
  one summary vector per *closed constituent*.  Constituent boundaries come
  from the same latent parser, at two nested levels (phrases and larger
  units: b_t = sigmoid((d_t - theta_level) / tau)), and half the heads read
  each level.  Long-range information has to pass through the induced
  structure, so the model cannot ignore it, and at inference a token only
  needs window + (number of closed constituents) keys instead of all T.
  A two-sided penalty pins the boundary rate to a target compression ratio,
  so the model chooses where constituents end but cannot switch the
  long-range path off (with a one-sided penalty it learned rate 0 and became
  a local-attention model).
* ``cca_fixed``: the same attention with boundaries at fixed intervals
  (every 3 and every 8 tokens).  The ablation that shows whether the learned
  structure, not just the compression, does the work.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    max_seq_len: int = 1024
    dropout: float = 0.0
    tree: bool = False
    arch: str = ""            # "", "cca" or "cca_fixed"; "" means tree-bias if tree else full attention
    window: int = 16          # cca: tokens attended directly
    cca_rates: tuple = (0.3, 0.12)   # cca: target boundary rate for each level
    cca_lambda: float = 20.0  # cca: weight of the (rate - target)^2 penalty
    cca_sharp: float = 2.0    # cca: weight of the mean b(1 - b) penalty
    parse_layer: int = 2      # parser reads the output of this many blocks
    tree_tau: float = 4.0     # sharpness of the soft maximum


def rope_cache(seq_len, head_dim, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, Dh)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    T = x.size(-2)
    cos, sin = cos[:T], sin[:T]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class LatentParser(nn.Module):
    """Predicts one syntactic distance per token from causal context."""

    def __init__(self, d_model, kernel=3):
        super().__init__()
        self.kernel = kernel
        self.conv = nn.Conv1d(d_model, d_model, kernel, groups=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))

    def forward(self, h):
        x = F.pad(h.transpose(1, 2), (self.kernel - 1, 0))      # left pad => causal
        x = self.conv(x).transpose(1, 2)
        return self.mlp(self.norm(x + h)).squeeze(-1)           # (B, T)


def tree_heights(dist, tau):
    """Soft height of the smallest span covering (j, i] for every i >= j.

    height[b, i, j] = (1/tau) * logsumexp_{k=j+1..i} tau * d_k, and 0 for i == j.
    A reversed logcumsumexp per query row keeps it exact when a large distance
    earlier in the sequence would swamp a single running sum.  O(T^2) memory.
    """
    B, T = dist.shape
    idx = torch.arange(T, device=dist.device)
    future = idx.view(1, T, 1) < idx.view(1, 1, T)              # key position k > query i
    D = (dist.float() * tau).unsqueeze(1).expand(B, T, T).masked_fill(future, -1e4)
    R = torch.logcumsumexp(D.flip(-1), dim=-1).flip(-1)         # R[i, j] = LSE over j <= k <= i
    h = torch.cat([R[..., 1:], R.new_zeros(B, T, 1)], dim=-1) / tau
    return h.masked_fill(~(idx.view(1, T, 1) > idx.view(1, 1, T)), 0.0)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, use_tree: bool):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.use_tree = use_tree
        if use_tree:
            # softplus(-0.5) ~ 0.47; heads can shrink it to ~0 to ignore the tree
            self.tree_slope = nn.Parameter(torch.full((cfg.n_head,), -0.5))

    def forward(self, x, cos, sin, heights=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        drop = self.dropout if self.training else 0.0
        if self.use_tree and heights is not None:
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            bias = -F.softplus(self.tree_slope).view(1, -1, 1, 1) * heights.unsqueeze(1)
            bias = bias.masked_fill(~causal, float("-inf")).to(q.dtype)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=drop)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


def constituent_structure(logits):
    """Soft constituents from boundary logits.

    logits: (B, L, T); logit[t] says how likely a new constituent starts at t.
    Returns pool (B, L, T, T), whose row j averages the constituent that ends
    at j, and gate (B, L, T), log P(the constituent ending at j is closed),
    which is the boundary probability at j + 1.
    """
    B, L, T = logits.shape
    keep = F.logsigmoid(-logits.float())                        # log(1 - b)
    cum = torch.cumsum(keep, dim=-1)
    idx = torch.arange(T, device=logits.device)
    causal = idx.view(T, 1) >= idx.view(1, T)
    A = torch.exp((cum.unsqueeze(-1) - cum.unsqueeze(-2)).masked_fill(~causal, -1e4))
    pool = A / A.sum(-1, keepdim=True)
    gate = torch.cat([F.logsigmoid(logits.float())[..., 1:], logits.new_full((B, L, 1), -30.0)], dim=-1)
    return pool, gate


class CCAAttention(nn.Module):
    """Sliding window over tokens + constituent summaries beyond the window."""

    def __init__(self, cfg: ModelConfig, use_global: bool, n_levels: int = 2):
        super().__init__()
        self.n_head, self.window, self.n_levels = cfg.n_head, cfg.window, n_levels
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.use_global = use_global
        self.register_buffer("level_of_head", torch.arange(cfg.n_head) % n_levels, persistent=False)

    def forward(self, x, cos, sin, struct=None):
        B, T, C = x.shape
        H, dh = self.n_head, C // self.n_head
        q, k, v = self.qkv(x).view(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        i = torch.arange(T, device=x.device)
        diff = i.view(T, 1) - i.view(1, T)                     # query - key
        local = torch.zeros(T, T, device=x.device).masked_fill(~((diff >= 0) & (diff < self.window)), float("-inf"))
        if not self.use_global or struct is None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=local.to(q.dtype))
        else:
            pool, gate = struct
            summ = torch.einsum("bltk,bkc->bltc", pool.to(x.dtype), x)            # (B, L, T, C)
            ks, vs = F.linear(summ, self.qkv.weight[C:]).view(B, self.n_levels, T, 2, H, dh).unbind(3)
            lv = self.level_of_head
            ks = ks.permute(0, 3, 1, 2, 4)[:, torch.arange(H), lv]                  # (B, H, T, dh)
            vs = vs.permute(0, 3, 1, 2, 4)[:, torch.arange(H), lv]
            ks = apply_rope(ks, cos, sin)
            far = gate[:, lv].unsqueeze(2).expand(B, H, T, T).masked_fill(~(diff >= self.window), float("-inf"))
            mask = torch.cat([local.expand(B, H, T, T), far], dim=-1).to(q.dtype)
            y = F.scaled_dot_product_attention(q, torch.cat([k, ks], 2), torch.cat([v, vs], 2), attn_mask=mask)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg, use_tree, cca=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CCAAttention(cfg, use_global=use_tree) if cca else Attention(cfg, use_tree)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False), nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False), nn.Dropout(cfg.dropout))

    def forward(self, x, cos, sin, heights=None):
        x = x + self.attn(self.ln1(x), cos, sin, heights)
        return x + self.mlp(self.ln2(x))


class LM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.arch = cfg.arch or ("tree" if cfg.tree else "full")
        cca = self.arch.startswith("cca")
        self.blocks = nn.ModuleList(
            [Block(cfg, (cfg.tree or cca) and i >= cfg.parse_layer, cca) for i in range(cfg.n_layer)])
        self.parser = LatentParser(cfg.d_model) if self.arch in ("tree", "cca") else None
        if self.arch == "cca":
            # level 1 threshold, and the gap to the coarser level 2 threshold
            self.level_theta = nn.Parameter(torch.tensor([0.0, 1.0]))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight
        cos, sin = rope_cache(cfg.max_seq_len, cfg.d_model // cfg.n_head, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("mlp.2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def boundary_logits(self, x):
        """(dist, logits (B, 2, T)) for the cca variants."""
        T = x.size(1)
        if self.arch == "cca":
            dist = self.parser(x)
            th = torch.stack([self.level_theta[0], self.level_theta[0] + F.softplus(self.level_theta[1])])
            return dist, dist.unsqueeze(1) - th.view(1, 2, 1)
        pos = torch.arange(T, device=x.device)
        fixed = torch.stack([pos % 3 == 0, pos % 8 == 0]).float() * 20 - 10
        return None, fixed.unsqueeze(0).expand(x.size(0), 2, T)

    def forward(self, idx, targets=None, return_hidden=False, return_dist=False):
        x = self.drop(self.wte(idx))
        heights = dist = blog = None
        for i, blk in enumerate(self.blocks):
            if i == self.cfg.parse_layer and self.arch == "tree":
                dist = self.parser(x)
                heights = tree_heights(dist, self.cfg.tree_tau)
            elif i == self.cfg.parse_layer and self.arch.startswith("cca"):
                dist, blog = self.boundary_logits(x)
                heights = constituent_structure(blog)
            x = blk(x, self.rope_cos, self.rope_sin, heights)
        h = self.ln_f(x)
        logits = self.head(h)
        out = {"logits": logits}
        if targets is not None:
            out["loss"] = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        if return_hidden:
            out["hidden"] = h
        if return_dist:
            out["dist"] = dist
            out["boundary"] = None if blog is None else torch.sigmoid(blog)
        if self.arch == "cca" and blog is not None:
            b = torch.sigmoid(blog)
            rates = b.mean(dim=(0, 2))
            target = torch.tensor(self.cfg.cca_rates, device=rates.device)
            out["aux"] = self.cfg.cca_lambda * ((rates - target) ** 2).sum() + self.cfg.cca_sharp * (b * (1 - b)).mean()
            out["boundary_rate"] = rates.detach()
        return out
