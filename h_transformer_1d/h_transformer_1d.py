import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# helpers

def exists(val):
    return val is not None

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        *,
        mult = 4
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class HAttention1D(nn.Module):
    def __init__(
        self,
        dim,
        *,
        causal = False,
        heads = 8,
        dim_head = 64,
        block_size = 16,
        num_hierarchies = 3
    ):
        super().__init__()
        self.causal = causal
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.block_size = block_size
        self.num_hierarchies = num_hierarchies
        inner_dim = heads * dim_head

        self.to_qkv = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None):
        b, n, h, device, bsz = *x.shape[:2], self.heads, x.device, self.block_size
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        return self.to_out(v)

# main class

class HTransformer1D(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        heads = 8,
        dim_head = 64,
        causal = False,
        ff_mult = 4,
        num_hierarchies = 3,  # number of hierarchical levels, defaults to 3 as used in the paper
        block_size = 16       # this is the Nr in the paper - Nb = (max_seq_len / tokens_per_block)
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.max_seq_len = max_seq_len

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, HAttention1D(dim, causal = causal, dim_head = dim_head, heads = heads, block_size = block_size, num_hierarchies = num_hierarchies)),
                PreNorm(dim, FeedForward(dim, mult = ff_mult))
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x, mask = None):
        b, n, device = *x.shape, x.device
        x = self.token_emb(x)

        for attn, ff in self.layers:
            x = ff(x) + x

        return self.to_logits(x)
