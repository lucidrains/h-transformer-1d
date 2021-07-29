from math import log2
import torch
from torch import nn, einsum, diagonal
import torch.nn.functional as F

from einops import rearrange, reduce, repeat

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
        x = self.norm(x)
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
        eps = 1e-8
    ):
        super().__init__()
        self.eps = eps
        self.causal = causal
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.block_size = block_size
        inner_dim = heads * dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None):
        b, n, h, device, bsz, causal, eps = *x.shape[:2], self.heads, x.device, self.block_size, self.causal, self.eps

        # derive queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split out heads, and also divide sequence into blocks

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # calculate number of levels until 2 x 2

        num_levels = int(log2(n // bsz)) - 1

        # coarsening

        coarsened_qkvs = [(q, k, v)]

        for level in range(num_levels):
            q = reduce(q, 'b (n r) d -> b n d', 'mean', r = 2)
            k = reduce(k, 'b (n r) d -> b n d', 'mean', r = 2)
            v = reduce(v, 'b (n r) d -> b n d', 'sum', r = 2)

            coarsened_qkvs.append((q, k, v))

        *coarsened_qkvs, top_level_qkvs = reversed(coarsened_qkvs)

        # half-attention function

        def calculate_Y_and_A(q, k, v, mask_A = False, remove_right_off_diagonals = False):
            if remove_right_off_diagonals:
                q, k, v = map(lambda t: rearrange(t, 'b (n r) z d -> b n r z d', r = 2), (q, k, v))
                q, k, v = map(lambda t: t[:, :, 0], (q, k, v))

            S = einsum('... i d, ... j d -> ... i j', q, k)

            if mask_A:
                device = q.device
                n = S.shape[-1]
                mask_value = -torch.finfo(S.dtype).max
                mask = torch.ones((n, n), device = device).triu(1).bool()
                mask = rearrange(mask, 'i j -> () () i j')
                S = S.masked_fill(mask, mask_value)

            A = S.exp()
            y = einsum('... i j, ... j d -> ... i d', A, v)

            A = A.sum(dim = -1)

            if remove_right_off_diagonals:
                y = rearrange(y, 'b n z d -> b n () z d')
                y = F.pad(y, (0, 0, 0, 0, 1, 0), value = 0.)
                y = rearrange(y, 'b n r z d -> b (n r) z d')

                A = rearrange(A, 'b n z -> b n () z')
                A = F.pad(A, (0, 0, 1, 0), value = 0.)
                A = rearrange(A, 'b n r z -> b (n r) z')

            y = rearrange(y, 'b ... n d -> b (... n) d')
            A = rearrange(A, 'b ... i -> b (... i)')
            return y, A

        # calculate Ys, as in the paper

        to_blocks = lambda t: rearrange(t, 'b (n z) d -> b n z d', z = bsz)
        Ys = []

        for ind, (q, k, v) in enumerate(coarsened_qkvs):
            q, k, v = map(to_blocks, (q, k, v))

            k = rearrange(k, 'b (n r) z d -> b n r z d', r = 2)
            k = torch.flip(k, dims = (2,))                          # so we pay attention to the off-diagonal blocks in the attention matrix
            k = rearrange(k, 'b n r z d -> b (n r) z d')

            v = rearrange(v, 'b (n r) z d -> b n r z d', r = 2)
            v = torch.flip(v, dims = (2,))
            v = rearrange(v, 'b n r z d -> b (n r) z d')

            coarsened_Y = calculate_Y_and_A(q, k, v, remove_right_off_diagonals = causal)
            Ys.append(coarsened_Y)

        top_level_Y = calculate_Y_and_A(*map(to_blocks, top_level_qkvs), mask_A = causal)
        Ys.append(top_level_Y)

        # interpolate

        Y = 0
        A = 0

        for Y_level, A_level in Ys:
            if torch.is_tensor(Y):
                Y = repeat(Y, 'b n d -> b (n r) d', r = 2)

            if torch.is_tensor(A):
                A = repeat(A, 'b n -> b (n r)', r = 2)

            Y = Y_level + Y
            A = A_level + A

        out = Y / rearrange(A + eps, 'b n -> b n ()')

        # merge heads

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # combine out

        return self.to_out(out)

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
        block_size = 128      # this is the Nr in the paper - Nb = (max_seq_len / tokens_per_block)
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.max_seq_len = max_seq_len

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, HAttention1D(dim, causal = causal, dim_head = dim_head, heads = heads, block_size = block_size)),
                PreNorm(dim, FeedForward(dim, mult = ff_mult))
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x, mask = None):
        b, n, device = *x.shape, x.device
        x = self.token_emb(x)

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        x = x + rearrange(pos_emb, 'n d -> () n d')

        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x

        return self.to_logits(x)
