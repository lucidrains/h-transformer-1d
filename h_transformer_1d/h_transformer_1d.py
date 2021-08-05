from math import log2, ceil
import torch
from torch import nn, einsum, diagonal
import torch.nn.functional as F

from einops import rearrange, reduce, repeat

# helpers

def exists(val):
    return val is not None

def masked_aggregate(tensor, mask = None, dim = -1, average = True):
    if not exists(mask):
        fn = torch.sum if not average else torch.mean
        return fn(tensor, dim = dim)

    diff_len = len(tensor.shape) - len(mask.shape)
    mask = mask[(..., *((None,) * diff_len))]
    tensor = tensor.masked_fill(~mask, 0.)

    total_el = mask.sum(dim = dim)
    agg = tensor.sum(dim = dim)

    if average:
        agg = agg / total_el.clamp(min = 1.)

    agg.masked_fill_(total_el == 0, 0.)
    return agg

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
        heads = 8,
        dim_head = 64,
        block_size = 16,
        eps = 1e-8
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.block_size = block_size
        inner_dim = heads * dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None):
        b, n, h, device, bsz, eps = *x.shape[:2], self.heads, x.device, self.block_size, self.eps

        # pad sequence length to power of 2

        pad_to_len = 2 ** ceil(log2(n))
        padding = pad_to_len - n

        if padding != 0:
            x = F.pad(x, (0, 0, 0, padding), value = 0.)
            if exists(mask):
                mask = F.pad(mask, (0, padding), value = False)

        # derive queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split out heads, and also divide sequence into blocks

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # calculate number of levels until 2 x 2

        num_levels = int(log2(pad_to_len // bsz)) - 1

        # coarsening

        qkvs = [(q, k, v, mask)]

        for level in range(num_levels):
            q, k, v = map(lambda t: rearrange(t, 'b (n r) d -> b n r d', r = 2), (q, k, v))

            if exists(mask):
                mask = rearrange(mask, 'b (n r) -> b n r', r = 2)

            # masked mean for queries and keys, but not values

            q = masked_aggregate(q, mask, dim = 2)
            k = masked_aggregate(k, mask, dim = 2)
            v = masked_aggregate(v, mask, dim = 2, average = False)

            if exists(mask):
                mask = torch.any(mask, dim = 2)

            coarsened_qkvs = (q, k, v, mask)
            qkvs.append(coarsened_qkvs)

        # half-attention function

        def calculate_Y_and_A(q, k, v, mask = None):
            S = einsum('... i d, ... j d -> ... i j', q, k)

            if exists(mask):
                mask_value = -torch.finfo(S.dtype).max
                S = S.masked_fill(~mask, mask_value)

            S = S - torch.amax(S, dim = -1, keepdim = True)
            A = S.exp()

            y = einsum('... i j, ... j d -> ... i d', A, v)

            A = A.sum(dim = -1)

            y = rearrange(y, 'b ... n d -> b (... n) d')
            A = rearrange(A, 'b ... i -> b (... i)')
            return y, A

        def flip_every_two(t):
            t = rearrange(t, 'b (n r) ... -> b n r ...', r = 2)
            t = torch.flip(t, dims = (2,))                          # so we pay attention to the off-diagonal blocks in the attention matrix
            t = rearrange(t, 'b n r ... -> b (n r) ...')
            return t

        to_blocks = lambda t: rearrange(t, 'b (n z) ... -> b n z ...', z = bsz)

        # calculate Ys, as in the paper

        Ys = []

        for ind, (q, k, v, mask) in enumerate(reversed(qkvs)):
            is_last = ind == (len(qkvs) - 1)

            q, k, v = map(to_blocks, (q, k, v))

            # generate the mask for S

            S_mask = None
            if exists(mask):
                mask = to_blocks(mask)
                q_mask = mask
                k_mask = flip_every_two(mask) if not is_last else mask
                S_mask = rearrange(q_mask, '... n -> ... n ()') * rearrange(k_mask, '... n -> ... () n')

            # flip keys and values to capture the off-diagonals

            if not is_last:
                k, v = map(flip_every_two, (k, v))

            Y_level = calculate_Y_and_A(q, k, v, mask = S_mask)
            Ys.append(Y_level)

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

        return self.to_out(out[:, :n])

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
        ff_mult = 4,
        block_size = 128      # this is the Nr in the paper - Nb = (max_seq_len / tokens_per_block)
    ):
        super().__init__()
        assert (max_seq_len % block_size) == 0, 'maximum sequence length must be divisible by the block size'
        num_blocks = max_seq_len // block_size
        assert log2(max_seq_len // block_size).is_integer(), f'number of blocks {num_blocks} must be a power of 2'

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.max_seq_len = max_seq_len

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, HAttention1D(dim, dim_head = dim_head, heads = heads, block_size = block_size)),
                PreNorm(dim, FeedForward(dim, mult = ff_mult))
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x, mask = None):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, 'sequence length must be less than the maximum sequence length'

        x = self.token_emb(x)

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        x = x + rearrange(pos_emb, 'n d -> () n d')

        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x

        return self.to_logits(x)
