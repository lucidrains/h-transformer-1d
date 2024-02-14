from math import log2, ceil
from functools import wraps

import torch
from torch import nn, einsum, diagonal
import torch.nn.functional as F

from h_transformer_1d.reversible import ReversibleSequence, SequentialSequence
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding
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

def shift(t, amount, mask = None):
    if amount == 0:
        return t

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return F.pad(t, (0, 0, amount, -amount), value = 0.)

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

# token shifting

class PreShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask = mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)

# hierarchical attention helper functions

def cast_for_op(cast_type, fn):
    @wraps(fn)
    def inner(t, *args, **kwargs):
        orig_type = t.dtype
        t = t.type(cast_type)
        out = fn(t, *args, **kwargs)
        out = out.type(orig_type)
        return out
    return inner

def flip_every_two(t):
    t = rearrange(t, 'b (n r) ... -> b n r ...', r = 2)
    t = torch.flip(t, dims = (2,))                          # so we pay attention to the off-diagonal blocks in the attention matrix
    t = rearrange(t, 'b n r ... -> b (n r) ...')
    return t

# attention

class HAttention1D(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 64,
        block_size = 16,
        pos_emb = None,
        eps = 1e-8,
        **kwargs
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.block_size = block_size
        inner_dim = heads * dim_head

        self.pos_emb = pos_emb
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None):
        b, n, h, device, bsz, eps = *x.shape[:2], self.heads, x.device, self.block_size, self.eps

        # pad sequence length to power of 2

        pad_to_len = 2 ** ceil(log2(n))
        padding = pad_to_len - n

        if mask is None:
            mask = torch.ones(x.shape[:2],dtype=torch.bool)

        if padding != 0:
            x = F.pad(x, (0, 0, 0, padding), value = 0.)
            mask = F.pad(mask, (0, padding), value = False)

        # derive queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split out heads, and also divide sequence into blocks

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        mask = repeat(mask, 'b n -> (b h) n', h = h)

        # scale

        q = q * self.scale

        # rotary pos emb

        if exists(self.pos_emb):
            freqs = self.pos_emb(torch.arange(pad_to_len, device = device))
            freqs = rearrange(freqs, 'n d -> () n d')
            q, k, v = map(lambda t: apply_rotary_emb(freqs, t), (q, k, v))

        # calculate number of levels until 2 x 2

        num_levels = int(log2(pad_to_len // bsz)) - 1
        assert num_levels >= -1, 'block size cannot be larger than padded matrix dimension'

        # coarsening

        qkvs = [(q, k, v, mask)]

        for level in range(num_levels):
            q, k, v = map(lambda t: rearrange(t, 'b (n r) d -> b n r d', r = 2), (q, k, v))

            mask = repeat(mask, 'b (n r) -> b n r', r = 2)

            # masked mean for queries and keys, but not values

            q = masked_aggregate(q, mask, dim = 2)
            k = masked_aggregate(k, mask, dim = 2)
            v = masked_aggregate(v, mask, dim = 2, average = False)

            mask = torch.any(mask, dim = 2)

            coarsened_qkvs = (q, k, v, mask)
            qkvs.append(coarsened_qkvs)

        # half-attention function

        def calculate_Y_and_A(q, k, v, mask = None):
            S = einsum('... i d, ... j d -> ... i j', q, k)

            mask_value = -torch.finfo(S.dtype).max
            S = S.masked_fill(~mask, mask_value)

            S = S - torch.max(S, dim = -1, keepdim = True).values
            A = S.exp()

            y = einsum('... i j, ... j d -> ... i d', A, v)

            A = A.sum(dim = -1)

            y = rearrange(y, 'b ... n d -> b (... n) d')
            A = rearrange(A, 'b ... i -> b (... i)')
            return y, A

        to_blocks = lambda t: rearrange(t, 'b (n z) ... -> b n z ...', z = bsz)

        # calculate Ys, as in the paper

        Ys = []

        for ind, (q, k, v, mask) in enumerate(reversed(qkvs)):
            is_last = ind == (len(qkvs) - 1)

            q, k, v = map(to_blocks, (q, k, v))

            # generate the mask for S

            S_mask = None
            mask = to_blocks(mask)
            q_mask = mask
            k_mask = cast_for_op(torch.int, flip_every_two)(mask) if not is_last else mask
            S_mask = rearrange(q_mask, '... n -> ... n ()') * rearrange(k_mask, '... n -> ... () n')

            # flip keys and values to capture the off-diagonals

            if not is_last:
                k, v = map(flip_every_two, (k, v))

            Y_level = calculate_Y_and_A(q, k, v, mask = S_mask)
            Ys.append(Y_level)

        # interpolate

        Y = 0
        A = 0

        for ind, (Y_level, A_level) in enumerate(Ys):
            is_last = ind == (len(Ys) - 1)

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

# causal attention

class CausalHAttention1D(nn.Module):
    def __init__(
        self,
        dim,
        *,
        max_seq_len,
        heads = 8,
        dim_head = 64,
        block_size = 16,
        eps = 1e-8,
        pos_emb = None
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.block_size = block_size
        inner_dim = heads * dim_head

        self.pos_emb = pos_emb

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        # derive mask

        num_levels = int(log2(max_seq_len // block_size)) - 1
        root_seq = torch.arange(max_seq_len)
        seqs = [root_seq]
        seq = root_seq

        for ind in range(num_levels):
            seq = rearrange(seq, '(n r) -> n r', r = 2)
            seq = seq.max(dim = -1).values
            expanded_mask_seq = repeat(seq, 'n -> (n r)', r = (2 ** (ind + 1)))
            seqs.append(expanded_mask_seq)

        seq_keys = torch.stack(seqs, dim = 0)
        mask = seq_keys > rearrange(root_seq, 'n -> () n')
        self.register_buffer('mask', mask)

    def forward(self, x, **kwargs):
        b, n, h, device, bsz, eps = *x.shape[:2], self.heads, x.device, self.block_size, self.eps

        # pad sequence length to power of 2

        pad_to_len = 2 ** ceil(log2(n))
        padding = pad_to_len - n

        if padding != 0:
            x = F.pad(x, (0, 0, 0, padding), value = 0.)

        # derive queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split out heads, and also divide sequence into blocks

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # rotary embedding

        if exists(self.pos_emb):
            freqs = self.pos_emb(torch.arange(pad_to_len, device = device))
            freqs = rearrange(freqs, 'n d -> () n d')
            q, k, v = map(lambda t: apply_rotary_emb(freqs, t), (q, k, v))

        # calculate number of levels until 2 x 2

        num_levels = int(log2(pad_to_len // bsz)) - 1

        # coarsening

        qkvs = [(q, k, v)]

        for level in range(num_levels):
            q, k, v = map(lambda t: rearrange(t, 'b (n r) d -> b n r d', r = 2), (q, k, v))

            # masked mean for queries and keys, but not values

            q = q.mean(dim = 2)
            k = k.mean(dim = 2)
            v = v.sum(dim = 2)

            coarsened_qkvs = (q, k, v)
            qkvs.append(coarsened_qkvs)

        # half-attention function

        def calculate_Y_and_A(q, k, v, mask_right_off_diagonals = False, causal_mask_diagonal = False):
            if mask_right_off_diagonals:
                q, k, v = map(lambda t: rearrange(t, 'b (n r) ... -> b n r ...', r = 2), (q, k, v))
                q, k, v = map(lambda t: t[:, :, 1], (q, k, v))

            S = einsum('... i d, ... j d -> ... i j', q, k)

            if causal_mask_diagonal:
                causal_mask = torch.ones(*S.shape[-2:], device = S.device).triu(1).bool()
                mask_value = -torch.finfo(S.dtype).max
                causal_mask = rearrange(causal_mask, 'i j -> () () i j')
                S = S.masked_fill(causal_mask, mask_value)

            S = S - torch.amax(S, dim = -1, keepdim = True)
            A = S.exp()

            y = einsum('... i j, ... j d -> ... i d', A, v)

            A = A.sum(dim = -1)

            if mask_right_off_diagonals:
                y, A = map(lambda t: rearrange(t, 'b n ... -> b n () ...'), (y, A))
                y = F.pad(y, (0, 0, 0, 0, 1, 0), value = 0.)
                A = F.pad(A, (0, 0, 1, 0), value = 0.)

            y = rearrange(y, 'b ... d -> b (...) d')
            A = rearrange(A, 'b ... -> b (...)')
            return y, A

        to_blocks = lambda t: rearrange(t, 'b (n z) ... -> b n z ...', z = bsz)

        # calculate Ys, as in the paper

        Ys = []

        for ind, (q, k, v) in enumerate(reversed(qkvs)):
            is_last = ind == (len(qkvs) - 1)

            q, k, v = map(to_blocks, (q, k, v))

            # flip keys and values to capture the off-diagonals

            if not is_last:
                k, v = map(flip_every_two, (k, v))

            Y_level = calculate_Y_and_A(q, k, v, mask_right_off_diagonals = not is_last, causal_mask_diagonal = is_last)
            Ys.append(Y_level)

        # interpolate

        def safe_cat(acc, el, dim = 0):
            if not exists(acc):
                return el
            return torch.cat((el, acc), dim = dim)

        Y = None
        A = None

        for Y_level, A_level in Ys:
            Y_level, A_level = map(lambda t: rearrange(t, '... -> () ...'), (Y_level, A_level))

            if torch.is_tensor(Y):
                Y = repeat(Y, '... n d -> ... (n r) d', r = 2)

            if torch.is_tensor(A):
                A = repeat(A, '... n -> ... (n r)', r = 2)

            Y = safe_cat(Y, Y_level)
            A = safe_cat(A, A_level)

        # create causal mask for Y and A

        causal_mask = self.mask[:(num_levels + 1), :pad_to_len]

        # mask and sum

        Y_causal_mask = rearrange(causal_mask, 'h n -> h () n ()')
        A_causal_mask = rearrange(causal_mask, 'h n -> h () n')

        Y = Y.masked_fill(Y_causal_mask, 0.)
        A = A.masked_fill(A_causal_mask, 0.)

        Y = Y.sum(dim = 0)
        A = A.sum(dim = 0)

        # normalize

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
        causal = False,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        block_size = 128,     # this is the Nr in the paper - Nb = (max_seq_len / tokens_per_block)
        pos_emb = None,
        reversible = False,
        shift_tokens = False
    ):
        super().__init__()
        assert (max_seq_len % block_size) == 0, 'maximum sequence length must be divisible by the block size'
        num_blocks = max_seq_len // block_size
        assert log2(max_seq_len // block_size).is_integer(), f'number of blocks {num_blocks} must be a power of 2'

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = RotaryEmbedding(dim = dim_head)
        self.max_seq_len = max_seq_len

        layers = nn.ModuleList([])

        attn_class = CausalHAttention1D if causal else HAttention1D
        attn_kwargs = dict(max_seq_len = max_seq_len) if causal else dict()

        shift_token_ranges = (0, 1) if shift_tokens else (-1, 0, 1)

        for ind in range(depth):
            attn = attn_class(dim, dim_head = dim_head, heads = heads, block_size = block_size, pos_emb = self.pos_emb, **attn_kwargs)
            ff = FeedForward(dim, mult = ff_mult)

            if shift_tokens:
                attn, ff = map(lambda t: PreShiftTokens(shift_token_ranges, t), (attn, ff))

            attn, ff = map(lambda t: PreNorm(dim, t), (attn, ff))
            layers.append(nn.ModuleList([attn ,ff]))

        execute_type = ReversibleSequence if reversible else SequentialSequence
        route_attn = ((True, False),) * depth
        attn_route_map = {'mask': route_attn}

        self.layers = execute_type(layers, args_route = {**attn_route_map})

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x, mask = None):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, 'sequence length must be less than the maximum sequence length'
        x = self.token_emb(x)
        x = self.layers(x, mask = mask)
        return self.to_logits(x)
