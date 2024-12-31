# Based on functions from Swin repo


def mlp_flops(tokens, dim, mlp_ratio):
    flops = 0
    # x = self.mlp(x)
    flops += 2 * tokens * dim * dim * mlp_ratio
    return flops


def self_attention_flops(tokens, dim, num_heads):
    flops = 0
    # qkv = self.qkv(x)
    flops += tokens * dim * 3 * dim
    # attn = (q @ k.transpose(-2, -1))
    flops += num_heads * tokens * (dim // num_heads) * tokens
    #  x = (attn @ v)
    flops += num_heads * tokens * tokens * (dim // num_heads)
    # x = self.proj(x)
    flops += tokens * dim * dim
    return flops


def cross_attention_flops(q_tokens, kv_tokens, dim, num_heads):
    flops = 0
    # q = self.q(x): (N, Tq, D) x (1, D, D) --> (N, Tq, D)
    flops += q_tokens * dim * dim
    # kv = self.kv(y): (N, Tkv, D) x (1, D, 2 * D) --> (N, Tkv, 2 * D)
    flops += kv_tokens * dim * 2 * dim
    # attn = (q @ k.transpose(-2, -1)): (N, Tq, D) x (N, D, Tkv) --> (N, Tq, Tkv)
    flops += num_heads * q_tokens * (dim // num_heads) * kv_tokens
    #  x = (attn @ v): (N, Tq, Tkv) x (N, Tkv, D) --> (N, Tq, D)
    flops += num_heads * q_tokens * kv_tokens * (dim // num_heads)
    # x = self.proj(x): (N, Tq, D) x (1, D, D) --> (N, Tq, D)
    flops += q_tokens * dim * dim
    return flops


def modulation_flops(dim):
    flops = 0
    # gate1, shift1, scale1, gate2, shift2, scale2 = self.modulation(emb).chunk(6, dim=1)
    flops += 6 * dim * dim
    return flops


def norm_flops(tokens, dim):
    flops = tokens * dim
    return flops


def patchify_flops(H, W, C, dim, patch_size):
    flops = C * dim * (H // patch_size) * (W // patch_size) * patch_size * patch_size
    return flops


def decoder_flops(tokens, H, W, dim):
    flops = 0
    # norm
    flops += norm_flops(tokens, dim)
    # linear
    flops += dim * 6 * H * W
    return flops


def modblock_flops(tokens, dim, num_heads):
    # core block
    flops = block_flops(tokens, dim, num_heads)
    # modulation
    flops += modulation_flops(dim)
    return flops


def xblock_flops(tokens, dim, num_heads, kv_tokens=2):
    # core block
    flops = block_flops(tokens, dim, num_heads)
    # cross-attention:
    flops += cross_attention_flops(tokens, kv_tokens, dim, num_heads)
    return flops


def block_flops(tokens, dim, num_heads):
    flops = 0
    # norm1
    flops += norm_flops(tokens, dim)
    # attention
    flops += self_attention_flops(tokens, dim, num_heads)
    # norm2
    flops += norm_flops(tokens, dim)
    # mlp
    flops += mlp_flops(tokens, dim, 4)
    return flops


def t_encoder_flops(dim, use_gembed: bool = False):
    # two-layer MLP: (256, dim) --> (dim, dim)
    input_dim = 512 if use_gembed else 256
    return input_dim * dim + dim * dim


def dit_flops(dim, num_blocks, num_heads, patch_size, H, W, C, extra_tokens=0, block="mod", gembed=False):
    block_fn = dict(basic=block_flops, mod=modblock_flops, x=xblock_flops)
    tokens = H * W // (patch_size * patch_size)
    tokens += extra_tokens
    flops = 0
    flops += patchify_flops(H, W, C, dim, patch_size)
    flops += t_encoder_flops(dim, gembed)
    flops += num_blocks * block_fn[block](tokens, dim, num_heads)
    flops += decoder_flops(tokens, H, W, dim)
    if block == "mod":
        # final ln mod:
        flops += 2 * dim * dim
    return flops

def get_dit_flops(model_size, patch_size, use_gembed=True):
    configs = {
        "H":  dict(dim=1280, num_blocks=32, num_heads=16),
        "XL": dict(dim=1152, num_blocks=28, num_heads=16),
        "L":  dict(dim=1024, num_blocks=24, num_heads=16),
        "B":  dict(dim=768,  num_blocks=12, num_heads=12),
        "S":  dict(dim=384,  num_blocks=12, num_heads=6),
    }
    conditioning = {
        "mod":           dict(block="mod",   extra_tokens=0),
        # "x":             dict(block="x",     extra_tokens=0),
        # "in-context":    dict(block="basic", extra_tokens=2),
        # "unconditional": dict(block="basic", extra_tokens=0),  # includes cls token
    }
    H, W, C = [32 , 32, 4]
    flops = dit_flops(
        dim=configs[model_size]["dim"],
        num_blocks=configs[model_size]["num_blocks"],
        num_heads=configs[model_size]["num_heads"],
        patch_size=patch_size,
        H=H,
        W=W,
        C=C,
        extra_tokens=0,
        block=conditioning["mod"]["block"],
        gembed=use_gembed
    )
    return flops