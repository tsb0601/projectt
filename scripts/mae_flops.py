# Based on functions from Swin repo


from email import encoders
from json import decoder


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


def additional_decoder_flops(tokens, dim1, dim2, H, W, C):
    # one input linear, one output linear and one output norm
    # dim1 --> dim2
    flops = 0
    flops += dim1 * dim2 + dim2 # with bias
    # norm
    flops += norm_flops(tokens, dim2)
    # linear
    flops += dim2 * H * W * C + H * W * C # with bias
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


def mae_flops(dim, num_blocks, num_heads, patch_size, H, W, C, decoder_blocks, decoder_dim, decoder_num_heads, extra_tokens=0, block="mod"):
    block_fn = dict(basic=block_flops, mod=modblock_flops, x=xblock_flops)
    tokens = H * W // (patch_size * patch_size)
    tokens += extra_tokens
    # for encoders
    encoder_flops = 0
    encoder_flops += patchify_flops(H, W, C, dim, patch_size)
    encoder_flops += num_blocks * block_fn[block](tokens, dim, num_heads)
    encoder_flops += norm_flops(tokens, dim) # final layer norm
    
    # for decoder
    decoder_flops = 0
    decoder_flops += additional_decoder_flops(tokens, dim, decoder_dim, H, W, C)
    decoder_flops += decoder_blocks * block_fn[block](tokens, decoder_dim, decoder_num_heads)
    flops = encoder_flops + decoder_flops
    return flops, encoder_flops, decoder_flops


if __name__ == "__main__":
    configs = {
        "H":  dict(dim=1280, num_blocks=32, num_heads=16),
        "L":  dict(dim=1024, num_blocks=24, num_heads=16),
        "B":  dict(dim=768,  num_blocks=12, num_heads=12),
        "S":  dict(dim=384,  num_blocks=12, num_heads=6),
    }
    decoder_configs = {
        "H":  dict(decoder_dim=1280, decoder_blocks=32, decoder_num_heads=16),
        "L":  dict(decoder_dim=1024, decoder_blocks=24, decoder_num_heads=16),
        "B":  dict(decoder_dim=768,  decoder_blocks=12, decoder_num_heads=12),
        "D":  dict(decoder_dim=512,  decoder_blocks=12, decoder_num_heads=6),
    }
    conditioning = {
        #"mod":           dict(block="mod",   extra_tokens=0),
        # "x":             dict(block="x",     extra_tokens=0),
        "in-context":    dict(block="basic", extra_tokens=1),
        # "unconditional": dict(block="basic", extra_tokens=0),  # includes cls token
    }
    use_gembs = [False, True]
    patch_sizes = [16]
    universal_kwargs = dict(H=224, W=224, C=3)
    for name, config in configs.items():
        for dec_name, decoder_config in decoder_configs.items():
            print(f"================================")
            for cond_name, cond_config in conditioning.items():
                for patch_size in patch_sizes:
                    flops, enc_flops, dec_flops = mae_flops(patch_size=patch_size, **config, **cond_config, **universal_kwargs, **decoder_config)
                    print(f"MAE-{name}-{dec_name} ({patch_size}x{patch_size}), {cond_name}: {flops / 1e9:.4f}G, Encoder: {enc_flops / 1e9:.4f}G, Decoder: {dec_flops / 1e9:.4f}G")
    print(f"================================")