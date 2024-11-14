class StatCounter:

    def __init__(self, flops=0, params=0):
        self.flops = flops
        self.params = params

    def __add__(self, other):
        self.flops += other.flops
        self.params += other.params
        return self
    
    def __str__(self):
        return f'GFLOPs: {self.flops / 1e9:.2f}, Params: {self.params / 1e6:.2f} M'


def group_norm_flops(H, W, C, groups=32):
    flops = H * W * C
    params = 2 * C
    return StatCounter(flops, params)


def attnblock_flops(H, W, C, num_heads=4):
    flops = StatCounter()
    tokens = H * W
    # h = nn.GroupNorm(epsilon=1e-5)(x)
    flops += group_norm_flops(H, W, C)
    # qkv = conv_nd(1, self.channels, self.channels * 3, 1, padding=0)(h)
    flops += conv1d_flops(tokens, 1, C, C * 3)
    # h = QKVAttentionLegacy(self.num_heads)(qkv)
    flops += attn_flops(tokens, C, num_heads)  # no factor of 3 here because it's distributed across QKV
    # h = conv_nd(1, self.channels, self.channels, 1, padding=0, zero_init=True)(h)
    flops += conv1d_flops(tokens, 1, C, C)
    return flops


def attn_flops(tokens, dim, num_heads):
    flops = 0
    # weight = jnp.einsum("bct,bcs->bts", q * scale, k * scale, precision=ADM_PRECISION)
    flops += num_heads * tokens * (dim // num_heads) * tokens
    # a = jnp.einsum("bts,bcs->bct", weight, v, precision=ADM_PRECISION)
    flops += num_heads * tokens * (dim // num_heads) * tokens
    return StatCounter(flops=flops, params=0)


def upsample_conv_flops(H, W, Cin, Cout):
    flops = conv2d_flops(H * 2, W * 2, 3, 3, Cin, Cout, stride=1, padding=1)
    return flops


def downsample_conv_flops(H, W, Cin, Cout):
    flops = conv2d_flops(H // 2, W // 2, 3, 3, Cin, Cout, stride=2, padding=1)
    return flops


def resblock_flops(H, W, C, emb_dim, Cout=None, up=False, down=False, use_scale_shift=True):
    flops = StatCounter()
    Cout = Cout or C
    # h = nn.GroupNorm(epsilon=1e-5)(x)
    flops += group_norm_flops(H, W, C)
    assert not (up and down)
    # if self.up or self.down:
    #     interpolation_fn = Upsample if self.up else Downsample
    #     h = interpolation_fn(self.channels, False)(h)
    #     x = interpolation_fn(self.channels, False)(x)
    if up:
        H *= 2
        W *= 2
    elif down:
        H //= 2
        W //= 2
    # h = conv_nd(self.dims, self.channels, out_channels, 3, padding=1)(h)
    flops += conv2d_flops(H, W, 3, 3, C, Cout, padding=1)

    # emb = linear(self.emb_channels, 2 * out_channels)(emb)
    if use_scale_shift:
        flops += StatCounter(flops=emb_dim * Cout * 2, params=emb_dim * Cout * 2)
    else:
        flops += StatCounter(flops=emb_dim * Cout, params=emb_dim * Cout)
    # h = nn.GroupNorm(epsilon=1e-5)(h)
    flops += group_norm_flops(H, W, Cout)
    # h = modulation(h, shift, scale)
    # h = conv_nd(self.dims, out_channels, out_channels, 3, padding=1, zero_init=True)(h)
    flops += conv2d_flops(H, W, 3, 3, Cout, Cout, padding=1)        
    # if out_channels == self.channels:
    #     x = x
    # else:
    #     x = conv_nd(self.dims, self.channels, out_channels, 1, padding=0)(x)
    if Cout != C:
        flops += conv2d_flops(H, W, 1, 1, C, Cout)
    return flops


def conv1d_flops(L, k, Cin, Cout, stride=1, padding=0):
    L_locations = (L + 2 * padding - k) // stride + 1
    flops = L_locations * Cin * Cout * k
    params = k * Cin * Cout
    return StatCounter(flops=flops, params=params)


def conv2d_flops(H, W, kH, kW, Cin, Cout, stride=1, padding=0):
    H_locations = (H + 2 * padding - kH) // stride + 1
    W_locations = (W + 2 * padding - kW) // stride + 1
    flops = H_locations * W_locations * Cin * Cout * kH * kW
    params =  Cin * Cout * kH * kW
    return StatCounter(flops=flops, params=params)


def adm_flops(
        H, 
        W, 
        Cin, 
        Cout, 
        Cmodel=256, 
        num_res_blocks=2, 
        attention_resolutions=(32, 16, 8),
        channel_mult=(1, 1, 2, 3, 4),
        num_heads=4,
        num_classes=1000
    ):
    attention_resolutions = [H // ai for ai in attention_resolutions]
    flops = StatCounter()
    time_embed_dim = Cmodel * 4
    # nn.Embedding table for y:
    flops += StatCounter(flops=0, params=num_classes * time_embed_dim)

    # emb = timestep_embedding(t, self.model_channels)
    # emb = linear(self.model_channels, time_embed_dim)(emb)
    flops += StatCounter(flops=time_embed_dim * Cmodel, params=time_embed_dim * Cmodel)
    # emb = nn.swish(emb)
    # emb = linear(time_embed_dim, time_embed_dim)(emb)
    flops += StatCounter(flops=time_embed_dim * time_embed_dim, params=time_embed_dim * time_embed_dim)

    ch = input_ch = int(channel_mult[0] * Cmodel)
    _feature_size = ch
    input_block_chans = [ch]
    ds = 1

    # Input blocks:
    # h = conv_nd(self.dims, self.channels, ch, 3, padding=1)(x)
    flops += conv2d_flops(H, W, 3, 3, Cin, ch, padding=1)
    for level, mult in enumerate(channel_mult):
        for _ in range(num_res_blocks):
            # h = ResBlock(
            #         ch,
            #         time_embed_dim,
            #         out_channels=int(mult * self.model_channels),
            #         dims=self.dims,
            #     )(h, emb)
            flops += resblock_flops(H // ds, W // ds, ch, time_embed_dim, Cout=int(mult * Cmodel))
            ch = int(mult * Cmodel)
            if ds in attention_resolutions:
                # h = AttentionBlock(
                #         ch,
                #         num_heads=self.num_heads,
                #         num_head_channels=self.num_head_channels,
                #     )(h)
                flops += attnblock_flops(H // ds, W // ds, ch, num_heads=num_heads)
            # hs.append(h)
            _feature_size += ch
            input_block_chans.append(ch)
        if level != len(channel_mult) - 1:
            out_ch = ch
            # h = ResBlock(
            #         ch,
            #         time_embed_dim,
            #         out_channels=out_ch,
            #         dims=self.dims,
            #         down=True
            #     )(h, emb)
            flops += resblock_flops(H // ds, W // ds, ch, time_embed_dim, Cout=out_ch, down=True)
            ch = out_ch
            input_block_chans.append(ch)
            ds *= 2
            _feature_size += ch

    # Middle block:
    # h = ResBlock(ch, time_embed_dim, dims=self.dims)(h, emb)
    flops += resblock_flops(H // ds, W // ds, ch, time_embed_dim)
    # h = AttentionBlock(ch, num_heads=self.num_heads, num_head_channels=self.num_head_channels)(h)
    flops += attnblock_flops(H // ds, W // ds, ch, num_heads=num_heads)
    # h = ResBlock(ch, time_embed_dim, dims=self.dims)(h, emb)
    flops += resblock_flops(H // ds, W // ds, ch, time_embed_dim)
    
    _feature_size += ch

    # Output blocks:
    for level, mult in list(enumerate(channel_mult))[::-1]:
        for i in range(num_res_blocks + 1):
            ich = input_block_chans.pop()
            # h = ResBlock(
            #         ch + ich,
            #         time_embed_dim,
            #         out_channels=int(self.model_channels * mult),
            #         dims=self.dims
            #     )(cat(h, hs), emb)
            flops += resblock_flops(H // ds, W // ds, ch + ich, time_embed_dim, Cout=int(Cmodel * mult))
            ch = int(Cmodel * mult)
            if ds in attention_resolutions:
                # h = AttentionBlock(
                #         ch,
                #         num_heads=num_heads_upsample,
                #         num_head_channels=self.num_head_channels,
                #     )(h)
                flops += attnblock_flops(H // ds, W // ds, ch, num_heads=num_heads)
            if level and i == num_res_blocks:
                out_ch = ch
                # h = ResBlock(
                #         ch,
                #         time_embed_dim,
                #         out_channels=out_ch,
                #         dims=self.dims,
                #         up=True,
                #     )(h, emb)
                flops += resblock_flops(H // ds, W // ds, ch, time_embed_dim, Cout=out_ch, up=True)
                ds //= 2
            _feature_size += ch

    assert ds == 1, f"ds={ds}"
    # Decoder:
    # h = nn.GroupNorm(epsilon=1e-5)(h)
    flops += group_norm_flops(H // ds, W // ds, ch)
    # z_and_sigma = conv_nd(self.dims, input_ch, self.out_channels, 3, padding=1, zero_init=True)(h)
    flops += conv2d_flops(H // ds, W // ds, 3, 3, input_ch, Cout, padding=1)
    return flops


if __name__ == "__main__":
    print("=================ADM-128x128=================")
    print()
    print(adm_flops(128, 128, 3, 6, channel_mult=(1,1,2,3,4)))
    print()
    print("=================ADM-256x256=================")
    print()
    print(adm_flops(256, 256, 3, 6, channel_mult=(1,1,2,2,4,4)))
    print()
    print("=================ADM-512x512=================")
    print()
    print(adm_flops(512, 512, 3, 6, channel_mult=(0.5,1,1,2,2,4,4)))
    print()
    print("=================ADM-U-256x256 (64x64, 64x64-->256x256)=================")
    print()
    base_model = adm_flops(64, 64, 3, 6, channel_mult=(1,2,3,4), Cmodel=192, num_res_blocks=3)
    upsample_model = adm_flops(256, 256, 6, 6, channel_mult=(1,1,2,2,4,4), Cmodel=192)
    print(f"Base model only: {base_model}")
    print(f"Upsample model only: {upsample_model}")
    print(f"Full model: {base_model + upsample_model}")
    print()
    print("=================ADM-U-512x512 (128x128, 128x128-->512x512)=================")
    print()
    base_model = adm_flops(128, 128, 3, 6, channel_mult=(1,1,2,3,4))
    upsample_model = adm_flops(512, 512, 6, 6, channel_mult=(1,1,2,2,4,4), Cmodel=192)
    print(f"Base model only: {base_model}")
    print(f"Upsample model only: {upsample_model}")
    print(f"Full model: {base_model + upsample_model}")
    print()
    print("=============================================")