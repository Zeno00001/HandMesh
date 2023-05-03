import torch
import numpy as np
from einops import rearrange, repeat

def uv_encoding(uv, feature_len, loooo=10000):
    '''
    shape     : (B, J, 2), -1 ~1
    encoding  : (B, J, feature_len * 2)

    feature_len should be {d_model /2}
    '''

    dim_t = torch.arange(feature_len, dtype=torch.float32, device=uv.device)
    dim_t = loooo ** (2 * (dim_t // 2) / feature_len)

    uv = (uv+1) / 2  # not affect outside uv
    uv = uv * 2*np.pi  # from (0, 1) to (0, 2*pi)

    pos_x = uv[:, :, 0:1] / dim_t  # (B, J, 1) -> (B, J, feature_len)
    pos_y = uv[:, :, 1:2] / dim_t

    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    # stack: [s s s s] [c c c c] -> (4, 2), [[s c] [s c] [s c] [s c]]
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    # (B, J, feature_len)
    pos = torch.cat((pos_y, pos_x), dim=2)
    # pos = rearrange(pos, 'B J C -> B C J')

    return pos


def t_encoding(frame_len, feature_len, device, loooo=10000):
    '''
    shape     : (1), frame_len
    encoding  : (F, feature_len)

    ~~feature_len should be {d_model /2}~~
    '''

    dim_t = torch.arange(feature_len, dtype=torch.float32, device=device)
    dim_t = loooo ** (2 * (dim_t // 2) / feature_len)

    t = torch.arange(frame_len, dtype=torch.float32, device=device).unsqueeze(1)
    t = t / (frame_len - 1)
    t = t * 2*np.pi # from (0, 1) to (0, 2*pi)

    pos_t = t[:, 0:1] / dim_t  # (F, 1) -> (F, feature_len)
    # pos_y = uv[:, 1:2] / dim_t

    pos_t = torch.stack((pos_t[:, 0::2].sin(), pos_t[:, 1::2].cos()), dim=2).flatten(1)
    # stack: [s s s s] [c c c c] -> (4, 2), [[s c] [s c] [s c] [s c]]
    return pos_t


def image_uv_encoding(width, feature_len):
    '''
    width = image feature.H or W
    return = encoding = (H, W, feature_len)

    Note that: Accept H==W case only
    '''
    y_s = repeat(torch.arange(width), 'H -> H W', W=width)
    x_s = repeat(torch.arange(width), 'W -> H W', H=width)
    uv_s = rearrange([x_s, y_s], 'C H W -> H W C').to(torch.float32)  # (H W 2)
    uv_s = uv_s / width + 0.5 / width
    uv_s = uv_s * 2 - 1
    # print(uv_s[:, :, 0])
    # print(uv_s[:, :, 1])  # (:, :, [x,y] )

    out = uv_encoding(uv_s, feature_len=feature_len)  # treat H, W as B, J
    return out  # (H, W, 256)


def zero_pad(tensor, target_channel, pad_behind):
    '''
    shape: (X1, X2, ..., C_)
    padding the channel C_ to {target_channel}, with zero
    pad_behind = bool

    return (X1, X2, ..., target_channel)
    '''
    shape = list(tensor.shape)
    shape[-1] = target_channel - shape[-1]
    zeros = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    if pad_behind:
        return torch.cat([tensor, zeros], dim=-1)
    else:
        return torch.cat([zeros, tensor], dim=-1)


def _example_padding():
    '''
        show example to merge 2D positional encoding and temporal encoding
        ( exp result is bad )
        2D pos  : [ ... ... ... ... ... pos pos pos ]
        temporal: [ temp temp temp temp ... ... ... ]
    '''
    B, J = 32, 21
    device = torch.device('cuda:0')
    EncodingChannelSplit = [156, 100]

    uv = torch.randn((B, J, 2))
    uv = torch.clamp(uv, -1, 1)
    uv_embed = uv_encoding(uv, feature_len=EncodingChannelSplit[0] // 2)
    uv_embed = zero_pad(uv_embed, 256, pad_behind=True)

    frame_len = 8
    serial_embed = t_encoding(frame_len, feature_len=EncodingChannelSplit[1], device=device)
    serial_embed = zero_pad(serial_embed, 256, pad_behind=False)

if __name__ == '__main__':
    device = torch.device('cuda:0')
    data = torch.arange(8, dtype=torch.float32, device=device).reshape((2, 2, 2))
    res = zero_pad(data, 5, pad_behind=True)
    print(res)

    #? 1D temporal
    # temporal = t_encoding(8, 256, device=device)
    # temporal_t = rearrange(temporal, 'J C -> C J')
    # res = torch.matmul(temporal, temporal_t)
    # print(res)

    #? 2D positional
    # pos = torch.arange(8, dtype=torch.float32, device=device)
    # pos = (pos - 4) / 4
    # pos = rearrange([pos, torch.zeros((8), dtype=torch.float32, device=device)], 'B T -> () T B')
    # pos_enc = uv_encoding(pos, 128)[0]  # -> (B, J, C)
    # res = torch.matmul(pos_enc, rearrange(pos_enc, 'J C -> C J'))
    # print(res)
