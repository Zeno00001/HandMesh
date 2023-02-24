import torch
import numpy as np
from einops import rearrange, repeat

def uv_encoding(uv, feature_len, loooo=10000):
    '''
    shape     : (B, J, 2), -1 ~1
    encoding  : (B, J, feature_len)

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
