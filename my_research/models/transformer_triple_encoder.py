'''
My updated transformer architecture, edit from PyTorch official Transformer code
see exp_encoder(), exp_decoder(), exp_transformer() for more details
'''

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from matplotlib import pyplot as plt

# for func input format
from torch import Tensor
from typing import Optional, Union, Callable, Any, Dict, List

CHECK_W = False
_variances = []  # ('name', var(x), var(_out), ratio)
def _append_variance(*row):
    tmp = [
        row[0],
        row[1].cpu().item(),
        row[2].cpu().item(),
    ]
    global _variances
    _variances += [tmp]

CHECK_VAR = False

def show_attn(attn, mode_cross2d=False):
    # attn.shape: (B, J, J)
    # type == numpy arr
    B = attn.shape[0]
    H = min(2, B)

    if mode_cross2d == False:
        for i in range(H): # first 5 batches
            ax = plt.subplot(H, 1, i+1)
            ax.imshow(attn[i])
            ax.set_title(f'image: {i}')
            # ax.axis('off')
    else:
        # (B J 256)
        pixels = attn.shape[2]
        width = int(pixels ** (1/2))
        for i in range(H):
            ax = plt.subplot(H, 1, i+1)
            ax.imshow(attn[0, i].reshape(width, width))
            ax.set_title(f'joint: {i}')

    plt.show()


class MyEncoderLayer(nn.TransformerEncoderLayer):
    '''
    ? Param
    ! Add x_embedding: None or sum(all embeddings)
        joint embedding

    ? Function
    ! Edit forward(), _sa_block()
    ! Add with_pos_embed() from DETR, transformer.py line:209
        MAKE sure embedding is same size to x
    '''
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None,
                 NormTwice=False,
                 AddCrossAttn2ImageFeat=False,
                 ):
        super().__init__(d_model, nhead, dim_feedforward, dropout,
                 activation,
                 layer_norm_eps, batch_first, norm_first,
                 device, dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}
        # self.norm1 = nn.LayerNorm((21, d_model), eps=layer_norm_eps, **factory_kwargs)
        # self.norm2 = nn.LayerNorm((21, d_model), eps=layer_norm_eps, **factory_kwargs)
        if NormTwice:
            assert self.norm_first == False, 'norm_first should be False, while applying NormTwice'
            self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)  # _sa out
            self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)  # _ff out
        else:
            def empty_func(x):
                return x
            self.norm3 = self.norm4 = empty_func

        # if ReWeighting:
        #     self.x_weight_bias = nn.Parameter(torch.tensor([1, 0], dtype=torch.float32))
        #     self.embed_weight_bias = nn.Parameter(torch.tensor([1, 0], dtype=torch.float32))

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        x_embedding: Optional[Tensor] = None,  # ! NEW joint embedding
        WeightedPaddingMask=False,           # Apply key <- key * src_key_passing_mask; src_mask~(0, 1)
        memory2: Optional[Tensor] = None, memory2_embedding: Optional[Tensor] = None,
        ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, x_embedding, WeightedPaddingMask)
            x = x + self._ff_block(self.norm2(x))
        else:
            _sa_out = self.norm3(self._sa_block(x, src_mask, src_key_padding_mask, x_embedding, WeightedPaddingMask))  # check _sa.var()
            if CHECK_VAR: _append_variance('E_sa_out', x.detach().var(), _sa_out.detach().var())
            x = self.norm1(x + _sa_out)

            _ff_out = self.norm4(self._ff_block(x))  # check _ff.var()
            if CHECK_VAR: _append_variance('E_ff_out', x.detach().var(), _ff_out.detach().var())
            x = self.norm2(x + _ff_out)

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                  x_embedding: Optional[Tensor],  # ! NEW joint embedding
                  WeightedPaddingMask,
                  ) -> Tensor:
        q = k = self.with_pos_embed(x, x_embedding)

        if WeightedPaddingMask:
            # Apply: key <- key * confidence_score
            k = k * rearrange(key_padding_mask, 'B J -> B J ()')
            key_padding_mask = None

        x, w = self.self_attn(query=q, key=k, value=x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=CHECK_W)  # True for attention map
        if CHECK_W:
            show_attn(w.detach().cpu().numpy())
        return self.dropout1(x)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if pos != None and tensor.shape != pos.shape:
            raise RuntimeError(f'positional embedding shape not matched, tensor: {tensor.shape}, pos: {pos.shape}')
        return tensor if pos is None else tensor + pos

class MyEncoder(nn.TransformerEncoder):
    '''
    ? Param
    ! Add x_embedding: None or sum(all embeddings)

    ? Function
    ! Edit forward()
    '''
    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
        x_embedding: Optional[Tensor] = None,  # ! NEW joint embedding
        WeightedPaddingMask=False,
        ReturnEachLayerOutput=False,
        memory2: Optional[Tensor] = None, memory2_embedding: Optional[Tensor] = None,
        ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        out_each_layer = []

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask,
                         x_embedding=x_embedding, WeightedPaddingMask=WeightedPaddingMask,
                         memory2=memory2, memory2_embedding=memory2_embedding,
                         )
            if ReturnEachLayerOutput:
                out_each_layer += [output]

        if self.norm is not None:
            output = self.norm(output)
            if ReturnEachLayerOutput:
                for i in range(self.num_layers):
                    out_each_layer[i] = self.norm(out_each_layer[i])

        return output, out_each_layer

class MyTransformer_Triple(nn.Transformer):
    '''
    implement my forwarding logics
    ! turns joint_embedding and serial_embedding to
    !   x_embedding, mem_embedding, tgt_embedding
        joint_embedding.shape : (J, D), J: joint_counts( may +1 for global feature)
        serial_embedding.shape: (S, D), S: max_serial counts, S == F(frame counts)
    '''
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None,
                 matrix: Optional[nn.Parameter]=None,
                 encoder_temporal=None, encoder_spatial=None,
                 ) -> None:
        super().__init__(d_model, nhead, num_decoder_layers, num_decoder_layers,
                         dim_feedforward, dropout, activation,
                         custom_encoder, custom_decoder,
                         layer_norm_eps, batch_first, norm_first,
                         device, dtype,
                         )
        assert encoder_temporal is not None and encoder_spatial is not None
        self.encoder_temporal = encoder_temporal
        self.encoder_spatial = encoder_spatial

        self._reset_parameters()
        self.matrix = matrix  # 49 to 21 matrix in SequencialReg2DDecode3D


    def forward(self, src,
                joint_embedding: Optional[Tensor] = None,  # ! NEW 3 embedding, (J, C)
                verts_embedding: Optional[Tensor] = None,  # (V, C)
                serial_embedding: Optional[Tensor] = None,  # (F, C)
                JointConfMask: Dict[str, Any] = None,
                ):
        if src.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")
        B, F, J, D = src.shape  # BatchSize, FrameCounts, JointCounts, featureDim

        serial_embedding = serial_embedding[:F, :]  # only previous F embeddings is used

        # Expand ALL embeddings to (B, F, J, D)
        joint_embedding = repeat(joint_embedding, 'J D -> B F J D', B=B, F=F)
        verts_embedding = repeat(verts_embedding, 'V D -> B F V D', B=B, F=F)
        serial_embedding = repeat(serial_embedding, 'F D -> B F J D', B=B, J=J)  # accept J or V

        encoder_out = []
        # Enc *3
        src = self.forward_encoder1(src, joint_embedding, JointConfMask)
        encoder_out += [src]
        src = self.forward_encoder_temporal(src, serial_embedding, JointConfMask)
        encoder_out += [src]
        src = self.forward_encoder_spatial(src, joint_embedding, verts_embedding)
        encoder_out += [src[:, :, :J]]

        if CHECK_VAR:
            import pandas as pd
            pd.set_option('display.max_rows', None)
            df = pd.DataFrame(_variances, columns=['name', 'x', '_out'])
            df['_out / x'] = df['_out'] / df['x']
            print(df)

        to_return = [src, encoder_out]
        # main_out, (Opt)encoder out

        return to_return  # return BFJD or BFVD or BF(J+V)D

    def forward_encoder1(self, src, joint_embedding, JointConfMask):
        B, F, J, D = src.shape
        x_embedding = joint_embedding

        # conf mask
        SA_conf_joint_mask = JointConfMask['joint conf']  # (B F J)
        WeightedPaddingMask = True

        # Reshape & Forward
        x_embedding = rearrange(x_embedding, 'B F J D -> (B F) J D')
        src = rearrange(src, 'B F J D -> (B F) J D')  # B, Seq, Dim
        SA_conf_joint_mask = rearrange(SA_conf_joint_mask, 'B F J -> (B F) J')

        memory_BFJD, _ = self.encoder(src,
                            mask=None,
                            src_key_padding_mask=SA_conf_joint_mask,
                            x_embedding=x_embedding,
                            WeightedPaddingMask=WeightedPaddingMask,
                            )

        return rearrange(memory_BFJD, '(B F) J D -> B F J D', B=B)

    def forward_encoder_temporal(self, src, serial_embedding, JointConfMask):
        B, F, J, D = src.shape
        x_embedding = serial_embedding

        # conf mask
        SA_conf_joint_mask = JointConfMask['joint conf']  # (B F J)
        WeightedPaddingMask = True

        # Reshape & Forward
        x_embedding = rearrange(x_embedding, 'B F J D -> (B J) F D')
        src = rearrange(src, 'B F J D -> (B J) F D')  # B, Seq, Dim
        SA_conf_joint_mask = rearrange(SA_conf_joint_mask, 'B F J -> (B J) F')  # turns to temporal conf

        memory_BFJD, _ = self.encoder_temporal(src,
                            mask=None,
                            src_key_padding_mask=SA_conf_joint_mask,
                            x_embedding=x_embedding,
                            WeightedPaddingMask=WeightedPaddingMask,
                            )

        return rearrange(memory_BFJD, '(B J) F D -> B F J D', B=B)

    def forward_encoder_spatial(self, src, joint_embedding, verts_embedding):
        B, F, J, D = src.shape
        V = verts_embedding.shape[2]
        device = src.device
        x_embedding = torch.cat([joint_embedding, verts_embedding], dim=2)  # BFJD

        src = rearrange(src, 'B F J D -> (B F) J D')
        srcJV = torch.cat([src, torch.bmm(self.matrix.repeat(B * F, 1, 1), src)], dim=1)
        # srcJV = torch.zeros((B * F, J+V, D), device=device)
        # srcJV[:, :J] = src[:]  # [feature, zero]
        # srcJV[:, J:] = torch.bmm(self.matrix.repeat(B, 1, 1), src[:])

        # Reshape & Forward
        x_embedding = rearrange(x_embedding, 'B F J D -> (B F) J D')
        # src = rearrange(src, 'B F JV D -> (B F) JV D')  # B, Seq, Dim


        memory_BFJD, _ = self.encoder_spatial(srcJV,
                            mask=None,
                            src_key_padding_mask=None,
                            x_embedding=x_embedding,
                            )

        return rearrange(memory_BFJD, '(B F) JV D -> B F JV D', B=B)


    def combine_embed(self, embed_1: Tensor, embed_2: Tensor):
        # commented to accept sum(joint[BFJD], serial[BF1D]) and
        #                     sum(verts[BFVD], serial[BF1D])
        # if embed_1 != None and embed_2 != None and embed_1.shape != embed_2.shape:
        #     raise RuntimeError(f'positional embedding shape not matched, 1: {embed_1.shape}, 2: {embed_2.shape}')

        if embed_1 == None:
            return embed_2
        elif embed_2 == None:
            return embed_1
        else:
            return embed_1 + embed_2


# supplement of transformer, encoder, decoder
def standardize(x, additional_weight_bias = None):
    # return x  # ! short cut of standardize(x) -> identity(x)
    std = x.std()
    mean = x.mean()
    std_x = (x - mean) / (std + 1e-8)
    if additional_weight_bias is not None:
        return std_x * additional_weight_bias[0] + additional_weight_bias[1]
    else:
        return std_x

def test_correctness_of_new_transformer():
    device = torch.device('cuda:0')
    matrix = nn.Parameter(torch.ones([49, 21])*0.01, requires_grad=True)
    model = get_transformer_triple(d_model=256, nhead=1, num_encoder_layers=3, num_decoder_layers=0, matrix=matrix,
                                   norm_first=False, NormTwice=True)
    model.eval()

    B, F, J, D = 4, 8, 21, 256
    V = 49
    data = torch.randn((B, F, J, D))
    EJ = torch.randn((J, D))
    ET = torch.randn((F, D))
    EV = torch.randn((V, D))
    conf = torch.randn((B, F, J))

    with torch.no_grad():
        out = model(data,
            joint_embedding=EJ,
            verts_embedding=EV,
            serial_embedding=ET,
            JointConfMask = {  # mask on KEYs
                'enc self': 'weight',           # ['mask', 'weight', 'no']
                'dec cross': 'weight',
                'joint mask': None, # [padding_mask, None], (B F J)
                'joint conf': conf,         # [None,         conf], (B F J)
            },
            )

def get_transformer_triple(d_model, nhead, num_encoder_layers, num_decoder_layers,
                    layer_norm_eps=1e-5, norm_first=False,
                    NormTwice=False,
                    matrix: Optional[nn.Parameter]=None,
                    ) -> MyTransformer_Triple:
    encoder_layer = MyEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, norm_first=norm_first, NormTwice=NormTwice)
                                 # dim_feedforward=2048, dropout=0.1, layer_norm_eps=layer_norm_eps
    encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)  # final norm in encoder
    # encoder_norm = nn.LayerNorm((21, d_model), eps=layer_norm_eps)  # final norm in encoder
    encoder = MyEncoder(encoder_layer=encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm)

    encoder_layer = MyEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, norm_first=norm_first, NormTwice=NormTwice)
    encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)  # final norm in encoder
    encoder_temporal = MyEncoder(encoder_layer=encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm)

    encoder_layer = MyEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, norm_first=norm_first, NormTwice=NormTwice)
    encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)  # final norm in encoder
    encoder_spatial = MyEncoder(encoder_layer=encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm)

    transformer = MyTransformer_Triple(
        d_model=d_model, nhead=nhead,
        # num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
        # dim_feedforward=2048, dropout=0.1,
        custom_encoder=encoder, custom_decoder=nn.Module(),
        # layer_norm_eps=layer_norm_eps,
        batch_first=True,
        matrix=matrix,
        encoder_temporal=encoder_temporal, encoder_spatial=encoder_spatial,
    )
    return transformer


if __name__ == '__main__':
    uv_reg = torch.randn((32, 22, 2))  # set joint counts to 22( + global feat )
    latent = torch.randn((32, 256, 4, 4))

    # exp_transformer()


    # backbone_feature = torch.randn((2, 8, ))