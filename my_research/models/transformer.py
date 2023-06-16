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

        self.AddCrossAttn2ImageFeat = AddCrossAttn2ImageFeat
        if AddCrossAttn2ImageFeat:
            # remain the naming rule in Decoder
            self.multihead_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                          **factory_kwargs)
            self.dropout2_2 = nn.Dropout(dropout)
            self.norm2_2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm5_2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)  # _mha2 out
            # _ca = norm5_2( mha_2( x, mem
            # x   = norm2_2( x+ _ca

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

            if self.AddCrossAttn2ImageFeat:
                _ca_out2 = self.norm5_2(self._mha_block2(
                    x, memory2, x_embedding=x_embedding, mem2_embedding=memory2_embedding))
                if CHECK_VAR: _append_variance('E_ca_out2', x.detach().var(), _ca_out2.detach().var())
                x = self.norm2_2(x + _ca_out2)

            _ff_out = self.norm4(self._ff_block(x))  # check _ff.var()
            if CHECK_VAR: _append_variance('E_ff_out', x.detach().var(), _ff_out.detach().var())
            x = self.norm2(x + _ff_out)
            # x = x + self.norm3(self._sa_block(x, src_mask, src_key_padding_mask, x_embedding))
            # x = self.norm1(x)
            # x = x + self.norm4(self._ff_block(x))
            # x = self.norm2(x)

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

    # multihead attention block, to cross image-feature
    def _mha_block2(self, x: Tensor, mem2: Tensor,
                    x_embedding: Tensor, mem2_embedding: Tensor,
                    ) -> Tensor:
        q = self.with_pos_embed(x, x_embedding)
        k = self.with_pos_embed(mem2, mem2_embedding)

        x, w = self.multihead_attn_2(query=q, key=k, value=mem2,
                                # attn_mask=attn_mask,
                                # key_padding_mask=key_padding_mask,
                                need_weights=CHECK_W)  # True for attention map
        if CHECK_W:
            show_attn(w.detach().cpu().numpy(), mode_cross2d=True)
        return self.dropout2_2(x)

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

class MyDecoderLayer(nn.TransformerDecoderLayer):
    '''
    ? Param
    ! Add tgt_embedding, memory_embedding
        each embedding = [None + [joint_embed + [serial_embed ]]]

    ? Function
    ! Edit forward(), _sa_block(), _mha_block()
    ! Add with_pos_embed() from DETR, transformer.py line:209
        ! MAKE sure embedding is same size to x
    '''
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None,
                 NormTwice=False,
                 AddCrossAttn2ImageFeat=False,
                 ) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout,
                 activation,
                 layer_norm_eps, batch_first, norm_first,
                 device, dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}
        # self.norm1 = nn.LayerNorm((21, d_model), eps=layer_norm_eps, **factory_kwargs)
        # self.norm2 = nn.LayerNorm((21, d_model), eps=layer_norm_eps, **factory_kwargs)
        # self.norm3 = nn.LayerNorm((21, d_model), eps=layer_norm_eps, **factory_kwargs)
        if NormTwice:
            assert self.norm_first == False, 'norm_first should be False, while applying NormTwice'
            self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)  # _sa out
            self.norm5 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)  # _mha out
            self.norm6 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)  # _ff out
        else:
            def empty_func(x):
                return x
            self.norm4 = self.norm5 = self.norm6 = empty_func

        self.AddCrossAttn2ImageFeat = AddCrossAttn2ImageFeat
        if AddCrossAttn2ImageFeat:
            self.multihead_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                          **factory_kwargs)
            self.dropout2_2 = nn.Dropout(dropout)
            self.norm2_2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm5_2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)  # _mha2 out
            # _ca = norm5_2( mha_2( x, mem
            # x   = norm2_2( x+ _ca

        # if ReWeighting:
        #     self.weight_bias = nn.ParameterDict({
        #         e: nn.Parameter(torch.tensor([1, 0], dtype=torch.float32)) for e in ['sa-tgt', 'sa-tgt_embed',
        #                                                                             'ca-tgt', 'ca-tgt_embed', 'ca-mem', 'ca-mem_embed']
        #     })

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                tgt_embedding: Optional[Tensor] = None, memory_embedding: Optional[Tensor] = None,  # ! NEW 2 embedding
                WeightedMemPaddingMask=False,  # Apply key <- key * memory_key_padding_mask; mem_mask~(0, 1)
                memory2: Optional[Tensor] = None, memory2_embedding: Optional[Tensor] = None,
                ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, x_embedding=tgt_embedding)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask,
                                    x_embedding=tgt_embedding, mem_embedding=memory_embedding,
                                    WeightedMemPaddingMask=WeightedMemPaddingMask)
            x = x + self._ff_block(self.norm3(x))
        else:
            _sa_out = self.norm4(self._sa_block(x, tgt_mask, tgt_key_padding_mask, x_embedding=tgt_embedding))
            if CHECK_VAR: _append_variance('D_sa_out', x.detach().var(), _sa_out.detach().var())
            x = self.norm1(x + _sa_out)

            _ca_out = self.norm5(self._mha_block(x, memory, memory_mask, memory_key_padding_mask,
                                                 x_embedding=tgt_embedding, mem_embedding=memory_embedding,
                                                 WeightedMemPaddingMask=WeightedMemPaddingMask))
            if CHECK_VAR: _append_variance('D_ca_out', x.detach().var(), _ca_out.detach().var())
            x = self.norm2(x + _ca_out)

            if self.AddCrossAttn2ImageFeat:
                _ca_out2 = self.norm5_2(self._mha_block2(
                    x, memory2, x_embedding=tgt_embedding, mem2_embedding=memory2_embedding))
                if CHECK_VAR: _append_variance('D_ca_out2', x.detach().var(), _ca_out2.detach().var())
                x = self.norm2_2(x + _ca_out2)

            _ff_out = self.norm6(self._ff_block(x))
            if CHECK_VAR: _append_variance('D_ff_out', x.detach().var(), _ff_out.detach().var())
            x = self.norm3(x + _ff_out)
            # x = self.norm1(x + self.norm4(self._sa_block(x, tgt_mask, tgt_key_padding_mask, x_embedding=tgt_embedding)))
            # x = self.norm2(x + self.norm5(self._mha_block(x, memory, memory_mask, memory_key_padding_mask,
            #                                               x_embedding=tgt_embedding, mem_embedding=memory_embedding)))
            # x = self.norm3(x + self.norm6(self._ff_block(x)))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                  x_embedding: Optional[Tensor],  # ! NEW joint_query embedding
                  ) -> Tensor:
        q = k = self.with_pos_embed(x, x_embedding)

        x, w = self.self_attn(query=q, key=k, value=x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=CHECK_W)  # True for attention map
        if CHECK_W:
            show_attn(w.detach().cpu().numpy())
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                   x_embedding: Optional[Tensor], mem_embedding: Optional[Tensor],  # ! NEW 2 embeddings
                   WeightedMemPaddingMask,
                   ) -> Tensor:
        q = self.with_pos_embed(x, x_embedding)
        k = self.with_pos_embed(mem, mem_embedding)
        # v = mem
        if WeightedMemPaddingMask:
            # Apply: key <- key * confidence_score
            k = k * rearrange(key_padding_mask, 'B FJ -> B FJ ()')
            key_padding_mask = None

        x, w = self.multihead_attn(query=q, key=k, value=mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=CHECK_W)  # True for attention map
        if CHECK_W:
            show_attn(w.detach().cpu().numpy())
        return self.dropout2(x)

    # multihead attention block, to cross image-feature
    def _mha_block2(self, x: Tensor, mem2: Tensor,
                    x_embedding: Tensor, mem2_embedding: Tensor,
                    ) -> Tensor:
        q = self.with_pos_embed(x, x_embedding)
        k = self.with_pos_embed(mem2, mem2_embedding)

        x, w = self.multihead_attn_2(query=q, key=k, value=mem2,
                                # attn_mask=attn_mask,
                                # key_padding_mask=key_padding_mask,
                                need_weights=CHECK_W)  # True for attention map
        if CHECK_W:
            show_attn(w.detach().cpu().numpy(), mode_cross2d=True)
        return self.dropout2_2(x)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if pos != None and tensor.shape != pos.shape:
            raise RuntimeError(f'positional embedding shape not matched, tensor: {tensor.shape}, pos: {pos.shape}')
        return tensor if pos is None else tensor + pos

class MyDecoder(nn.TransformerDecoder):
    '''
    ? Param
    ! Add tgt_embedding, memory_embedding
        each embedding = [None + [joint_embed + [serial_embed ]]]

    ? Function
    ! Edit forward()
    '''
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_embedding: Optional[Tensor] = None, memory_embedding: Optional[Tensor] = None,  # ! NEW 2 embedding
                WeightedMemPaddingMask: bool = False,
                memory2: Optional[Tensor] = None, memory2_embedding: Optional[Tensor] = None,
                ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        # for mod in self.layers:
        for i, mod in enumerate(self.layers):
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         tgt_embedding=tgt_embedding,
                         memory_embedding=memory_embedding,
                         WeightedMemPaddingMask=WeightedMemPaddingMask,
                         memory2=memory2, memory2_embedding=memory2_embedding,
                         )

        if self.norm is not None:
            output = self.norm(output)

        return output

class MyTransformer(nn.Transformer):
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
                 Mode: Optional[str]='base',
                 DecoderForwardConfigs: Optional[dict]=None,
                 matrix: Optional[nn.Parameter]=None,
                 EncAddCrossAttn2ImageFeat: Optional[bool]=False,
                 DecAddCrossAttn2ImageFeat: Optional[bool]=False,
                 ) -> None:
        super().__init__(d_model, nhead, num_decoder_layers, num_decoder_layers,
                         dim_feedforward, dropout, activation,
                         custom_encoder, custom_decoder,
                         layer_norm_eps, batch_first, norm_first,
                         device, dtype,
                         )
        self.Mode = Mode
        assert DecoderForwardConfigs is not None, 'DecoderForwardConfigs should be initial in get_transformer()'
        self.DecoderForwardConfigs = DecoderForwardConfigs
        self.matrix = matrix  # 49 to 21 matrix in SequencialReg2DDecode3D
        self.EncCross2ImageFeat = EncAddCrossAttn2ImageFeat  # Additional CrossAttn Layer in Encoder
        self.DecCross2ImageFeat = DecAddCrossAttn2ImageFeat  # Additional CrossAttn Layer in Decoder

        # params
        # balancing encodings
        # if ReWeighting:
        #     encodings = ['enc-2D pos', 'enc-joint',              'enc-serial',
        #                  'dec-2D pos', 'dec-joint', 'dec-verts', 'dec-serial']
        #     self.encoding_w_b = nn.ParameterDict({
        #         e: nn.Parameter(torch.tensor([1, 0], dtype=torch.float32)) for e in encodings
        #     })
        # balancing token and encoding-> each encoder

    def forward_old(self, src: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,  # ! remove tgt param
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                joint_embedding: Optional[Tensor] = None,  # ! NEW 3 embedding, (J, C)
                serial_embedding: Optional[Tensor] = None,  # (F, C)
                positional_embedding: Optional[Tensor] = None  # (BF, J, C)
                ) -> Tensor:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, E)` for unbatched input, :math:`(S, N, E)` if `batch_first=False` or
              `(N, S, E)` if `batch_first=True`.
            - tgt: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(T)` for unbatched input otherwise :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        # ! tgt is forwarded from zeros()
        # is_batched = src.dim() == 3
        # if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
        #     raise RuntimeError("the batch number of src and tgt must be equal")
        # elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
        #     raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        B, F, J, D = src.shape  # BatchSize, FrameCounts, JointCounts, featureDim
        serial_embedding = serial_embedding[:F, :]  # only previous F embeddings is used

        # duplicate with self.combine_embed()
        def sum_embedding(x_embed, add_embed):
            ''' return x_embed + add_embed '''
            if x_embed == None:
                if add_embed == None:
                    return None
                else:
                    return add_embed
            else:
                if add_embed == None:
                    return x_embed
                else:
                    assert x_embed.shape == add_embed.shape, f'shape not matched: {x_embed.shape} / {add_embed.shape}'
                    return x_embed + add_embed

        # Expand ALL embeddings to (B, F, J, D)
        if joint_embedding is not None:
            joint_embedding = repeat(joint_embedding, 'J D -> B F J D', B=B, F=F)
        # if positional_embedding is not None:
        #     pass
        if serial_embedding is not None:
            serial_embedding = repeat(serial_embedding, 'F D -> B F J D', B=B, J=J)

        # Encoder x_embedding
        x_embedding = None
        x_embedding = self.combine_embed(
            x_embedding,
            joint_embedding
        )
        x_embedding = self.combine_embed(
            x_embedding,
            positional_embedding
        )  # x_embedding == None | joint_ | positional_ | joint_ + positional_
        x_embedding = rearrange(x_embedding, 'B F J D -> (B F) J D')

        src = rearrange(src, 'B F J D -> (B F) J D')  # B, Seq, Dim
        memory_BFJD = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask,
                              x_embedding=x_embedding)
        memory_BFJD = rearrange(memory_BFJD, '(B F) J D -> B F J D', B=B)

        # # encoder only
        # return memory_BFJD

        def update_embedding(memory, appended, dim):
            if memory == None:
                return appended
            else:
                return torch.cat((memory, appended), dim=dim)  # ([2j, j], D) or (B, [2j, j], D)

        mem_embedding = None

        # Decoder
        mem_embedding_BFJD = None
        batched_full_embedding = self.combine_embed(
            batched_full_embedding,
            joint_embedding
        )
        batched_full_embedding = self.combine_embed(
            batched_full_embedding,
            positional_embedding
        )
        batched_full_embedding = self.combine_embed(
            batched_full_embedding,
            serial_embedding
        )  # embedding_ == None | joint_ | positional_ | serial_ | combination_
        batched_full_embedding = rearrange(batched_full_embedding, 'B F J D -> B (F J) D')

        # full_joint_embedding = repeat(joint_embedding, 'J D -> F J D', F=F)
        # full_serial_embedding = repeat(serial_embedding, 'F D -> F J D', J=J)
        # full_embedding = full_joint_embedding + full_serial_embedding

        # batched_full_embedding = repeat(full_embedding, 'F J D -> B (F J) D', B=B)

        memory_BJsD = None  # (B,J,D) -> (B,J*2,D) -> (B,J*3,D) -> (B,J*4,D)
        for frame_id in range(F):
            # Each embedding
            tgt_embedding = batched_full_embedding[:, J* frame_id: J* (frame_id+1), :]   # [0:J] -> [J:2J] -> [2J:3J]
            mem_embedding = batched_full_embedding[:,            : J* (frame_id+1), :]   # [0:J] -> [0:2J] -> [0 :3J]
            # print(tgt_embedding[0])
            # print(mem_embedding[0])

            # Data
            tgt = torch.zeros((B, J, D), device=src.device)  # to gpu
            memory_BJsD = update_embedding(memory_BJsD, memory_BFJD[:, frame_id, :, :], dim=1)

            output = self.decoder(tgt, memory_BJsD, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  tgt_embedding=tgt_embedding, memory_embedding=mem_embedding)

            # update last memory, with decoder output( predict result of this frame )
            memory_BJsD[:, J*frame_id: J*(frame_id+1), :] = output

        return rearrange(memory_BJsD, 'B (F J) D -> B F J D', F=F)


    def forward(self, src,
                joint_embedding: Optional[Tensor] = None,  # ! NEW 3 embedding, (J, C)
                verts_embedding: Optional[Tensor] = None,  # (V, C)
                serial_embedding: Optional[Tensor] = None,  # (F, C)
                positional_embedding: Optional[Tensor] = None,  # (B, F, J, C)
                image_feature: Optional[Tensor] = None,  # B F (H W) C
                image_positional_embedding: Optional[Tensor] =None,  # B F (H W) C
                DiagonalMask: Dict[str, List] = None,
                JointConfMask: Dict[str, Any] = None,
                ReturnEncoderOutput = 'no',
                ):
        if src.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")
        B, F, J, D = src.shape  # BatchSize, FrameCounts, JointCounts, featureDim

        serial_embedding = serial_embedding[:F, :]  # only previous F embeddings is used

        # Expand ALL embeddings to (B, F, J, D)
        if joint_embedding is not None:  # for enc_SA and dec_CA if DecOut == 49
            joint_embedding = repeat(joint_embedding, 'J D -> B F J D', B=B, F=F)
        if self.DecoderForwardConfigs['DecOutCount'] == '49 verts' and \
            verts_embedding is not None:
            verts_embedding = repeat(verts_embedding, 'V D -> B F V D', B=B, F=F)
        # if positional_embedding is not None:             # (B F J D)
        #     pass
        if serial_embedding is not None:
            serial_embedding = repeat(serial_embedding, 'F D -> B F () D', B=B)  # accept J or V

        # Prepare Masking
        ## DiagonalMask = {'enc self': ['part', 0.1], ...}
        if DiagonalMask is None:
            DiagonalMask = {
                'enc self' : ['no', 0],
                'dec self' : ['no', 0],
                'dec cross': ['no', 0],
            }
        ## JointConfMask = {'mode': 'mask', 'joint mask': tensor, ...}
        if JointConfMask is None:
            JointConfMask = {
                'enc self': 'no',
                'dec cross': 'no',
                'joint mask': None,
                'joint conf': None,
            }

        # Enc / Dec
        if self.Mode == 'base':
            memory_BFJD, each_enc_out = self.forward_encoder(src,
                joint_embedding, positional_embedding, serial_embedding,
                image_feature, image_positional_embedding,
                DiagonalMask, JointConfMask,
                ReturnEachEncoderLayerOutput = (ReturnEncoderOutput=='each'),
                )
            out = self.forward_decoder(memory_BFJD, # .clone(),
                joint_embedding, positional_embedding, serial_embedding, verts_embedding,
                image_feature, image_positional_embedding,
                DiagonalMask, JointConfMask,
                )
        elif self.Mode == 'encoder only':
            #TODO: Different mask shape, No DiagMask function temporary
            # 'B F J D -> B (F J) D'
            # diff encoder input shape
            # diff norm dimensions(F*J, 256)
            out, each_enc_out = self.forward_encoder(src,
                joint_embedding, positional_embedding, serial_embedding,
                image_feature, image_positional_embedding,
                DiagonalMask, JointConfMask,
                ReturnEachEncoderLayerOutput = (ReturnEncoderOutput=='each'),
                # temporal_mode=True,  # ? normal encoder temporally, for encoder only ablation
                )
        elif self.Mode == 'decoder only':
            out = self.forward_decoder(src, # .clone(),
                joint_embedding, positional_embedding, serial_embedding, verts_embedding,
                image_feature, image_positional_embedding,
                DiagonalMask, JointConfMask,
                )

        if CHECK_VAR:
            import pandas as pd
            pd.set_option('display.max_rows', None)
            df = pd.DataFrame(_variances, columns=['name', 'x', '_out'])
            df['_out / x'] = df['_out'] / df['x']
            print(df)

        to_return = [out, []]
        # main_out, (Opt)encoder out

        # ['no', 'last', 'each']
        if ReturnEncoderOutput == 'last':
            assert self.Mode=='base', f'mode should be "base" if ReturnEncoderOutput=="last", get: {self.Mode}'
            to_return[1] = [memory_BFJD]
        elif ReturnEncoderOutput == 'each':
            to_return[1] = each_enc_out

        return to_return  # return BFJD or BFVD or BF(J+V)D


    def forward_encoder(self, src,
                        joint_embedding, positional_embedding, serial_embedding,
                        image_feature, image_positional_embedding,  #  B F (H W) C
                        DiagonalMask: Dict[str, List],
                        JointConfMask: Dict[str, Any],
                        ReturnEachEncoderLayerOutput,
                        temporal_mode=False):
        ''' temporal_mode: reshape to B (F J) D, rather than (B F) F D '''
        B, F, J, D = src.shape
        L = self.encoder.num_layers
        device = src.device

        # Encoder x_embedding
        x_embedding = None
        x_embedding = self.combine_embed(x_embedding, joint_embedding)
        x_embedding = self.combine_embed(x_embedding, positional_embedding) 
        # x_embedding == None | joint_ | positional_ | joint_ + positional_
        if temporal_mode:
            x_embedding = self.combine_embed(x_embedding, serial_embedding)


        # Encoder DiagMask
        SA_diag_mask = self._prepare_diag_mask(B*F, J, _mode= DiagonalMask['enc self'][0],
                                                       _ratio=DiagonalMask['enc self'][1], _device=device)
        if SA_diag_mask is not None:
            SA_diag_mask = rearrange(SA_diag_mask, '(B F) J1 J2 -> B F J1 J2', B=B)
        # SA_diag_mask == None | (B F J J)
        # Encoder ConfMask
        SA_conf_joint_mask = None
        WeightedPaddingMask = False
        if JointConfMask['enc self'] == 'mask':
            SA_conf_joint_mask = JointConfMask['joint mask']  # (B F J)
        elif JointConfMask['enc self'] == 'weight':
            SA_conf_joint_mask = JointConfMask['joint conf']  # (B F J)
            WeightedPaddingMask = True

        memory2 = memory2_embedding = None
        if self.EncCross2ImageFeat:
            memory2 = rearrange(image_feature, 'B F HW C -> (B F) HW C')
            memory2_embedding = rearrange(image_positional_embedding, 'B F HW C -> (B F) HW C')

        # Reshape & Forward
        if not temporal_mode:
            x_embedding = rearrange(x_embedding, 'B F J D -> (B F) J D')
            src = rearrange(src, 'B F J D -> (B F) J D')  # B, Seq, Dim
            if SA_diag_mask is not None:
                SA_diag_mask = rearrange(SA_diag_mask, 'B F J1 J2 -> (B F) J1 J2')
            if SA_conf_joint_mask is not None:
                SA_conf_joint_mask = rearrange(SA_conf_joint_mask, 'B F J -> (B F) J')

            memory_BFJD, mem_list = self.encoder(src,
                                       mask=SA_diag_mask,
                                       src_key_padding_mask=SA_conf_joint_mask,
                                       x_embedding=x_embedding,
                                       WeightedPaddingMask=WeightedPaddingMask,
                                       ReturnEachLayerOutput=ReturnEachEncoderLayerOutput,

                                       memory2=memory2,
                                       memory2_embedding=memory2_embedding,  # all positional_emb are same
                                       )

            if mem_list != []: # or if ReturnEachEncoderLayerOutput:
                for i in range(len(mem_list)):
                    mem_list[i] = rearrange(mem_list[i], '(B F) J D -> B F J D', B=B)
            return rearrange(memory_BFJD, '(B F) J D -> B F J D', B=B), mem_list

        else:
            x_embedding = rearrange(x_embedding, 'B F J D -> B (F J) D')
            src = rearrange(src, 'B F J D -> B (F J) D')  # B, Seq, Dim
            if SA_conf_joint_mask is not None:
                SA_conf_joint_mask = rearrange(SA_conf_joint_mask, 'B F J -> B (F J)')

            memory_BFJD, mem_list = self.encoder(src,
                                       mask=None,
                                       src_key_padding_mask=SA_conf_joint_mask,
                                       x_embedding=x_embedding,
                                       WeightedPaddingMask=WeightedPaddingMask,
                                       ReturnEachLayerOutput=ReturnEachEncoderLayerOutput)

            if mem_list != []: # or if ReturnEachEncoderLayerOutput:
                for i in range(len(mem_list)):
                    mem_list[i] = rearrange(mem_list[i], 'B (F J) D -> B F J D', F=F)
            return rearrange(memory_BFJD, 'B (F J) D -> B F J D', F=F), mem_list


    def forward_decoder(self, memory_BFJD,
                        joint_embedding, positional_embedding, serial_embedding, verts_embedding,
                        image_feature, image_positional_embedding,  #  B F (H W) C
                        DiagonalMask: Dict[str, List],
                        JointConfMask: Dict[str, Any]):
        ''' Note that: ..._BFJD.shape == (B F J D), ..._BJsD.shape == (B FJ D)
        '''

        DecOutCount = self.DecoderForwardConfigs['DecOutCount']

        B, F, J, D = memory_BFJD.shape
        V = 49
        # tgt_N = 21 if DecOutCount == '21 joint' else 49
        tgt_N = 49  if DecOutCount == '49 verts' else \
                J   if DecOutCount == '21 joint' else \
                J+V if DecOutCount == '21 + 49'  else -1

        L = self.decoder.num_layers
        device = memory_BFJD.device

        def update_embedding(memory, appended, dim):
            if memory == None:
                return appended
            else:
                return torch.cat((memory, appended), dim=dim)  # ([2j, j], D) or (B, [2j, j], D)

        # Decoder Embeddings
        mem_embedding_BFJD = None
        mem_embedding_BFJD = self.combine_embed(mem_embedding_BFJD, joint_embedding)
        mem_embedding_BFJD = self.combine_embed(mem_embedding_BFJD, positional_embedding)
        mem_embedding_BFJD = self.combine_embed(mem_embedding_BFJD, serial_embedding)  # (BFJD) + (BF1D)
        # embedding_ == None | joint_ | positional_ | serial_ | combination_
        if DecOutCount in ('49 verts', '21 + 49'):
            tgt_embedding_BFJD = None
            tgt_embedding_BFJD = self.combine_embed(tgt_embedding_BFJD, verts_embedding)
            tgt_embedding_BFJD = self.combine_embed(tgt_embedding_BFJD, serial_embedding)  # (BFVD) + (BF1D)
            # no positional embed: (BFVD) <-> (BFJD)
        elif DecOutCount == '21 joint':
            tgt_embedding_BFJD = mem_embedding_BFJD
        if DecOutCount == '21 + 49':
            # combine mem_embed(joint embed) and tgt embed(verts embed)
            tgt_embedding_BFJD = \
                torch.cat([mem_embedding_BFJD, tgt_embedding_BFJD], dim=2)  # J and V in (BFJD, BFVD)


        # Decoder DiagMask
        SA_diag_mask_BFJJ = self._prepare_diag_mask(B*F, tgt_N, _mode= DiagonalMask['dec self'][0],
                                                                _ratio=DiagonalMask['dec self'][1], _device=device)
        CA_diag_mask_BFJJ = self._prepare_diag_mask(B*F, J,     _mode= DiagonalMask['dec cross'][0],
                                                                _ratio=DiagonalMask['dec cross'][1], _device=device)
        if SA_diag_mask_BFJJ is not None:
            SA_diag_mask_BFJJ = rearrange(SA_diag_mask_BFJJ, '(B F) J1 J2 -> B F J1 J2', B=B)
        if CA_diag_mask_BFJJ is not None:
            CA_diag_mask_BFJJ = rearrange(CA_diag_mask_BFJJ, '(B F) J1 J2 -> B F J1 J2', B=B)
            # needs to be appended after zero_mask if frame > 0
        # {SA_diag_mask_BFJJ, CA_diag_mask_BFJJ} == None | (B F J J)

        # Decoder ConfMask
        mem_joint_conf_mask_BFJ = None
        WeightedMemPaddingMask = False
        if JointConfMask['dec cross'] == 'mask':
            mem_joint_conf_mask_BFJ = JointConfMask['joint mask']  # (B F J)
        elif JointConfMask['dec cross'] == 'weight':
            mem_joint_conf_mask_BFJ = JointConfMask['joint conf']  # (B F J)
            WeightedMemPaddingMask = True
        # TODO, check if DecOutCount == '21 + 49' have "NO" effect to mem_joint_conf_mask_BFJ
        # attn.key = 21x

        ###### Prepared to enter Decoder ######
        # Reshape embedding, mask, memory
        mem_embedding_BJsD = rearrange(mem_embedding_BFJD, 'B F J D -> B (F J) D')

        mem_conf_mask_BJs = None
        if mem_joint_conf_mask_BFJ is not None:
            mem_conf_mask_BJs = rearrange(mem_joint_conf_mask_BFJ, 'B F J -> B (F J)')
        # Init
        memory_BJsD = None  # (B,J,D) -> (B,2J,D) -> (B,3J,D)
        if self.DecoderForwardConfigs['DecMemUpdate'] == 'full':
            memory_BJsD = rearrange(memory_BFJD, 'B F J D -> B (F J) D').clone()
            # fix bug in FR70FF
            # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
        output_F_BVD = []

        # Iteratively predict each frame
        for frame_id in range(F):
            # Iterative embedding: {tgt_|mem_embedding}
            tgt_embedding = tgt_embedding_BFJD[:, frame_id]  # (B J D) or (B V D) or (B J+V D)
            if self.DecoderForwardConfigs['DecMemUpdate'] == 'append':
                mem_embedding = mem_embedding_BJsD[:, : J* (frame_id+1)] # [0:J] -> [0:2J] -> [0 :3J]
            else:
                mem_embedding = mem_embedding_BJsD                       # (B FJ D)

            # Iterative data
            if self.DecoderForwardConfigs['DecSrcContent'] == 'zero': # <- 21/ 49/ 70
                tgt = torch.zeros((B, tgt_N, D), device=device)
            elif self.DecoderForwardConfigs['DecSrcContent'] == 'feature': # <- 21/ 49
                if DecOutCount == '21 joint':
                    tgt = memory_BFJD[:, frame_id]
                elif DecOutCount == '49 verts':
                    tgt = torch.bmm(self.matrix.repeat(B, 1, 1), memory_BFJD[:, frame_id])  # (B 49 21) @ (B J D)
            else:
                assert DecOutCount == '21 + 49', f'DecOutCount must be 70 while DecSrcContent == {self.DecoderForwardConfigs["DecSrcContent"]}'
                _joint_content, _verts_content = self.DecoderForwardConfigs['DecSrcContent'].split()
                tgt = torch.zeros((B, tgt_N, D), device=device)
                if _joint_content == 'feature':
                    tgt[:, :J] = memory_BFJD[:, frame_id]  # [feature, zero]
                if _verts_content == 'feature':
                    tgt[:, J:] = torch.bmm(self.matrix.repeat(B, 1, 1), memory_BFJD[:, frame_id])

            if self.DecoderForwardConfigs['DecMemUpdate'] == 'append':
                memory_BJsD = update_embedding(memory_BJsD, memory_BFJD[:, frame_id], dim=1)
            else:                                          # 'full'
                pass  # memory_BJsD = full, no need to change

            # Iterative diag mask
            SA_diag_mask = CA_diag_mask = None
            if SA_diag_mask_BFJJ is not None:  # (B F tgt_N tgt_N): (B F J J) or (B F V V)
                SA_diag_mask = SA_diag_mask_BFJJ[:, frame_id]
            if CA_diag_mask_BFJJ is not None:
                CA_diag_mask = CA_diag_mask_BFJJ[:, frame_id]
                CA_diag_mask = self._append_cross_attn_diag_mask(CA_diag_mask, frame_id, F, self.DecoderForwardConfigs['DecMemUpdate'])
                # [\] -> [O O O \] if 'append',     [\] -> [O O O \ O O O O] if 'full'
            # SA_diag_mask = None | (B, J, J)
            # CA_diag_mask = None | (B, J, J * (frame+1)) | (B, J, J * F)

            # Iterative conf joint mask
            mem_padding_mask = None
            if mem_conf_mask_BJs is not None:  # DecOutCount == '21 joint
                if self.DecoderForwardConfigs['DecMemUpdate'] == 'append':
                    mem_padding_mask = mem_conf_mask_BJs[:, : J* (frame_id+1)]
                else:                                          # 'full'
                    mem_padding_mask = mem_conf_mask_BJs

            memory2 = memory2_embedding = None
            if self.DecCross2ImageFeat:
                memory2 = image_feature[:, frame_id]
                memory2_embedding = image_positional_embedding[:, frame_id]

            output = self.decoder(tgt, memory_BJsD,
                tgt_mask=SA_diag_mask, memory_mask=CA_diag_mask,
                tgt_key_padding_mask=None, memory_key_padding_mask=mem_padding_mask,
                tgt_embedding=tgt_embedding, memory_embedding=mem_embedding,
                WeightedMemPaddingMask=WeightedMemPaddingMask,

                memory2=memory2,
                memory2_embedding=memory2_embedding,  # all positional_emb are same
            )

            # Iterative update memory
            if self.DecoderForwardConfigs['DecMemReplace'] == True:  # DecOutCount in ('21 joint', '21 + 49')
                if DecOutCount == '21 + 49':
                    memory_BJsD[:, J*frame_id: J*(frame_id+1), :] = output[:, :21, :]  # (B J+V D)
                else:
                    memory_BJsD[:, J*frame_id: J*(frame_id+1), :] = output
            output_F_BVD += [output]

        return rearrange(output_F_BVD, 'F B J D -> B F J D')


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

    def _prepare_diag_mask(self, _counts, _joints, _mode, _device, _ratio=0.1):
        '''
        _counts: Batch * Frame (* Head)
        _joints: attention mask width
        _mode: {'no', 'part', 'full'}
        return: Size(_counts, _joints, _joints) masks

        ...

        encoder self-attn
            (BFL, J, J)
            == (BF, J, J) for (0 ... L: layer_counts)
        decoder self, cross
            (BFL, J, J)
            == (BL, J, J) for (0 ... F)

            # cross = ones (cat) mask

        BFL: Batch, Frame(, Head)
        '''
        if self.training == False:
            return None
        if _mode == 'no':
            return None

        elif _mode == 'part':
            diagonal = torch.rand((_counts, _joints), device=_device) < _ratio  # (BFL, J)
        elif _mode == 'full':
            diagonal = torch.rand(_counts, device=_device) < _ratio  # (BFL)
            diagonal = repeat(diagonal, 'BFL -> BFL J', J=_joints)
        else:
            raise Exception(f'_mode should be in ("no", "part", "full"), but got: {_mode}')

        masks = torch.diag_embed(diagonal)
        # print(masks.shape)
        return masks

    def _append_cross_attn_diag_mask(self, _diag_mask: Tensor, _frame_id: int, _max_frame=None, _DecMemUpdate: str='append'):
        '''
        [\] -> [O O O \], used in decoder cross-attn
            or [O O O \ O O O O] if _DecMemUpdate=='full'

        _diag_mask: Size(B, J, J) for frame in (0 ... F)
             (X) or Size(B *H, J, J) for frame in (0 ... F), if nhead > 1

        _frame_id: (0 ~ F-1) if _DecMemUpdate=='append'
        _max_frame: (F)      if _DecMemUpdate=='full'

        _DecMemUpdate: method to update memory part, 'append' or 'full'

        return: mask with Size(B, J, [_frame_id *J] ) if 'append'
                                    [O O O O \]
        return: mask with Size(B, J, F*J)             if 'full'
                                    [O O O \ O O O O]

        Note that: only applied on joint_2_joint case
                                no verts_2_joint case
        '''
        _device = _diag_mask.device
        B, J, _ = _diag_mask.shape

        if _DecMemUpdate == 'append':
            _zeros_shape = (B, J, J * (_frame_id+1))
        else:
            assert _max_frame is not None, f'_max_frame should be specified(int) if _DecMemUpdate=="full", got: None'
            _zeros_shape = (B, J, J * _max_frame)

        masks = torch.zeros(_zeros_shape, device=_device, dtype=torch.bool)
        masks[:, :, _frame_id*J : (_frame_id+1)*J] = _diag_mask  # rightest square <- diag mask

        return masks


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
    transformer = get_transformer(
        256, nhead=1, num_encoder_layers=3, num_decoder_layers=3,
        norm_first=False).to(device).eval()

    x = torch.randn((16, 8, 21, 256)).to(device)
    j_emb = torch.randn((21, 256)).to(device)
    p_emb = torch.randn((16, 8, 21, 256)).to(device)
    s_emb = torch.randn((20, 256)).to(device)

    with torch.no_grad():
        # edit forward() to compare old_version: forward_
        #                       and new_version
        #                   with torch.equal(out1, out2)
        # pass check
        out = transformer(x, joint_embedding=j_emb, positional_embedding=p_emb, serial_embedding=s_emb)

def exp_encoder():
    d_model = 128 # feature len
    nhead = 4  # each head process 128/4 feature
    layer_norm_eps = 1e-5
    device = torch.device('cuda', 0)
    factory_kwargs = {'device': device, 'dtype': None}

    encoder_layer = MyEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
    encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)  # final norm in encoder
    encoder = MyEncoder(encoder_layer=encoder_layer, num_layers=3, norm=encoder_norm)

    x = torch.empty((32, 22, 128)).to(device)  # B J D == Batch Sequence Dimension_of_feature
    embedding = torch.empty((1, 22, 128)).to(device)
    embedding = embedding.expand((32, -1, -1))
    encoder.to(device)

    x = encoder(x, x_embedding=embedding)
    print(x.device)
    print(x.shape)

def exp_decoder():
    d_model = 128 # feature len
    nhead = 4  # each head process 128/4 feature
    layer_norm_eps = 1e-5
    device = torch.device('cuda', 0)
    factory_kwargs = {'device': device, 'dtype': None}

    decoder_layer = MyDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
    decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)  # final norm in decoder
    decoder = MyDecoder(decoder_layer=decoder_layer, num_layers=3, norm=decoder_norm)

    x = torch.empty((32, 22, 128)).to(device)  # B J D == Batch Sequence Dimension_of_feature
    x_embedding = torch.empty((1, 22, 128)).to(device)
    x_embedding = x_embedding.expand((32, -1, -1))

    mem = torch.empty((32, 22, 128)).to(device)
    mem_embedding = torch.empty((1, 22, 128)).to(device)
    mem_embedding = mem_embedding.expand((32, -1, -1))

    decoder.to(device)

    x = decoder(x, mem, tgt_embedding=x_embedding, memory_embedding=mem_embedding)
    print(decoder)
    print(x.device)
    print(x.shape)

def exp_transformer():
    d_model = 128 # feature len
    nhead = 4  # each head process 128/4 feature
    layer_norm_eps = 1e-5
    device = torch.device('cuda', 0)
    # factory_kwargs = {'device': device, 'dtype': None}
    factory_kwargs = {}

    encoder_layer = MyEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
    encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)  # final norm in encoder
    encoder = MyEncoder(encoder_layer=encoder_layer, num_layers=3, norm=encoder_norm)

    decoder_layer = MyDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
    decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)  # final norm in decoder
    decoder = MyDecoder(decoder_layer=decoder_layer, num_layers=3, norm=decoder_norm)

    transformer = MyTransformer(d_model=d_model, nhead=nhead,
                                num_encoder_layers=2, num_decoder_layers=2,
                                dim_feedforward=2048, dropout=0.1,
                                custom_encoder=encoder, custom_decoder=decoder,
                                layer_norm_eps=layer_norm_eps,
                                batch_first=True, **factory_kwargs).to(device)

    B, F, J, D = 3, 8, 22, 128
    joint_embedding = torch.randn((J, D)).to(device)
    serial_embedding = torch.randn((F, D)).to(device)
    x = torch.randn((B, F, J, D)).to(device)  # B F J D
    out = transformer(x, joint_embedding=joint_embedding, serial_embedding=serial_embedding)

def transformer_config_correctness_check(
    norm_first, NormTwice,
    Mode,
    DecOutCount, DecSrcContent, DecMemUpdate, DecMemReplace,
    matrix,
    ):
    ErrorMessages = []

    InputDict = {
        'norm_first': norm_first,
        'NormTwice': NormTwice,
        'Mode': Mode,
        'DecOutCount': DecOutCount,
        'DecSrcContent': DecSrcContent,
        'DecMemUpdate': DecMemUpdate,
        'DecMemReplace': DecMemReplace,
    }

    PossibleConfigs = {
        'norm_first':       [True, False],
        'NormTwice':        [True, False],
        'Mode':             ['base', 'encoder only', 'decoder only'],
        'DecOutCount':      ['21 joint', '49 verts', '21 + 49'],
        'DecSrcContent':    ['zero', 'feature',
                             'zero zero',    'feature zero',
                             'zero feature', 'feature feature'],
        'DecMemUpdate':     ['append', 'full'],
        'DecMemReplace':    [True, False],
    }

    # All config contents should be in list
    for key, value in InputDict.items():
        if value not in PossibleConfigs[key]:
            ErrorMessages += [f'{key} should be in {PossibleConfigs[key]}, got {key}: {value}']

    # Configs Conflict
    ## NormTwice, norm_first
    if NormTwice == True:
        if norm_first != False:
            ErrorMessages += [f'norm_first should be True while NormFirst == True, got norm_first: {norm_first}']

    ## Decoder configs
    if DecMemReplace == True:
        if DecOutCount in ('49 verts'):
            ErrorMessages += [f'DecOutCount should be "21 joint" or "21 + 49" while DecMemReplace == True, got DecOutCount: {DecOutCount}']
    if DecSrcContent == 'feature':
        if DecOutCount in ('49 verts'):
            if matrix is None:
                ErrorMessages += [f'DecOutCount should be "21 joint" or "21 + 49" while DecSrcContent == "feature", got DecOutCount: {DecOutCount}']
            elif not isinstance(matrix, nn.Parameter):
                ErrorMessages += [f'"matrix" must be type: nn.Parameter while DecSrcContent == "feature" & DecOutCount == "49 verts", ' + \
                                  f'got: {type(matrix)}']
            elif not matrix.shape == (49, 21):
                ErrorMessages += [f'"matrix.shape" must be (49, 21) while DecSrcContent == "feature" & DecOutCount == "49 verts", ' + \
                                  f'got: {matrix.shape}']
    if DecOutCount == '21 + 49':
        if len(DecSrcContent.split()) != 2:
            ErrorMessages += [f'DecSrcContent should have 2 (zero/feature)s while DecOutCount == "21 + 49", got: {DecSrcContent}']
    else:
        if len(DecSrcContent.split()) != 1:
            ErrorMessages += [f'DecSrcContent should have only 1 (zero/feature) while DecOutCount in "21 joint" or "49 verts", ' + \
                              f'got: {DecSrcContent}']

    if ErrorMessages != []:
        print('[Config Error]')
        for err_msg in ErrorMessages:
            print(err_msg)
        raise Exception('transformer configs conflict, printed')

def get_transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                    layer_norm_eps=1e-5, norm_first=False,
                    NormTwice=False,
                    Mode: Optional[str]='base',
                    DecoderForwardConfigs: Optional[dict]=None,
                    matrix: Optional[nn.Parameter]=None,
                    EncAddCrossAttn2ImageFeat: Optional[bool]=False,
                    DecAddCrossAttn2ImageFeat: Optional[bool]=False,
                    ) -> MyTransformer:
    if DecoderForwardConfigs is None:
        DecoderForwardConfigs = {
            'DecOutCount': '21 joint',
            'DecSrcContent': 'zero',
            'DecMemUpdate': 'append',
            'DecMemReplace': True,
        }
    transformer_config_correctness_check(norm_first=norm_first, NormTwice=NormTwice, Mode=Mode,
                                         **DecoderForwardConfigs, matrix=matrix)

    encoder_layer = MyEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, norm_first=norm_first, NormTwice=NormTwice,
                                   AddCrossAttn2ImageFeat=EncAddCrossAttn2ImageFeat)
                                 # dim_feedforward=2048, dropout=0.1, layer_norm_eps=layer_norm_eps
    encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)  # final norm in encoder
    # encoder_norm = nn.LayerNorm((21, d_model), eps=layer_norm_eps)  # final norm in encoder
    encoder = MyEncoder(encoder_layer=encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm)

    decoder_layer = MyDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, norm_first=norm_first, NormTwice=NormTwice,
                                   AddCrossAttn2ImageFeat=DecAddCrossAttn2ImageFeat)
                                 # dim_feedforward=2048, dropout=0.1, layer_norm_eps=layer_norm_eps
    decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)  # final norm in decoder
    # decoder_norm = nn.LayerNorm((21, d_model), eps=layer_norm_eps)  # final norm in decoder
    decoder = MyDecoder(decoder_layer=decoder_layer, num_layers=num_decoder_layers, norm=decoder_norm)

    transformer = MyTransformer(
        d_model=d_model, nhead=nhead,
        # num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
        # dim_feedforward=2048, dropout=0.1,
        custom_encoder=encoder, custom_decoder=decoder,
        # layer_norm_eps=layer_norm_eps,
        batch_first=True,
        Mode=Mode,
        DecoderForwardConfigs=DecoderForwardConfigs,
        matrix=matrix,
        EncAddCrossAttn2ImageFeat=EncAddCrossAttn2ImageFeat,
        DecAddCrossAttn2ImageFeat=DecAddCrossAttn2ImageFeat,
    )
    return transformer

def conv_layer(channel_in, channel_out, ks=1, stride=1, padding=0, dilation=1, bias=False, bn=True, relu=True, group=1):
    """Conv block

    Args:
        channel_in (int): input channel size
        channel_out (int): output channel size
        ks (int, optional): kernel size. Defaults to 1.
        stride (int, optional): Defaults to 1.
        padding (int, optional): Defaults to 0.
        dilation (int, optional): Defaults to 1.
        bias (bool, optional): Defaults to False.
        bn (bool, optional): Defaults to True.
        relu (bool, optional): Defaults to True.
        group (int, optional): group conv parameter. Defaults to 1.

    Returns:
        Sequential: a block with bn and relu
    """
    _conv = nn.Conv2d
    sequence = [_conv(channel_in, channel_out, kernel_size=ks, stride=stride, padding=padding, dilation=dilation,
                      bias=bias, groups=group)]
    if bn:
        sequence.append(nn.BatchNorm2d(channel_out))
    if relu:
        sequence.append(nn.ReLU())

    return nn.Sequential(*sequence)

class Reg2DDecode3D(nn.Module):
    def __init__(self, latent_size, out_channels):
        super(Reg2DDecode3D, self).__init__()
        self.latent_size = latent_size
        self.out_channels = out_channels
        self.de_layer_conv = conv_layer(self.latent_size, self.out_channels[- 1], 1, bn=False, relu=False)

        nhead = 4

        self.joint_embed = nn.Embedding(22, self.latent_size)
        self.serial_embed = nn.Embedding(10, self.latent_size)
        # ? exp correctness
        # counter = torch.arange(25)
        # self.joint_embed = counter[:22, None].expand(-1, self.latent_size)
        # self.serial_embed = counter[:8, None] * -1
        # self.serial_embed = self.serial_embed.expand(-1, self.latent_size)
        self.transformer = get_transformer(self.latent_size, nhead=4, num_encoder_layers=3, num_decoder_layers=3)


    def index(self, feat, uv):
        uv = uv.unsqueeze(2)  # [B, N, 1, 2]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        # feat: [B, C, H, W], uv: [B, Ho, Wo, 2(i, j)], samples: [B, C, Ho, Wo]
        # sample[B, C, y, x] = feat[B, C, Hi, Wj] where (i, j) = uv[B, y, x]

        return samples[:, :, :, 0]  # [B, C, N]

    def forward(self, uv, x):
        uv = torch.clamp((uv - 0.5) * 2, -1, 1)
        x = self.de_layer_conv(x)
        x = self.index(x, uv).permute(0, 2, 1)
        # [B, N, C]
        x = rearrange(x, '(B F) J D -> B F J D', F=8)
        x = self.transformer(x, joint_embedding=self.joint_embed.weight, serial_embedding=self.serial_embed.weight)
        return x


if __name__ == '__main__':
    uv_reg = torch.randn((32, 22, 2))  # set joint counts to 22( + global feat )
    latent = torch.randn((32, 256, 4, 4))

    model_reg = Reg2DDecode3D(256, [32, 64, 128, 256])

    out = model_reg(uv_reg, latent)
    # exp_transformer()


    # backbone_feature = torch.randn((2, 8, ))