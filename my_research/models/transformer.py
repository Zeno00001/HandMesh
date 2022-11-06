'''
My updated transformer architecture, edit from PyTorch official Transformer code
see exp_encoder(), exp_decoder(), exp_transformer() for more details
'''

import torch
from torch import nn
from einops import rearrange, repeat

# for func input format
from torch import Tensor
from typing import Optional


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
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
        x_embedding: Optional[Tensor] = None,  # ! NEW joint embedding
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
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, x_embedding)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, x_embedding))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                  x_embedding: Optional[Tensor],  # ! NEW joint embedding
                  ) -> Tensor:
        q = k = self.with_pos_embed(x, x_embedding)
        # q = k = self.with_pos_embed(q, serial_embedding)

        x = self.self_attn(query=q, key=k, value=x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
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

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, x_embedding=x_embedding)

        if self.norm is not None:
            output = self.norm(output)

        return output

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
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                tgt_embedding: Optional[Tensor] = None, memory_embedding: Optional[Tensor] = None  # ! NEW 2 embedding
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
                                    x_embedding=tgt_embedding, mem_embedding=memory_embedding)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, x_embedding=tgt_embedding))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask,
                                               x_embedding=tgt_embedding, mem_embedding=memory_embedding))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                  x_embedding: Optional[Tensor],  # ! NEW joint_query embedding
                  ) -> Tensor:
        q = k = self.with_pos_embed(x, x_embedding)

        x = self.self_attn(query=q, key=k, value=x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                   x_embedding: Optional[Tensor], mem_embedding: Optional[Tensor]  # ! NEW 2 embeddings
                   ) -> Tensor:
        q = self.with_pos_embed(x, x_embedding)
        k = self.with_pos_embed(mem, mem_embedding)
        # v = mem

        x = self.multihead_attn(query=q, key=k, value=mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

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
                tgt_embedding: Optional[Tensor] = None, memory_embedding: Optional[Tensor] = None  # ! NEW 2 embedding
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

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         tgt_embedding=tgt_embedding,
                         memory_embedding=memory_embedding)

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
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,  # ! remove tgt param
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                joint_embedding: Optional[Tensor] = None, serial_embedding: Optional[Tensor] = None  # ! NEW 2 embedding
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
        src = rearrange(src, 'B F J D -> (B F) J D')  # B, Seq, Dim

        # Encoder x_embedding
        x_embedding = None
        if joint_embedding != None:
            x_embedding = repeat(joint_embedding, 'J D -> BF J D', BF = B * F)
        memory_BFJD = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask,
                              x_embedding=x_embedding)
        memory_BFJD = rearrange(memory_BFJD, '(B F) J D -> B F J D', B=B)

        def update_embedding(memory, appended, dim):
            if memory == None:
                return appended
            else:
                return torch.cat((memory, appended), dim=dim)  # ([2j, j], D) or (B, [2j, j], D)

        mem_embedding = None

        # Decoder
        full_joint_embedding = repeat(joint_embedding, 'J D -> F J D', F=F)
        full_serial_embedding = repeat(serial_embedding, 'F D -> F J D', J=J)
        full_embedding = full_joint_embedding + full_serial_embedding

        batched_full_embedding = repeat(full_embedding, 'F J D -> B (F J) D', B=B)

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

    '''
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,  # ! remove tgt param
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                joint_embedding: Optional[Tensor] = None, serial_embedding: Optional[Tensor] = None  # ! NEW 2 embedding
                ) -> Tensor:
        # ! tgt is forwarded from zeros()
        # ! old method to compute embedding

        if src.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        B, F, J, D = src.shape  # BatchSize, FrameCounts, JointCounts, featureDim
        src = rearrange(src, 'B F J D -> (B F) J D')  # B, Seq, Dim

        # Encoder x_embedding
        x_embedding = repeat(joint_embedding, 'J D -> BF J D', BF = B * F)
        memory_BFJD = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask,
                              x_embedding=x_embedding)
        memory_BFJD = rearrange(memory_BFJD, '(B F) J D -> B F J D', B=B)

        def update_embedding(memory, appended, dim):
            if memory == None:
                return appended
            else:
                return torch.cat((memory, appended), dim=dim)  # ([2j, j], D) or (B, [2j, j], D)

        # Decoder
        mem_embedding = None
        memory_BJsD = None  # (B,J,D) -> (B,J*2,D) -> (B,J*3,D) -> (B,J*4,D)
        for frame_id in range(F):
            # Decoder tgt_embedding
            tgt_joint_embedding = joint_embedding
            tgt_serial_embedding = repeat(serial_embedding[frame_id], 'D -> J D', J=J)
            tgt_embedding = tgt_joint_embedding + tgt_serial_embedding

            # Decoder mem_embedding
            mem_embedding = update_embedding(mem_embedding, tgt_embedding, dim=0)  # ([2J, J], D) -> (3J, D)

            batched_tgt_embedding = repeat(tgt_embedding, 'J D -> B J D', B=B)
            batched_mem_embedding = repeat(mem_embedding, 'J D -> B J D', B=B)

            # Data
            tgt = torch.zeros_like(batched_tgt_embedding)
            memory_BJsD = update_embedding(memory_BJsD, memory_BFJD[:, frame_id, :, :], dim=1)

            output = self.decoder(tgt, memory_BJsD, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  tgt_embedding=batched_tgt_embedding, memory_embedding=batched_mem_embedding)

            # update last memory, with decoder output( predict result of this frame )
            memory_BJsD[:, J*frame_id:J*(frame_id+1), :] = output

        return rearrange(memory_BJsD, 'B (F J) D -> B F J D', F=F)
    '''

    def combine_embed(self, embed_1: Optional[Tensor], embed_2: Optional[Tensor]):
        if embed_1 != None and embed_2 != None and embed_1.shape != embed_2.shape:
            raise RuntimeError(f'positional embedding shape not matched, 1: {embed_1.shape}, 2: {embed_2.shape}')

        if embed_1 == None:
            return embed_2
        elif embed_2 == None:
            return embed_1
        else:
            return embed_1 + embed_2


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

def get_transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                    layer_norm_eps=1e-5
                    ):
    encoder_layer = MyEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
                                 # dim_feedforward=2048, dropout=0.1, layer_norm_eps=layer_norm_eps
    encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)  # final norm in encoder
    encoder = MyEncoder(encoder_layer=encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm)

    decoder_layer = MyDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
                                 # dim_feedforward=2048, dropout=0.1, layer_norm_eps=layer_norm_eps
    decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)  # final norm in decoder
    decoder = MyDecoder(decoder_layer=decoder_layer, num_layers=num_decoder_layers, norm=decoder_norm)

    transformer = MyTransformer(
        d_model=d_model, nhead=nhead,
        # num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
        # dim_feedforward=2048, dropout=0.1,
        custom_encoder=encoder, custom_decoder=decoder,
        # layer_norm_eps=layer_norm_eps,
        batch_first=True
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