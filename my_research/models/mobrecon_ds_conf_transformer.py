# Copyright (c) Xingyu Chen. All Rights Reserved.

"""
 * @file mobrecon_ds_conf_transformer.py
 * @author chenxingyu (chenxy.sean@gmail.com)
 * @edited clashroyaleisgood @github
 * @brief MobRecon + Transformer model
 * 
 * @copyright Copyright (c) 2022 chenxingyu
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import torch
from einops import rearrange, repeat, reduce
# from my_research.models.densestack import DenseStack_Backnone
from my_research.models.densestack_conf import DenseStack_Conf_Backbone
# from my_research.models.modules import Reg2DDecode3D
from my_research.models.modules import SpiralDeblock, conv_layer, linear_layer
from my_research.models.transformer import get_transformer
from my_research.models.positional_embedding import uv_encoding

from my_research.models.loss import l1_loss, normal_loss, edge_length_loss, contrastive_loss_3d, contrastive_loss_2d
from utils.read import spiral_tramsform
from conv.spiralconv import SpiralConv
from conv.dsconv import DSConv
from my_research.build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class MobRecon_DS_conf_Transformer(nn.Module):
    def __init__(self, cfg):
        """Init a MobRecon-DenseStack + conf + transformer model

        Args:
            cfg : config file
        """
        super(MobRecon_DS_conf_Transformer, self).__init__()
        self.cfg = cfg
        self.backbone = DenseStack_Conf_Backbone(latent_size=cfg.MODEL.LATENT_SIZE,
                                                 kpts_num=cfg.MODEL.KPTS_NUM)
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        template_fp = os.path.join(cur_dir, '../../template/template.ply')
        transform_fp = os.path.join(cur_dir, '../../template', 'transform.pkl')
        spiral_indices, _, up_transform, tmp = spiral_tramsform(transform_fp,
                                                                template_fp,
                                                                cfg.MODEL.SPIRAL.DOWN_SCALE,
                                                                cfg.MODEL.SPIRAL.LEN,
                                                                cfg.MODEL.SPIRAL.DILATION)
        for i in range(len(up_transform)):
            up_transform[i] = (*up_transform[i]._indices(), up_transform[i]._values())

        self.decoder3d = SequencialReg2DDecode3D(
            cfg.MODEL.LATENT_SIZE, 
            cfg.MODEL.SPIRAL.OUT_CHANNELS, 
            spiral_indices, 
            up_transform, 
            cfg.MODEL.KPTS_NUM,
            meshconv=(SpiralConv, DSConv)[cfg.MODEL.SPIRAL.TYPE=='DSConv'],
            use_global_feat=True
        )

    def forward(self, x):
        '''
        x: (B, F, 3, 128, 128)
        out:
            verts: (B, F, 778, 3)
            joint_img: (B, F, 21, 2)
            joint_conf: (B, F, 21, 1)
        '''
        B, F, _, _, _ = x.shape
        x = rearrange(x, 'B F c h w -> (B F) c h w')

        latent, pred2d_pt = self.backbone(x)  # (#, 256, 4, 4), (#, 21, 3)
        #! NEW frame_len
        pred3d, pred_j3d = self.decoder3d(pred2d_pt, latent, frame_len=F)  # (#, 778, 3)
        # vert, joints

        out = {
            'verts': rearrange(pred3d, '(B F) V D -> B F V D', B=B),
            'joint_img': rearrange(pred2d_pt[:, :, :2], '(B F) J D -> B F J D', B=B),  # (B, F, 21, 2)
            'joint_conf': rearrange(pred2d_pt[:, :, 2:], '(B F) J D -> B F J D', B=B), # (B, F, 21, 1)
        }
        if pred_j3d != []:
            # combine list(tensor) to tensor if exists
            out['joints'] = rearrange(pred_j3d, 'H (B F) J D -> H B F J D', B=B)  # H for each Head output
        return out

    def loss(self, **kwargs):
        '''
        ! ALL in shape (B, F, ...)
        '''
        # reshape
        B = kwargs['verts_gt'].shape[0]
        ## verts
        verts_pred = rearrange(kwargs['verts_pred'], 'B F V D -> (B F) V D')    # (BF, 778, 3)
        verts_gt   = rearrange(kwargs['verts_gt'], 'B F V D -> (B F) V D')      # (BF, 778, 3)
        ## joint
        joint_img_pred  = rearrange(kwargs['joint_img_pred'], 'B F J D -> (B F) J D')   # (BF, 21, 2)
        joint_img_gt    = rearrange(kwargs['joint_img_gt'], 'B F J D -> (B F) J D')     # (BF, 21, 2)
        joint_conf_pred = rearrange(kwargs['joint_conf_pred'], 'B F J D -> (B F) J D')  # (BF, 21, 1)
        _distance = ((joint_img_pred - joint_img_gt)**2).sum(axis=2).sqrt()  # (#, 21)
        _distance = _distance.detach()
        joint_conf_gt = 2 - 2 * torch.sigmoid(_distance * 30)
        ## joint 3d
        if kwargs.get('joint_3d_pred') is not None:
            Headcounts = kwargs['joint_3d_pred'].shape[0]  # (H B F J 3), H for each Head output
            joint_preds = rearrange(kwargs['joint_3d_pred'], 'H B F J D -> H (B F) J D')
            joint_gt = rearrange(kwargs['joint_3d_gt'], 'B F J D -> (B F) J D')

        # compute loss
        loss_dict = dict()

        loss_dict['verts_loss'] = l1_loss(verts_pred, verts_gt)
        loss_dict['joint_img_loss'] = l1_loss(joint_img_pred, joint_img_gt)
        loss_dict['joint_conf_loss'] = 0.1 * l1_loss(joint_conf_pred.view(-1, 21), joint_conf_gt)
        if kwargs.get('joint_3d_pred') is not None:
            joint_3d_pred_loss = 0
            for head_i in range(Headcounts):
                joint_3d_pred_loss += l1_loss(joint_preds[head_i], joint_gt)
            joint_3d_pred_loss /= Headcounts
            loss_dict['joint_3d_loss'] = 0.5 * joint_3d_pred_loss
        
        loss_dict['normal_loss'] = 0.1 * normal_loss(verts_pred, verts_gt, kwargs['face'].to(verts_pred.device))
        loss_dict['edge_loss'] = edge_length_loss(verts_pred, verts_gt, kwargs['face'].to(verts_pred.device))

        loss_dict['loss'] = loss_dict.get('verts_loss', 0) \
                            + loss_dict.get('normal_loss', 0) \
                            + loss_dict.get('edge_loss', 0) \
                            + loss_dict.get('joint_img_loss', 0) \
                            + loss_dict.get('joint_conf_loss', 0) \
                            + loss_dict.get('joint_3d_loss', 0)

        return loss_dict

# my editted version: add transformer, new joint -> global feature
class SequencialReg2DDecode3D(nn.Module):
    def __init__(self, latent_size, out_channels, spiral_indices, up_transform, uv_channel, meshconv=SpiralConv,
                 use_global_feat=True,  # append additional global_feature as one joint
                 # freeze_GCN=True       # freeze GCN model
                 ):
        """Init a 3D decoding with sprial convolution

        Args:
            latent_size (int): feature dim of backbone feature
            out_channels (list): feature dim of each spiral layer
            spiral_indices (list): neighbourhood of each hand vertex
            up_transform (list): upsampling matrix of each hand mesh level
            uv_channel (int): amount of 2D landmark 
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(SequencialReg2DDecode3D, self).__init__()
        self.latent_size = latent_size
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.num_vert = [u[0].size(0)//3 for u in self.up_transform] + [self.up_transform[-1][0].size(0)//6]
        self.uv_channel = uv_channel
        # self.de_layer_conv = conv_layer(self.latent_size, self.out_channels[- 1] -3,  # reserve 3 places for [uvc]
        #                                 1, bn=False, relu=False)
        self.de_layer_conv = conv_layer(self.latent_size, self.out_channels[- 1],
                                        1, bn=False, relu=False)  # bn=False, default
        self.de_layer = nn.ModuleList()
        for idx in range(len(self.out_channels)):
            if idx == 0:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
            else:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
        self.head = meshconv(self.out_channels[0], 3, self.spiral_indices[0])
        self.upsample = nn.Parameter(torch.ones([self.num_vert[-1], self.uv_channel])*0.01, requires_grad=True)
        # 0.01 with shape [49, 21]
        # verts feature: [49, channels] = upsample @ [21, channels]
        # self._freeze_GCN()  # freeze at epoch 10
        self.joint_head = nn.Sequential(
            linear_layer(self.latent_size, 128, bn=False),
            linear_layer(128, 64, bn=False),
            linear_layer(64, 3, bn=False, relu=False)
        )

        # ! NEW
        self.use_global_feat = use_global_feat
        # self.joint_embed = nn.Embedding(21+use_global_feat, self.latent_size)
        self.joint_embed = nn.Embedding(21, self.latent_size)
        self.verts_embed = nn.Embedding(49, self.latent_size)
        self.serial_embed = nn.Embedding(20, self.latent_size)  # max possible frame counts

        # ! EDIT model parameters here
        _ARCH = 'b33'  # b/d/e: base/ de/encoder only, 33: enc & dec layer counts
        _NORM = 'twice'  #  ['twice', 'once', 'first'], once/twice-> norm_last
        _DF = 'FX49F'
        self.transformer = get_transformer(
            self.latent_size, nhead=1, num_encoder_layers=int(_ARCH[1]), num_decoder_layers=int(_ARCH[2]),
            norm_first=   True  if _NORM == 'first' else False,
            NormTwice=    True  if _NORM == 'twice' else False,
            Mode=        'base' if _ARCH[0] == 'b'  else
                 'decoder only' if _ARCH[0] == 'd'  else
                 'encoder only' if _ARCH[0] == 'e'  else 'error',
            # norm_first=False,       # norm first to normalize joint features
            # NormTwice=True,         # norm_first should be False
            # Mode='base',            # ['base', 'encoder only', 'decoder only']

            # in Decoder
            DecoderForwardConfigs = {
                'DecMemUpdate':   'append' if _DF[0  ] == 'A'  else
                                    'full' if _DF[0  ] == 'F'  else 'error', # ['append', 'full']
                'DecMemReplace':     True  if _DF[1  ] == 'R'  else  False,  # [True, False]
                'DecOutCount':  '21 joint' if _DF[2:4] == '21' else 
                                '49 verts' if _DF[2:4] == '49' else
                                 '21 + 49' if _DF[2:4] == '70' else 'error', # ['21 joint', '49 verts', '21 + 49']
                'DecSrcContent':    'zero' if _DF[4  ] == 'Z'  else
                                 'feature' if _DF[4  ] == 'F'  else 'error', # ['zero', 'feature']
                # 'DecMemUpdate': 'full',   # ['append', 'full']
                # 'DecMemReplace': True,      # [True, False]
                # 'DecOutCount': '21 + 49',  # ['21 joint', '49 verts', '21 + 49']
                # 'DecSrcContent': 'feature', # ['zero', 'feature']
            },  # TODO: error occur while DecSrcContent ==  'feature && mode='decoder only'
            matrix=self.upsample,   # (49, 21) mat, used while decoder forwarding from Verts Feature
                                    # the weight won't copied two times
            )
        # self.feature_norm = nn.LayerNorm((21, self.latent_size), eps=1e-5)
        self.feature_norm = nn.LayerNorm(self.latent_size, eps=1e-5)

    def _freeze_GCN(self):
        self.upsample.requires_grad = False
        for k, v in self.head.named_parameters():
            # print(k)
            v.requires_grad = False
        for k, v in self.de_layer.named_parameters():
            # print(k)
            v.requires_grad = False

    def index(self, feat, uv):
        uv = uv.unsqueeze(2)  # [B, N, 1, 2]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        # index features in feat[:,:,i, j]
        # (i, j) = (uv[:,:,1,0], uv[:,:,1,1])
        return samples[:, :, :, 0]  # [B, C, N]

    def get_padding_mask(self, conf: torch.Tensor, ratio=0.2, acceptable_conf=0.5):
        '''
        used in Encoder: src_key_padding_mask
        used in Decoder: memory_key_padding_mask
        in:  conf.shape == (B, J, 1)
        out: mask.shape == (B, J) --> each Joint should be masked or not

        ratio: masked lowest {ratio} joint
        acceptable_conf: masked conf < acceptable_conf

        mask = conf < min(conf.ratio, acceptable_conf)
        mask joint that
            1. joint.conf < hand.conf.ratio
        and 2. joint.conf < acceptable_conf
        '''
        assert conf.shape[2] == 1 and len(conf.shape) == 3, f'conf should be (B, J, 1), get: {conf.shape}'
        B = conf.shape[0]
        conf = rearrange(conf, 'B J () -> B J')
        threshold = torch.minimum(
            torch.quantile(conf, ratio, dim=1, keepdim=True),  # (B, 1)
            torch.ones((B, 1), device=conf.device) * acceptable_conf
        )
        mask = conf < threshold
        return mask

    def show_conf_hist(self, conf):
        from matplotlib import pyplot as plt
        conf = conf.cpu().numpy().flatten()
        plt.hist(conf)
        plt.show()

    def forward(self, uvc, x, frame_len):
        '''
        uvc: [(B F) J 3] -> 3: [u v c]
        x  : [(B F) D H W]

        return [(B F) J 3] -> 3: [x, y, z]
        ---------
        procedure
        uvc -> uv  : (BF, J, 2)
               conf: (BF, J, 1)
        x   -> x   : (BF, new_D -3, H, W)
        glob-> glob: (BF, 1, new_D -3)  # global average pool
        x   -> x   : (BF, J, new_D -3)
        x   -> x   : (BF, J, new_D)     # .cat(uv, conf)
        glob-> glob: (BF, 1, new_D)     # .cat[0, 0, 1], uv:(-1, 1)

        x   -> x   : (B, F, J+1, new_D) # .cat(glob), arrange
        '''
        uv = torch.clamp((uvc[:, :, :2] - 0.5) * 2, -1, 1).detach()  # ! NEW
        conf = torch.clamp(uvc[:, :, 2:], 0, 1).detach()  # ! NEW, [160, 21, 1]
        # self.show_conf_hist(conf)
        padding_mask = self.get_padding_mask(conf)
        uv_embed = uv_encoding(uv, feature_len=self.latent_size // 2)
        # uv_embed = None

        x = self.de_layer_conv(x)  # change channel to self.latent_size
        # (BF, 256, 4, 4) -> (BF, 256 -3, 4, 4)

        x = self.index(x, uv).permute(0, 2, 1)  # [BF, 21, D=256-3]
        # x = torch.cat([x, uv, conf], dim=2)     # [BF, 21, D=256]

        uv_embed = rearrange(uv_embed, '(B F) J D -> B F J D', F=frame_len)
        padding_mask = rearrange(padding_mask, '(B F) J -> B F J', F=frame_len)
        conf = rearrange(conf, '(B F) J () -> B F J', F=frame_len)

        x = rearrange(x, '(B F) J D -> B F J D', F=frame_len)  # ! new
        x = self.feature_norm(x)  # norm (B, features)

        J = x.shape[2]
        x, enc_x = self.transformer(x, joint_embedding=self.joint_embed.weight, verts_embedding=self.verts_embed.weight,
                                    serial_embedding=self.serial_embed.weight, positional_embedding=uv_embed,
                    DiagonalMask = { # mode ,   p
                        'enc self' : ['no', 0.0],   # 'part' of the diag
                        'dec self' : ['no', 0.0],   # 'full' of the diag
                        'dec cross': ['no', 0.0],   # 'no' diag masks are applied
                    },
                    JointConfMask = {  # mask on KEYs
                        'enc self': 'no',           # ['mask', 'weight', 'no']
                        'dec cross': 'no',
                        'joint mask': padding_mask, # [padding_mask, None], (B F J)
                        'joint conf': conf,         # [None,         conf], (B F J)
                    },
                    ReturnEncoderOutput='last'      # ['no', 'last', 'each']
                                                    # =='last', only if Mode=='base
                    )
        x = rearrange(x, 'B F J D -> (B F) J D')  # J or V or J+V
        assert isinstance(enc_x, list), 'enc_x should be list'
        for i in range(len(enc_x)):
            enc_x[i] = rearrange(enc_x[i], 'B F J D -> (B F) J D')

        # separate joint, verts form out
        if self.transformer.DecoderForwardConfigs['DecOutCount'] == '21 + 49':
            out_joint = x[:, :J]
            out_verts = x[:, J:]
            x = out_verts

        # joint predictor
        pred_joint = []
        # pred_joint = [self.joint_head(x)]             # decoder out
        if enc_x != []:  # ReturnEncoderOutput != 'no'
            for enc_i in enc_x:                         # encoder out
                pred_joint += [self.joint_head(enc_i)]
        # pred_joint = [self.joint_head(out_joint)]       # decoder out, DecOutCount == J+V


        # 21joint to 49 verts
        if self.transformer.DecoderForwardConfigs['DecOutCount'] == '21 joint':
            x = torch.bmm(self.upsample.repeat(x.size(0), 1, 1).to(x.device), x)  # (BF 49 D)
        num_features = len(self.de_layer)
        for i, layer in enumerate(self.de_layer):
            x = layer(x, self.up_transform[num_features - i - 1])
        pred = self.head(x)

        return pred, pred_joint
        # return rearrange(pred, '(B F) J D -> B F J D', F=frame_len)


if __name__ == '__main__':
    """Test the model
    """
    from my_research.seq_main import setup
    from options.cfg_options import CFGOptions
    args = CFGOptions().parse()
    args.config_file = 'my_research/configs/mobrecon_ds_conf_transformer.yml'
    cfg = setup(args)

    # model = MobRecon_DS(cfg)
    # model_out = model(torch.zeros(2, 6, 128, 128))
    # print(model_out['verts'].size())

    model = MobRecon_DS_conf_Transformer(cfg)
    # model.eval()
    # (#, 256, 4, 4), (#, 21, 2)
    pred2d_pt, latent = torch.empty(32, 21, 2), torch.empty(32, 256, 7, 7)
    images = torch.empty((4, 8, 3, 128, 128))
    with torch.no_grad():
        # model_out = model(torch.empty(2, 3, 128, 128))
        # model_out = model(pred2d_pt, latent, 8)
        model_out = model(images)
        print(model_out['verts'].size())
