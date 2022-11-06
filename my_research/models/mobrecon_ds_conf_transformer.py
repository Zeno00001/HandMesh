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
from my_research.models.modules import SpiralDeblock, conv_layer
from my_research.models.transformer import get_transformer

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
        pred3d = self.decoder3d(pred2d_pt, latent, frame_len=F)  # (#, 778, 3)

        return {'verts': rearrange(pred3d, '(B F) V D -> B F V D', B=B),
                'joint_img': rearrange(pred2d_pt[:, :, :2], '(B F) J D -> B F J D', B=B),  # (B, F, 21, 2)
                'joint_conf': rearrange(pred2d_pt[:, :, 2:], '(B F) J D -> B F J D', B=B)  # (B, F, 21, 1)
                }

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

        # compute loss
        loss_dict = dict()

        loss_dict['verts_loss'] = l1_loss(verts_pred, verts_gt)
        loss_dict['joint_img_loss'] = l1_loss(joint_img_pred, joint_img_gt)
        loss_dict['joint_conf_loss'] = 0.1 * l1_loss(joint_conf_pred.view(-1, 21), joint_conf_gt)
        
        loss_dict['normal_loss'] = 0.1 * normal_loss(verts_pred, verts_gt, kwargs['face'].to(verts_pred.device))
        loss_dict['edge_loss'] = edge_length_loss(verts_pred, verts_gt, kwargs['face'].to(verts_pred.device))

        loss_dict['loss'] = loss_dict.get('verts_loss', 0) \
                            + loss_dict.get('normal_loss', 0) \
                            + loss_dict.get('edge_loss', 0) \
                            + loss_dict.get('joint_img_loss', 0) \
                            + loss_dict.get('joint_conf_loss')

        return loss_dict

# my editted version: add transformer, new joint -> global feature
class SequencialReg2DDecode3D(nn.Module):
    def __init__(self, latent_size, out_channels, spiral_indices, up_transform, uv_channel, meshconv=SpiralConv,
                 use_global_feat=True  # append additional global_feature as one joint
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
        self.de_layer_conv = conv_layer(self.latent_size, self.out_channels[- 1] -3,  # reserve 3 places for [uvc]
                                        1, bn=False, relu=False)
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

        # ! NEW
        self.use_global_feat = use_global_feat
        self.joint_embed = nn.Embedding(21+use_global_feat, self.latent_size)
        self.serial_embed = nn.Embedding(20, self.latent_size)  # max possible frame counts
        self.transformer = get_transformer(self.latent_size, nhead=4, num_encoder_layers=3, num_decoder_layers=3)


    def index(self, feat, uv):
        uv = uv.unsqueeze(2)  # [B, N, 1, 2]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        # index features in feat[:,:,i, j]
        # (i, j) = (uv[:,:,1,0], uv[:,:,1,1])
        return samples[:, :, :, 0]  # [B, C, N]

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
        uv = torch.clamp((uvc[:, :, :2] - 0.5) * 2, -1, 1)  # ! NEW
        conf = torch.clamp(uvc[:, :, 2:], 0, 1)  # ! NEW

        x = self.de_layer_conv(x)  # change channel to self.latent_size
        # (BF, 256, 4, 4) -> (BF, 256 -3, 4, 4)


        #? if self.use_global_feat:
        global_feat = reduce(x, 'BF D H W -> BF () D', 'mean')  # ! new, D=256-3
        _feat_uvc = torch.tensor([0, 0, 1], device=x.device)
        _feat_uvc = repeat(_feat_uvc, 'D -> BF () D', BF=global_feat.shape[0])  # [BF, 1, 3]
        global_feat = torch.cat([global_feat, _feat_uvc], dim=2)  # [BF, 1, 256]

        x = self.index(x, uv).permute(0, 2, 1)  # [BF, 21, D=256-3]
        x = torch.cat([x, uv, conf], dim=2)     # [BF, 21, D=256]

        x = torch.cat([x, global_feat], dim=1)  # [BF, J=22, D]

        x = rearrange(x, '(B F) J D -> B F J D', F=frame_len)  # ! new
        # accept [B, F, J, D]
        x = self.transformer(x, joint_embedding=self.joint_embed.weight, serial_embedding=self.serial_embed.weight)

        #? if self.use_global_feat:
        x = x[:, :, :21, :]  # [B, F, J, D], set J to 21

        x = rearrange(x, 'B F J D -> (B F) J D')

        # 21joint to 49 verts
        x = torch.bmm(self.upsample.repeat(x.size(0), 1, 1).to(x.device), x)
        num_features = len(self.de_layer)
        for i, layer in enumerate(self.de_layer):
            x = layer(x, self.up_transform[num_features - i - 1])
        pred = self.head(x)

        return pred
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
    model.eval()
    # (#, 256, 4, 4), (#, 21, 2)
    pred2d_pt, latent = torch.empty(32, 21, 2), torch.empty(32, 256, 7, 7)
    images = torch.empty((4, 8, 3, 128, 128))
    with torch.no_grad():
        # model_out = model(torch.empty(2, 3, 128, 128))
        # model_out = model(pred2d_pt, latent, 8)
        model_out = model(images)
        print(model_out['verts'].size())
