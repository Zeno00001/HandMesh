# Copyright (c) Xingyu Chen. All Rights Reserved.

"""
 * @file mobrecon_ds.py
 * @author chenxingyu (chenxy.sean@gmail.com)
 * @brief MobRecon model 
 * @version 0.1
 * @date 2022-04-28
 * 
 * @copyright Copyright (c) 2022 chenxingyu
 * 
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import torch
from einops import rearrange
from my_research.models.densestack import DenseStack_Backnone
from my_research.models.densestack_conf import DenseStack_Conf_Backbone
from my_research.models.modules import Reg2DDecode3D
from my_research.models.loss import l1_loss, normal_loss, edge_length_loss, contrastive_loss_3d, contrastive_loss_2d
from utils.read import spiral_tramsform
from conv.spiralconv import SpiralConv
from conv.dsconv import DSConv
from my_research.build import MODEL_REGISTRY

# ! NOTICE that: using UVC backbone, but no loss is applied on Confidence
@MODEL_REGISTRY.register()
class MobRecon_DS_SEQ(nn.Module):
    def __init__(self, cfg):
        """Init a MobRecon-DenseStack model

        Args:
            cfg : config file
        """
        super(MobRecon_DS_SEQ, self).__init__()
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
        self.decoder3d = Reg2DDecode3D(cfg.MODEL.LATENT_SIZE, 
                                       cfg.MODEL.SPIRAL.OUT_CHANNELS, 
                                       spiral_indices, 
                                       up_transform, 
                                       cfg.MODEL.KPTS_NUM,
                                       meshconv=(SpiralConv, DSConv)[cfg.MODEL.SPIRAL.TYPE=='DSConv'])

    def forward(self, x):
        # latent, pred2d_pt = self.backbone(x)  # (#, 256, 4, 4), (#, 21, 2)
        # pred3d = self.decoder3d(pred2d_pt, latent)            # (#, 778, 3)
        B, F, _, _, _ = x.shape
        x = rearrange(x, 'B F c h w -> (B F) c h w')
        # ? remove Frame dimension

        latent, pred2d_pt, feat8x8 = self.backbone(x)
        pred3d = self.decoder3d(pred2d_pt[:, :, :2], latent)

        out = {
            'verts': rearrange(pred3d, '(B F) V D -> B F V D', B=B),
            'joint_img': rearrange(pred2d_pt[:, :, :2], '(B F) J D -> B F J D', B=B),  # (B, F, 21, 2)
            # 'joint_conf': rearrange(pred2d_pt[:, :, 2:], '(B F) J D -> B F J D', B=B), # (B, F, 21, 1)
        }

        return out

    def loss(self, **kwargs):
        # reshape
        B = kwargs['verts_gt'].shape[0]
        ## verts
        verts_pred = rearrange(kwargs['verts_pred'], 'B F V D -> (B F) V D')    # (BF, 778, 3)
        verts_gt   = rearrange(kwargs['verts_gt'], 'B F V D -> (B F) V D')      # (BF, 778, 3)
        ## joint
        joint_img_pred  = rearrange(kwargs['joint_img_pred'], 'B F J D -> (B F) J D')   # (BF, 21, 2)
        joint_img_gt    = rearrange(kwargs['joint_img_gt'], 'B F J D -> (B F) J D')     # (BF, 21, 2)

        loss_dict = dict()

        loss_dict['verts_loss'] = l1_loss(verts_pred, verts_gt)
        loss_dict['joint_img_loss'] = l1_loss(joint_img_pred, joint_img_gt)
        # if self.cfg.DATA.CONTRASTIVE:
        #     loss_dict['normal_loss'] = 0.05 * (normal_loss(kwargs['verts_pred'][..., :3], kwargs['verts_gt'][..., :3], kwargs['face']) + \
        #                                        normal_loss(kwargs['verts_pred'][..., 3:], kwargs['verts_gt'][..., 3:], kwargs['face']))
        #     loss_dict['edge_loss'] = 0.5 * (edge_length_loss(kwargs['verts_pred'][..., :3], kwargs['verts_gt'][..., :3], kwargs['face']) + \
        #                                     edge_length_loss(kwargs['verts_pred'][..., 3:], kwargs['verts_gt'][..., 3:], kwargs['face']))
        #     if kwargs['aug_param'] is not None:
        #         loss_dict['con3d_loss'] = contrastive_loss_3d(kwargs['verts_pred'], kwargs['aug_param'])
        #         loss_dict['con2d_loss'] = contrastive_loss_2d(kwargs['joint_img_pred'], kwargs['bb2img_trans'], kwargs['size'])
        # else:
        loss_dict['normal_loss'] = 0.1 * normal_loss(verts_pred, verts_gt, kwargs['face'].to(verts_pred.device))
        loss_dict['edge_loss'] = edge_length_loss(verts_pred, verts_gt, kwargs['face'].to(verts_pred.device))

        loss_dict['loss'] = loss_dict.get('verts_loss', 0) \
                            + loss_dict.get('normal_loss', 0) \
                            + loss_dict.get('edge_loss', 0) \
                            + loss_dict.get('joint_img_loss', 0) \
                            + loss_dict.get('con3d_loss', 0) \
                            + loss_dict.get('con2d_loss', 0)

        return loss_dict

class Test(nn.Module):
    def __init__(self, cfg):
        super(Test, self).__init__()
        self.cfg = cfg
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
        self.decoder3d = Reg2DDecode3D(cfg.MODEL.LATENT_SIZE, 
                                       cfg.MODEL.SPIRAL.OUT_CHANNELS, 
                                       spiral_indices, 
                                       up_transform, 
                                       cfg.MODEL.KPTS_NUM,
                                       meshconv=(SpiralConv, DSConv)[cfg.MODEL.SPIRAL.TYPE=='DSConv'])

    def forward(self, pred2d_pt, latent):   # (#, 256, 4, 4), (#, 21, 2)
        pred3d = self.decoder3d(pred2d_pt, latent)            # (#, 778, 3)

        return {'verts': pred3d,
                'joint_img': pred2d_pt
                }

if __name__ == '__main__':
    """Test the model
    """
    from my_research.main import setup
    from options.cfg_options import CFGOptions
    args = CFGOptions().parse()
    args.config_file = 'my_research/configs/mobrecon_ds.yml'
    cfg = setup(args)

    # model = MobRecon_DS(cfg)
    # model_out = model(torch.zeros(2, 6, 128, 128))
    # print(model_out['verts'].size())

    model = Test(cfg)
    model.eval()
    # (#, 256, 4, 4), (#, 21, 2)
    pred2d_pt, latent = torch.empty(32, 21, 2), torch.empty(32, 256, 7, 7)
    with torch.no_grad():
        # model_out = model(torch.empty(2, 3, 128, 128))
        model_out = model(pred2d_pt, latent)
        print(model_out['verts'].size())
