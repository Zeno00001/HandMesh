import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import torch
from my_research.models.resnetstack import ResnetStack_Backbone, Bottleneck
from my_research.models.modules import Reg2DDecode3D
from my_research.models.loss import l1_loss, normal_loss, edge_length_loss, contrastive_loss_3d, contrastive_loss_2d
from utils.read import spiral_tramsform
from conv.spiralconv import SpiralConv
from conv.dsconv import DSConv
from my_research.build import MODEL_REGISTRY

'''
Almost same to mobrecon_ds.py
the only difference is 'ResnetStack_Backbone'
'''

@MODEL_REGISTRY.register()
class MobRecon_RS(nn.Module):
    def __init__(self, cfg):
        """Init a MobRecon-ResnetStack model

        Args:
            cfg : config file
        """
        super(MobRecon_RS, self).__init__()
        self.cfg = cfg
        self.backbone = ResnetStack_Backbone(block=Bottleneck,
                                             layers=[3, 4, 6, 3],  # ResNet50 Layer counts
                                             latent_size=cfg.MODEL.LATENT_SIZE)
                                            #  kpts_num=cfg.MODEL.KPTS_NUM
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
        if x.size(1) == 6:
            pred3d_list = []
            pred2d_pt_list = []
            for i in range(2):
                latent, pred2d_pt = self.backbone(x[:, 3*i:3*i+3])
                pred3d = self.decoder3d(pred2d_pt, latent)
                pred3d_list.append(pred3d)
                pred2d_pt_list.append(pred2d_pt)
            pred2d_pt = torch.cat(pred2d_pt_list, -1)  # (#, 256, 4, 4), (#, 21, 2  *2)
            pred3d = torch.cat(pred3d_list, -1)                        # (#, 778, 3  *2)
        else:
            latent, pred2d_pt = self.backbone(x)  # (#, 256, 4, 4), (#, 21, 2)
            pred3d = self.decoder3d(pred2d_pt, latent)            # (#, 778, 3)

        return {'verts': pred3d,
                'joint_img': pred2d_pt
                }

    def loss(self, **kwargs):
        loss_dict = dict()

        loss_dict['verts_loss'] = l1_loss(kwargs['verts_pred'], kwargs['verts_gt'])
        loss_dict['joint_img_loss'] = l1_loss(kwargs['joint_img_pred'], kwargs['joint_img_gt'])
        if self.cfg.DATA.CONTRASTIVE:
            loss_dict['normal_loss'] = 0.05 * (normal_loss(kwargs['verts_pred'][..., :3], kwargs['verts_gt'][..., :3], kwargs['face']) + \
                                               normal_loss(kwargs['verts_pred'][..., 3:], kwargs['verts_gt'][..., 3:], kwargs['face']))
            loss_dict['edge_loss'] = 0.5 * (edge_length_loss(kwargs['verts_pred'][..., :3], kwargs['verts_gt'][..., :3], kwargs['face']) + \
                                            edge_length_loss(kwargs['verts_pred'][..., 3:], kwargs['verts_gt'][..., 3:], kwargs['face']))
            if kwargs['aug_param'] is not None:
                loss_dict['con3d_loss'] = contrastive_loss_3d(kwargs['verts_pred'], kwargs['aug_param'])
                loss_dict['con2d_loss'] = contrastive_loss_2d(kwargs['joint_img_pred'], kwargs['bb2img_trans'], kwargs['size'])
        else:
            loss_dict['normal_loss'] = 0.1 * normal_loss(kwargs['verts_pred'], kwargs['verts_gt'], kwargs['face'].to(kwargs['verts_pred'].device))
            loss_dict['edge_loss'] = edge_length_loss(kwargs['verts_pred'], kwargs['verts_gt'], kwargs['face'].to(kwargs['verts_pred'].device))

        loss_dict['loss'] = loss_dict.get('verts_loss', 0) \
                            + loss_dict.get('normal_loss', 0) \
                            + loss_dict.get('edge_loss', 0) \
                            + loss_dict.get('joint_img_loss', 0) \
                            + loss_dict.get('con3d_loss', 0) \
                            + loss_dict.get('con2d_loss', 0)

        return loss_dict

if __name__ == '__main__':
    """Test the model
    """
    from my_research.main import setup
    from options.cfg_options import CFGOptions
    args = CFGOptions().parse()
    args.config_file = 'my_research/configs/mobrecon_rs.yml'
    cfg = setup(args)

    model = MobRecon_RS(cfg)
    model_out = model(torch.zeros(2, 6, 128, 128))
    print(model_out['verts'].size())
