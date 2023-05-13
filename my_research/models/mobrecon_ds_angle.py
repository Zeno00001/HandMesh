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
from my_research.models.densestack import DenseStack_Backnone
# from my_research.models.modules import Reg2DDecode3D
from my_research.models.modules import conv_layer, SpiralDeblock
from my_research.models.loss import l1_loss, normal_loss, edge_length_loss, contrastive_loss_3d, contrastive_loss_2d, bce_wlog_loss
from utils.read import spiral_tramsform
from conv.spiralconv import SpiralConv
from conv.dsconv import DSConv
from my_research.build import MODEL_REGISTRY
from einops import rearrange


@MODEL_REGISTRY.register()
class MobRecon_DS_Angle(nn.Module):
    def __init__(self, cfg):
        """Init a MobRecon-DenseStack model

        Args:
            cfg : config file
        """
        super(MobRecon_DS_Angle, self).__init__()
        self.cfg = cfg
        self.backbone = DenseStack_Backnone(latent_size=cfg.MODEL.LATENT_SIZE,
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
        self.freeze()

    def freeze(self):
        self.__freeze(self.backbone)
        self.__freeze(self.decoder3d.de_layer_conv)
        self.__freeze(self.decoder3d.upsample)
        self.__freeze(self.decoder3d.de_layer)  # GCN
        self.__freeze(self.decoder3d.head)      # GCN

    def __freeze(self, mod: nn.parameter.Parameter):
        if isinstance(mod, nn.Module):
            for k, v in mod.named_parameters():
                v.requires_grad = False
        elif isinstance(mod, nn.parameter.Parameter):
            mod.requires_grad = False

    def _eval_bn_layers(self):
        for v in self.modules():
            if isinstance(v, nn.BatchNorm2d) or isinstance(v, nn.BatchNorm1d):
                v.eval()

    def forward(self, x):
        if x.size(1) == 6:
            pred3d_list = []
            pred2d_pt_list = []
            negativeness_list = []
            for i in range(2):
                latent, pred2d_pt = self.backbone(x[:, 3*i:3*i+3])
                pred3d, negativeness = self.decoder3d(pred2d_pt, latent)  # (#, 778, 3), (#, 2)
                pred3d_list.append(pred3d)
                pred2d_pt_list.append(pred2d_pt)
                negativeness_list.append(negativeness)
            pred2d_pt = torch.cat(pred2d_pt_list, -1)  # (#, 256, 4, 4), (#, 21, 2  *2)
            pred3d = torch.cat(pred3d_list, -1)                        # (#, 778, 3  *2)
            negativeness = rearrange(negativeness_list, 'Cont B H -> B (H Cont)')  # (Cont B 2) -> (B 4)
                                                        # [head1 cont1, head1 cont2, head2 cont1, head2 cont2]
        else:
            latent, pred2d_pt = self.backbone(x)  # (#, 256, 4, 4), (#, 21, 2)
            pred3d, negativeness = self.decoder3d(pred2d_pt, latent)            # (#, 778, 3), (#, 2)

        return {'verts': pred3d,
                'joint_img': pred2d_pt,
                'negative': negativeness,
                }

    def loss(self, **kwargs):
        loss_dict = dict()

        loss_dict['verts_loss'] = l1_loss(kwargs['verts_pred'], kwargs['verts_gt'])
        # loss_dict['joint_img_loss'] = l1_loss(kwargs['joint_img_pred'], kwargs['joint_img_gt'])
        if self.cfg.DATA.CONTRASTIVE:
            # loss_dict['normal_loss'] = 0.05 * (normal_loss(kwargs['verts_pred'][..., :3], kwargs['verts_gt'][..., :3], kwargs['face']) + \
            #                                    normal_loss(kwargs['verts_pred'][..., 3:], kwargs['verts_gt'][..., 3:], kwargs['face']))
            # loss_dict['edge_loss'] = 0.5 * (edge_length_loss(kwargs['verts_pred'][..., :3], kwargs['verts_gt'][..., :3], kwargs['face']) + \
            #                                 edge_length_loss(kwargs['verts_pred'][..., 3:], kwargs['verts_gt'][..., 3:], kwargs['face']))
            # if kwargs['aug_param'] is not None:
            #     loss_dict['con3d_loss'] = contrastive_loss_3d(kwargs['verts_pred'], kwargs['aug_param'])
            #     loss_dict['con2d_loss'] = contrastive_loss_2d(kwargs['joint_img_pred'], kwargs['bb2img_trans'], kwargs['size'])
            assert kwargs['negative_pred'] is not None
            total_counts = kwargs['negative_pred'].shape[1]  # (Batch, Head * Contrastive)
            loss_dict['negative_thumb_loss'] = 0
            # head_weight = [0.25, 0.25, 0.25, 0.25]
            head_weight = [0.5, 0.5]
            # assert sum(head_weight) == 1.0
            for h_idx in range(len(head_weight)):
                for i in range(2):
                    idx = h_idx*2 + i
                    loss_dict['negative_thumb_loss'] += 0.5 * head_weight[h_idx] * \
                                                        bce_wlog_loss(kwargs['negative_pred'][:, idx], kwargs['negative_gt'])
        else:
            loss_dict['normal_loss'] = 0.1 * normal_loss(kwargs['verts_pred'], kwargs['verts_gt'], kwargs['face'].to(kwargs['verts_pred'].device))
            loss_dict['edge_loss'] = edge_length_loss(kwargs['verts_pred'], kwargs['verts_gt'], kwargs['face'].to(kwargs['verts_pred'].device))

        loss_dict['loss'] = loss_dict.get('verts_loss', 0) \
                            + loss_dict.get('normal_loss', 0) \
                            + loss_dict.get('edge_loss', 0) \
                            + loss_dict.get('joint_img_loss', 0) \
                            + loss_dict.get('con3d_loss', 0) \
                            + loss_dict.get('con2d_loss', 0) \
                            + loss_dict.get('negative_thumb_loss', 0)

        return loss_dict

class AngleNegativePredictionHead(nn.Module):
    def __init__(self, verts=98, dim=256):
        '''
            input:  the output of GCN first layer (and laters): (49 256 -> 98 256 -> 195 128 -> 389 64 -> 778 32)
                    (B, verts, dim), verts := 98, dim := 256

            output: the negativeness of the thumb: (B, 1)
        '''
        super(AngleNegativePredictionHead, self).__init__()
        dim_reduce_table = [dim, dim//4, dim//16]  # 256, 64, 16 or 128, 32, 8
        self.linear1 = nn.Linear(dim_reduce_table[0], dim_reduce_table[1])  # compress Channel-dim
        # self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(dim_reduce_table[1], dim_reduce_table[2])
        # self.relu2 = nn.ReLU()

        # Reshape to (verts * Dim)
        overall_dim = verts * dim_reduce_table[2]  # total dim of verts * dim
        self.linear3 = nn.Linear(overall_dim, overall_dim // 4)  # 98 * 16 -> ...98 * 4
        self.linear4 = nn.Linear(overall_dim // 4, overall_dim // 16)  # 98 * 4 -> 98 |
        self.linear5 = nn.Linear(overall_dim // 16, 1)

        self.dropout1 = nn.Dropout(p=0.5, inplace=True)
        self.dropout2 = nn.Dropout(p=0.5, inplace=True)
        self.dropout3 = nn.Dropout(p=0.5, inplace=True)

        # ver.0:
        # Linear(256, 16) -> ReLU -> Reshape(to verts * 16) -> Linear(*, 1)

    def forward(self, x):
        x = self.linear1(x)     # (98 256) -> (98 64)
        x = self.dropout1(x)
        x = self.linear2(x)     # (98 64) -> (98 16)
        x = rearrange(x, 'B V C -> B (V C)')
        x = self.dropout2(x)
        x = self.linear3(x)     # (98*16) -> (98*4)
        x = self.dropout3(x)
        x = self.linear4(x)     # (98*4) -> (98*1)
        x = self.linear5(x)     # (98) -> (1)
        return x


# Advanced modules, adopt from `modules.py`
class Reg2DDecode3D(nn.Module):
    def __init__(self, latent_size, out_channels, spiral_indices, up_transform, uv_channel, meshconv=SpiralConv):
        """Init a 3D decoding with sprial convolution

        Args:
            latent_size (int): feature dim of backbone feature
            out_channels (list): feature dim of each spiral layer
            spiral_indices (list): neighbourhood of each hand vertex
            up_transform (list): upsampling matrix of each hand mesh level
            uv_channel (int): amount of 2D landmark 
            meshconv (optional): conv method, supporting SpiralConv, DSConv. Defaults to SpiralConv.
        """
        super(Reg2DDecode3D, self).__init__()
        self.latent_size = latent_size
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.num_vert = [u[0].size(0)//3 for u in self.up_transform] + [self.up_transform[-1][0].size(0)//6]
        self.uv_channel = uv_channel
        self.de_layer_conv = conv_layer(self.latent_size, self.out_channels[- 1], 1, bn=False, relu=False)
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

        # verts: [49, 98, 195, 389, 778]
        self.negative_head1 = AngleNegativePredictionHead(verts=98, dim=256)
        self.negative_head2 = AngleNegativePredictionHead(verts=195, dim=128)
        # self.negative_head3 = AngleNegativePredictionHead(verts=389, dim=64)
        # self.negative_head4 = AngleNegativePredictionHead(verts=778, dim=32)


    def index(self, feat, uv):
        uv = uv.unsqueeze(2)  # [B, N, 1, 2]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        # index features in feat[:,:,i, j]
        # (i, j) = (uv[:,:,1,0], uv[:,:,1,1])
        return samples[:, :, :, 0]  # [B, C, N]

    def forward(self, uv, x):
        uv = torch.clamp((uv - 0.5) * 2, -1, 1)
        x = self.de_layer_conv(x)
        x = self.index(x, uv).permute(0, 2, 1)  # [B, 21, channel]
        x = torch.bmm(self.upsample.repeat(x.size(0), 1, 1).to(x.device), x)
        num_features = len(self.de_layer)
        negativeness_list = []
        for i, layer in enumerate(self.de_layer):
            x = layer(x, self.up_transform[num_features - i - 1])
            # print(x.shape)
            # after 1st GCN layer
            if i == 0:
                negativeness_list += [self.negative_head1(x)]
            elif i == 1:
                negativeness_list += [self.negative_head2(x)]
            # elif i == 2:
            #     negativeness_list += [self.negative_head3(x)]
            # elif i == 3:
            #     negativeness_list += [self.negative_head4(x)]
            # B, [49, 98, 195, 389, 778], [256, 256, 128, 64, 32]
        pred = self.head(x)

        return pred, rearrange(negativeness_list, 'H B () -> B H')  # Batch, Head

if __name__ == '__main__':
    """Test the model
    """
    from my_research.main import setup
    from options.cfg_options import CFGOptions
    args = CFGOptions().parse()
    args.config_file = 'my_research/configs/mobrecon_ds_angle.yml'
    cfg = setup(args)

    model = MobRecon_DS_Angle(cfg)
    model.eval()
    model_out = model(torch.zeros(2, 6, 128, 128))
    print(model_out['negativeness'].size())

