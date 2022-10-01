# Copyright (c) Xingyu Chen. All Rights Reserved.
# Edited from HandMesh/mobrecon/datasets/freihand.py
'''
--- Raw Data Structure ---

HanCo/
    rgb/                | images    | 14.1 G
    rgb_color_auto/     | images    | 19.0 G
    rgb_color_sample/   | images    | 19.1 G
    rgb_merged/         | images    | 20.2 G
    rgb_homo/           | images    | 7.8 G

    numpy_seq/          | npz files: verts, [joint, global_t], [intrinsic]
    mesh/               | ply files: verts
    numpy/              | npz files: verts, [joint, global_t], [intrinsic]
    mesh_intrinsic/     | npz files: intrinsic, support for ply files

contains:
    numpy_seq = {
        'verts':    [#, 778, 3],
        'joint':    [#, 21,  3], (optional)
        'global_t:  [#, 1,   3], (optional)
        'intrinsic':[#, 3,   3], (optional)
    }
    numpy = { ... }
    ply = { ... }

--- Descriptions of Criterions --- Decide Train/Test set ---

1. is_train                 : if a sequence is under a green background
2. os.listdir(rgb_augment)  : 4 augment folder are the same
                            : subset of is_train
3. sum(has_fit) == len(seq) : if a sequence has mano fit in ALL frames

Training set: Crit 2 & Crit 3
    Crit 2: fas full augment data images
    Crit 3: has full MANO parameters

Testing set : Crit 3 - (Crit 2) == Crit 3 & (not Crit 2)
    Crit 3 & (Crit 1 - Crit 2): has green background BUT not have full augment -> testing set
    Crit 3 & (not Crit 1)     : not have green background & full augment       -> testing set

--- FORMAT ---

| ------------------  rgb/  ------------------- | ---------------  rgb_merged/  --------------- | rgb_color_asample/ | rgb_color_auto/ | 
| train sequence 1  | train sequence 2  | ...   | train sequence 1  | train sequence 2  | ...   |
|   cam0 - cam7     |   cam0 - cam7     | ...   |   cam0 - cam7     |   cam0 - cam7     | ...   |


folders  [0:5]      = ['rgb', 'rgb_color_auto', 'rgb_color_sample', 'rgb_homo', 'rgb_merged']
train_seq[0:1193]   = [0, 2, ...], skip some invalid sequences
cam_id   [0:8]      = [0, ..., 7]

dataset[i]
|       | index -> folder, seq_id, cam_id | folder, seq_id, cam_id -> index |
| ----- | ------------------------------- | ------------------------------- |
| Train | i / [len(train_seq) * 8],       | folder_id * [len(train_seq) *8] |
|       | i % [len(train_seq) * 8] / 8,   | + seq_id * 8 + cam_id           |
|       | i % 8                           |                                 |
| ----- | ------------------------------- | ------------------------------- |
| Test  | i / 8, i % 8                    | seq_id * 8 + cam_id             |
| ----- | ------------------------------- | ------------------------------- |
'''

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import torch
import torch.utils.data as data
import numpy as np
from utils.fh_utils import load_db_annotation, read_mesh, read_img, read_img_abs, read_mask_woclip, projectPoints
from utils.vis import base_transform, inv_base_tranmsform, cnt_area
import cv2
from utils.augmentation import Augmentation
from termcolor import cprint
from utils.preprocessing import augmentation, augmentation_2d, trans_point2d
from mobrecon.tools.kinematics import MPIIHandJoints
from mobrecon.models.loss import contrastive_loss_3d, contrastive_loss_2d
import vctoolkit as vc
from mobrecon.build import DATA_REGISTRY

@DATA_REGISTRY.register()
class HanCo(data.Dataset):
    def __init__(self, cfg, phase='train') -> None:
        '''Init a FreiHAND Dataset

        Args:
            cfg : config file
            phase (str, optional): train or eval. Defaults to 'train'.
        '''
        super(HanCo, self).__init__()
        self.cfg = cfg
        self.phase = phase
        self.train_seq, self.test_seq = self._get_valid_sequence(
            hanco_root=self.cfg.DATA.HANCO.ROOT, DEBUG=False
        )
        self.image_aug = ['rgb', 'rgb_color_auto', 'rgb_color_sample', 'rgb_homo', 'rgb_merged']
        self.len_train_seq_cam = len(self.train_seq) * 8  # (sequence X cam) counts of all augment folders

        self.color_aug = Augmentation() if cfg.DATA.COLOR_AUG and 'train' in self.phase else None

    def _get_valid_sequence(self, hanco_root, DEBUG):
        '''Compute valid sequence from meta.json
        return train_list, test_list

        Details: see Top[Descriptions of Criterions, Decide Train/Test set]
        '''
        meta_file = os.path.join(hanco_root, 'meta.json')
        with open(meta_file, 'r') as file:
            meta_data = json.load(file)
        # keys are: 'is_train': 綠幕, 'subject_id': 人物, 'is_valid': MANO 有人工檢驗過,
        #           'object_id': 手握                  , 'has_fit':  有 MANO 參數

        # Crit 3: has MANO fit
        hasfit_seq = []
        for i, seq in enumerate(meta_data['has_fit']):
            ''' seq: List[bool] '''
            if sum(seq) == len(seq):
                hasfit_seq += [i]

        # Crit 2: has full augment images
        #         all augment have same sequences, so check 1 type is enough
        dir_rgb_color_auto = os.path.join(hanco_root, 'rgb_color_auto')

        full_aug_seq = [int(folder) for folder in os.listdir(dir_rgb_color_auto)]

        set_hasfit_seq = set(hasfit_seq)
        set_full_aug_seq = set(full_aug_seq)

        # Train / Test set
        train_set = set_hasfit_seq & set_full_aug_seq
        test_set  = set_hasfit_seq - set_full_aug_seq

        # Check
        dir_mask = os.path.join(hanco_root, 'mask_hand')
        set_mask_seq = set(int(folder) for folder in os.listdir(dir_mask))

        assert train_set & test_set == set(), '(X) train & test set should have NO intersection'
        assert all(0 <= e < 1518 for e in train_set), '(X) index in train_set, not in range(0, 1518)'
        assert all(0 <= e < 1518 for e in test_set),  '(X) index in test_set , not in range(0, 1518)'
        assert train_set <= set_full_aug_seq, '(X) train_set ALL have full augment images(rgb_color_auto, ...)'
        # assert test_set <= set(rgb_list)
        assert train_set <= set_hasfit_seq, '(X) train_set ALL have MANO fit'
        assert test_set  <= set_hasfit_seq, '(X) test_set  ALL have MANO fit'
        assert train_set <= set_mask_seq, '(X) train_set ALL have mask_hand'
        assert test_set  <= set_mask_seq, '(X) test_set  ALL have mask_hand'

        print('Pass checks')

        if DEBUG:
            print('train and test set...')
            print('> 1. have no intersection')
            print('> 2. Both are in valid sequence range(0, 1518), for all seq, frames')
            print('> 3. TRAIN set have all augment images')
            print('> 4. Both have MANO fit, for all seq, frames')
            print('> 5. Both have mask_hand, for all seq, frames')

        train_list = sorted(list(train_set))
        test_list = sorted(list(test_set))
        print(f'Train({len(train_list)}): {train_list[:5]}, ...', end=', ')
        print(f'Test({len(test_list)}):   {test_list[:5]}...')
        return train_list, test_list

    def __len__(self):
        if self.phase == 'train':
            return len(self.image_aug) * len(self.train_seq) * 8
        elif self.phase == 'test':
            return len(self.test_seq) * 8

    def __getitem__(self, index):
        if self.phase == 'train':
            aug_folder = index / self.len_train_seq_cam
            seq_id = self.train_seq[index % self.len_train_seq_cam] # No. ? in train_seq
            cam_id = index % 8
        elif self.phase == 'test':
            seq_id = index / 8
            cam_id = index % 8

@DATA_REGISTRY.register()
class FreiHAND(data.Dataset):
    def __init__(self, cfg, phase='train', writer=None):
        """Init a FreiHAND Dataset

        Args:
            cfg : config file
            phase (str, optional): train or eval. Defaults to 'train'.
            writer (optional): log file. Defaults to None.
        """
        super(FreiHAND, self).__init__()
        self.cfg = cfg
        self.phase = phase
        self.db_data_anno = tuple(load_db_annotation(self.cfg.DATA.FREIHAND.ROOT, set_name=self.phase))
        self.color_aug = Augmentation() if cfg.DATA.COLOR_AUG and 'train' in self.phase else None
        self.one_version_len = len(self.db_data_anno)
        if 'train' in self.phase:
            self.db_data_anno *= 4
        if writer is not None:
            writer.print_str('Loaded FreiHand {} {} samples'.format(self.phase, str(len(self.db_data_anno))))
        cprint('Loaded FreiHand {} {} samples'.format(self.phase, str(len(self.db_data_anno))), 'red')

    def __getitem__(self, idx):  # True
        if 'train' in self.phase:
            if self.cfg.DATA.CONTRASTIVE:
                return self.get_contrastive_sample(idx)
            else:
                return self.get_training_sample(idx)
        elif 'eval' in self.phase or 'test' in self.phase:
            return self.get_eval_sample(idx)
        else:
            raise Exception('phase error')

    def get_contrastive_sample(self, idx):
        """Get contrastive FreiHAND samples for consistency learning
        """
        # read
        img = read_img_abs(idx, self.cfg.DATA.FREIHAND.ROOT, 'training')
        vert = read_mesh(idx % self.one_version_len, self.cfg.DATA.FREIHAND.ROOT).x.numpy()
        mask = read_mask_woclip(idx % self.one_version_len, self.cfg.DATA.FREIHAND.ROOT, 'training')
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        contours.sort(key=cnt_area, reverse=True)
        bbox = cv2.boundingRect(contours[0])
        center = [bbox[0]+bbox[2]*0.5, bbox[1]+bbox[3]*0.5]
        w, h = bbox[2], bbox[3]
        bbox = [center[0]-0.5 * max(w, h), center[1]-0.5 * max(w, h), max(w, h), max(w, h)]
        K, mano, joint_cam = self.db_data_anno[idx]
        K, joint_cam, mano = np.array(K), np.array(joint_cam), np.array(mano)
        # print(f'K:\n{K}')
        joint_img = projectPoints(joint_cam, K)
        princpt = K[0:2, 2].astype(np.float32)
        focal = np.array( [K[0, 0], K[1, 1]], dtype=np.float32)
        # multiple aug
        roi_list = []
        calib_list = []
        mask_list = []
        vert_list = []
        joint_cam_list = []
        joint_img_list = []
        aug_param_list = []
        bb2img_trans_list = []
        for _ in range(2):
            # augmentation
            roi, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, roi_mask = augmentation(img.copy(), bbox, self.phase,
                                                                                            exclude_flip=not self.cfg.DATA.FREIHAND.FLIP,
                                                                                            input_img_shape=(self.cfg.DATA.SIZE, self.cfg.DATA.SIZE),
                                                                                            mask=mask.copy(),
                                                                                            base_scale=self.cfg.DATA.FREIHAND.BASE_SCALE,
                                                                                            scale_factor=self.cfg.DATA.FREIHAND.SCALE,
                                                                                            rot_factor=self.cfg.DATA.FREIHAND.ROT,
                                                                                            shift_wh=[bbox[2], bbox[3]],
                                                                                            gaussian_std=self.cfg.DATA.STD)
            if self.color_aug is not None:
                roi = self.color_aug(roi)
            roi = base_transform(roi, self.cfg.DATA.SIZE, mean=self.cfg.DATA.IMG_MEAN, std=self.cfg.DATA.IMG_STD)
            # img = inv_based_tranmsform(roi)
            # cv2.imshow('test', img)
            # cv2.waitKey(0)
            roi = torch.from_numpy(roi).float()
            roi_mask = torch.from_numpy(roi_mask).float()
            bb2img_trans = torch.from_numpy(bb2img_trans).float()
            aug_param = torch.from_numpy(aug_param).float()

            # joints
            joint_img_, princpt_ = augmentation_2d(img, joint_img, princpt, img2bb_trans, do_flip)
            joint_img_ = torch.from_numpy(joint_img_[:, :2]).float() / self.cfg.DATA.SIZE

            # 3D rot
            rot = aug_param[0].item()
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                                    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                                    [0, 0, 1]], dtype=np.float32)
            joint_cam_ = torch.from_numpy(np.dot(rot_aug_mat, joint_cam.T).T).float()
            vert_ = torch.from_numpy(np.dot(rot_aug_mat, vert.T).T).float()

            # K
            focal_ = focal * roi.size(1) / (bbox[2]*aug_param[1])
            calib = np.eye(4)
            calib[0, 0] = focal_[0]
            calib[1, 1] = focal_[1]
            calib[:2, 2:3] = princpt_[:, None]
            calib = torch.from_numpy(calib).float()

            roi_list.append(roi)
            mask_list.append(roi_mask.unsqueeze(0))
            calib_list.append(calib)
            vert_list.append(vert_)
            joint_cam_list.append(joint_cam_)
            joint_img_list.append(joint_img_)
            aug_param_list.append(aug_param)
            bb2img_trans_list.append(bb2img_trans)

        # print(f'calib:\n{calib_list[0]}')
        roi = torch.cat(roi_list, 0)
        mask = torch.cat(mask_list, 0)
        calib = torch.cat(calib_list, 0)
        joint_cam = torch.cat(joint_cam_list, -1)
        vert = torch.cat(vert_list, -1)
        joint_img = torch.cat(joint_img_list, -1)
        aug_param = torch.cat(aug_param_list, 0)
        bb2img_trans = torch.cat(bb2img_trans_list, -1)

        # postprocess root and joint_cam
        root = joint_cam[0].clone()
        joint_cam -= root
        vert -= root
        joint_cam /= 0.2
        vert /= 0.2

        # out
        res = {'img': roi, 'joint_img': joint_img, 'joint_cam': joint_cam, 'verts': vert, 'mask': mask,
               'root': root, 'calib': calib, 'aug_param': aug_param, 'bb2img_trans': bb2img_trans,}

        return res

    def get_training_sample(self, idx):
        """Get a FreiHAND sample for training
        """
        # read
        img = read_img_abs(idx, self.cfg.DATA.FREIHAND.ROOT, 'training')
        vert = read_mesh(idx % self.one_version_len, self.cfg.DATA.FREIHAND.ROOT).x.numpy()
        mask = read_mask_woclip(idx % self.one_version_len, self.cfg.DATA.FREIHAND.ROOT, 'training')
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        contours.sort(key=cnt_area, reverse=True)
        bbox = cv2.boundingRect(contours[0])
        center = [bbox[0]+bbox[2]*0.5, bbox[1]+bbox[3]*0.5]
        w, h = bbox[2], bbox[3]
        bbox = [center[0]-0.5 * max(w, h), center[1]-0.5 * max(w, h), max(w, h), max(w, h)]
        K, mano, joint_cam = self.db_data_anno[idx]
        K, joint_cam, mano = np.array(K), np.array(joint_cam), np.array(mano)
        joint_img = projectPoints(joint_cam, K)
        princpt = K[0:2, 2].astype(np.float32)
        focal = np.array( [K[0, 0], K[1, 1]], dtype=np.float32)

        # augmentation
        roi, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, mask = augmentation(img, bbox, self.phase,
                                                                                        exclude_flip=not self.cfg.DATA.FREIHAND.FLIP,
                                                                                        input_img_shape=(self.cfg.DATA.SIZE, self.cfg.DATA.SIZE),
                                                                                        mask=mask,
                                                                                        base_scale=self.cfg.DATA.FREIHAND.BASE_SCALE,
                                                                                        scale_factor=self.cfg.DATA.FREIHAND.SCALE,
                                                                                        rot_factor=self.cfg.DATA.FREIHAND.ROT,
                                                                                        shift_wh=[bbox[2], bbox[3]],
                                                                                        gaussian_std=self.cfg.DATA.STD)
        if self.color_aug is not None:
            roi = self.color_aug(roi)
        roi = base_transform(roi, self.cfg.DATA.SIZE, mean=self.cfg.DATA.IMG_MEAN, std=self.cfg.DATA.IMG_STD)
        # img = inv_based_tranmsform(roi)
        # cv2.imshow('test', img)
        # cv2.waitKey(0)
        roi = torch.from_numpy(roi).float()
        mask = torch.from_numpy(mask).float()
        bb2img_trans = torch.from_numpy(bb2img_trans).float()

        # joints
        joint_img, princpt = augmentation_2d(img, joint_img, princpt, img2bb_trans, do_flip)
        joint_img = torch.from_numpy(joint_img[:, :2]).float() / self.cfg.DATA.SIZE

        # 3D rot
        rot = aug_param[0]
        rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                                [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                                [0, 0, 1]], dtype=np.float32)
        ''' 拇指朝 z+ 方向旋轉 (-rot) 度
        | cos(-rot) | -sin(-rot) | 0 |
        | sin(-rot) |  cos(-rot) | 0 |
        |     0     |    0       | 1 |

        因為 內部的 rot 是指旋轉 bbox! 旋轉後的 bbox 再經由 affine trans 轉到輸出圖片座標時，事實上做的旋轉是倒過來的
        '''
        joint_cam = np.dot(rot_aug_mat, joint_cam.T).T
        vert = np.dot(rot_aug_mat, vert.T).T

        # K
        focal = focal * roi.size(1) / (bbox[2]*aug_param[1])
        calib = np.eye(4)
        calib[0, 0] = focal[0]
        calib[1, 1] = focal[1]
        calib[:2, 2:3] = princpt[:, None]
        calib = torch.from_numpy(calib).float()

        # postprocess root and joint_cam
        root = joint_cam[0].copy()
        joint_cam -= root
        vert -= root
        joint_cam /= 0.2
        vert /= 0.2
        root = torch.from_numpy(root).float()
        joint_cam = torch.from_numpy(joint_cam).float()
        vert = torch.from_numpy(vert).float()

        # out
        res = {'img': roi, 'joint_img': joint_img, 'joint_cam': joint_cam, 'verts': vert, 'mask': mask, 'root': root, 'calib': calib}

        return res

    def get_eval_sample(self, idx):
        """Get FreiHAND sample for evaluation
        """
        # read
        img = read_img(idx, self.cfg.DATA.FREIHAND.ROOT, 'evaluation', 'gs')
        K, scale = self.db_data_anno[idx]
        K = np.array(K)
        princpt = K[0:2, 2].astype(np.float32)
        focal = np.array( [K[0, 0], K[1, 1]], dtype=np.float32)
        bbox = [img.shape[1]//2-50, img.shape[0]//2-50, 100, 100]  # img: (224, 224), bbox=[62, 62, 100, 100]
        center = [bbox[0]+bbox[2]*0.5, bbox[1]+bbox[3]*0.5]
        w, h = bbox[2], bbox[3]
        bbox = [center[0]-0.5 * max(w, h), center[1]-0.5 * max(w, h), max(w, h), max(w, h)]  # square(left, top, w, h)

        # aug
        roi, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, _ = augmentation(img, bbox, self.phase,
                                                                                        exclude_flip=not self.cfg.DATA.FREIHAND.FLIP,
                                                                                        input_img_shape=(self.cfg.DATA.SIZE, self.cfg.DATA.SIZE),
                                                                                        mask=None,
                                                                                        base_scale=self.cfg.DATA.FREIHAND.BASE_SCALE,
                                                                                        scale_factor=self.cfg.DATA.FREIHAND.SCALE,
                                                                                        rot_factor=self.cfg.DATA.FREIHAND.ROT,
                                                                                        shift_wh=[bbox[2], bbox[3]],
                                                                                        gaussian_std=self.cfg.DATA.STD)
        # aug_param: [旋轉徑度, bbox 放大倍率(框到更多手外圍的圖), 平移 x 佔畫面比例, 平移 y 佔畫面比例]
        roi = base_transform(roi, self.cfg.DATA.SIZE, mean=self.cfg.DATA.IMG_MEAN, std=self.cfg.DATA.IMG_STD)
        roi = torch.from_numpy(roi).float()

        # s     : 放大倍率 = roi.size(1) / (bbox[2]*aug_param[1])
        # Um, Vm: 平移距離 = roi.size(1) * aug_param[2], ...[3]
        #                                平移 x 佔整張圖的比例
        # Evaluation 時，不會有 shift，只會有 bbox 的移動

        # Joint, augmentation_2d, just like get_training_sample() do
        princpt = trans_point2d(princpt, img2bb_trans)

        # K
        focal = focal * roi.size(1) / (bbox[2]*aug_param[1])  # 放大倍率為：result(roi) / origin(bbox*scale, 擴大 bbox 擷取框框的部份)
        calib = np.eye(4)
        calib[0, 0] = focal[0]
        calib[1, 1] = focal[1]
        calib[:2, 2:3] = princpt[:, None]
        # print(f'K:\n{K}')
        # print(f'calib:\n{calib}')
        calib = torch.from_numpy(calib).float()

        return {'img': roi, 'calib': calib, 'idx': idx}

    def __len__(self):

        return len(self.db_data_anno)

    def visualization(self, res, idx):
        """Visualization of correctness
        """
        import matplotlib.pyplot as plt
        from mobrecon.tools.vis import perspective
        num_sample = (1, 2)[self.cfg.DATA.CONTRASTIVE]
        for i in range(num_sample):
            fig = plt.figure(figsize=(8, 2))
            img = inv_base_tranmsform(res['img'].numpy()[i*3:(i+1)*3])
            # joint_img
            if 'joint_img' in res:
                ax = plt.subplot(1, 4, 1)
                vis_joint_img = vc.render_bones_from_uv(np.flip(res['joint_img'].numpy()[:, i*2:(i+1)*2]*self.cfg.DATA.SIZE, axis=-1).copy(),
                                                        img.copy(), MPIIHandJoints, thickness=2)
                ax.imshow(vis_joint_img)
                ax.set_title('kps2d')
                ax.axis('off')
            # aligned joint_cam
            if 'joint_cam' in res:
                ax = plt.subplot(1, 4, 2)
                xyz = res['joint_cam'].numpy()[:, i*3:(i+1)*3].copy()
                root = res['root'].numpy()[i*3:(i+1)*3].copy()
                xyz = xyz * 0.2 + root
                proj3d = perspective(torch.from_numpy(xyz.copy()).permute(1, 0).unsqueeze(0), res['calib'][i*4:(i+1)*4].unsqueeze(0))[0].numpy().T
                vis_joint_img = vc.render_bones_from_uv(np.flip(proj3d[:, :2], axis=-1).copy(),
                                                        img.copy(), MPIIHandJoints, thickness=2)
                ax.imshow(vis_joint_img)
                ax.set_title('kps3d2d')
                ax.axis('off')
            # aligned verts
            if 'verts' in res:
                ax = plt.subplot(1, 4, 3)
                vert = res['verts'].numpy()[:, i*3:(i+1)*3].copy()
                vert = vert * 0.2 + root
                proj_vert = perspective(torch.from_numpy(vert.copy()).permute(1, 0).unsqueeze(0), res['calib'][i*4:(i+1)*4].unsqueeze(0))[0].numpy().T
                ax.imshow(img)
                plt.plot(proj_vert[:, 0], proj_vert[:, 1], 'o', color='red', markersize=1)
                ax.set_title('verts')
                ax.axis('off')
            # mask
            if 'mask' in res:
                ax = plt.subplot(1, 4, 4)
                if res['mask'].ndim == 3:
                    mask = res['mask'].numpy()[i] * 255
                else:
                    mask = res['mask'].numpy() * 255
                mask_ = np.concatenate([mask[:, :, None]] + [np.zeros_like(mask[:, :, None])] * 2, 2).astype(np.uint8)
                img_mask = cv2.addWeighted(img, 1, mask_, 0.5, 1)
                ax.imshow(img_mask)
                ax.set_title('mask')
                ax.axis('off')
            plt.show()
        if self.cfg.DATA.CONTRASTIVE:
            aug_param = data['aug_param'].unsqueeze(0)
            vert = data['verts'].unsqueeze(0)
            joint_img = data['joint_img'].unsqueeze(0)
            uv_trans = data['bb2img_trans'].unsqueeze(0)
            loss3d = contrastive_loss_3d(vert, aug_param)
            loss2d = contrastive_loss_2d(joint_img, uv_trans, data['img'].size(2))
            print(idx, loss3d, loss2d)


if __name__ == '__main__':
    """Test the dataset
    """
    from mobrecon.main import setup
    from options.cfg_options import CFGOptions

    args = CFGOptions().parse()
    args.config_file = 'mobrecon/configs/mobrecon_rs.yml'
    cfg = setup(args)

    dataset = HanCo(cfg, 'train')
    # for i in range(0, len(dataset), len(dataset)//10):
    #     print(i)
    #     data = dataset.__getitem__(i)
    #     dataset.visualization(data, i)

    # idx = 1090
    # data = dataset[idx]
    # dataset.visualization(data, idx)
