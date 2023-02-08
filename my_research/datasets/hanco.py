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

directory name:
f'{seq_id:04d}', f'cam{cam_id}', f'{frame_id:08d}'

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

| ------------------  rgb/  ------------------- | -------------  rgb_color_auto/  ------------- | ... | 
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

dataset[i] = {
    'img': (#, 3, *shape),
    'mask': (#, *shape),

    'joint_cam': (#, 21, 3),
    'joint_img': (#, 21, 2),
    'verts': (#, 778, 3),

    'root': (#, 3),
    'calib': (#, 4, 4),
}

'''

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import torch
import torch.utils.data as data
import numpy as np
from utils.fh_utils import load_db_annotation, read_mesh, read_img_abs, read_mask_woclip, projectPoints
from utils.hanco_utils import read_img, read_mask, read_annot

from utils.vis import base_transform, inv_base_tranmsform, cnt_area
import cv2
from utils.augmentation import Augmentation
from termcolor import cprint
from utils.preprocessing import augmentation, augmentation_2d, trans_point2d
from my_research.tools.kinematics import MPIIHandJoints
from my_research.models.loss import contrastive_loss_3d, contrastive_loss_2d
import vctoolkit as vc
from my_research.build import DATA_REGISTRY

@DATA_REGISTRY.register()
class HanCo(data.Dataset):
    def __init__(self, cfg, phase='train', frame_counts=8, writer=None):
        '''Init a FreiHAND Dataset

        Args:
            cfg : config file
            frame_counts: (int, optional): transformer got how many frames at a time. Default to 8.
                not required in 'test' phase
            phase (str, optional): train or eval. Defaults to 'train'.
        '''
        super(HanCo, self).__init__()
        self.cfg = cfg
        if phase == 'val':
            phase = 'valid'

        self.phase = phase
        assert phase in ('train', 'test', 'valid'), f'phase should be "train" or "test" or "valid", got: {phase}'
        self.frame_counts = frame_counts
        assert 1 < frame_counts <= 20, f'frame_counts should be in (2, 21), got: {frame_counts}'

        self.hanco_root = self.cfg.DATA.HANCO.ROOT

        split_info_path = os.path.join(self.hanco_root, 'dataset_split_info.npz')
        if os.path.isfile(split_info_path):
            split_info = np.load(split_info_path)
            self.train_seq, self.valid_seq, self.test_seq, self.valid_seq_start_frame = \
                list(split_info['train_seq']), list(split_info['valid_seq']), \
                list(split_info['test_seq']), list(split_info['valid_seq_start_frame'])
        else:
            self.train_seq, self.test_seq = self._get_valid_sequence(
                hanco_root=self.hanco_root, DEBUG=False
            )
            # ! split train_seq into {train_seq, valid_seq}
            np.random.seed(0)
            self.train_seq, self.valid_seq = self._split_train_valid(self.train_seq, ratio=0.03)  # 583:237592
            self.valid_seq_start_frame = self._get_valid_seq_start_frame(self.valid_seq)
            np.savez(split_info_path, train_seq=self.train_seq, valid_seq=self.valid_seq,
                test_seq=self.test_seq, valid_seq_start_frame=self.valid_seq_start_frame,
                frame_count=self.frame_counts, ratio=0.03)

        self.len_train_seq_cam = len(self.train_seq) * 8  # (sequence X cam) counts of all augment folders
        self.len_valid_seq_cam = len(self.valid_seq) * 8

        self.image_aug = ['rgb', 'rgb_color_auto', 'rgb_color_sample', 'rgb_homo', 'rgb_merged']

        self.color_aug = Augmentation() if cfg.DATA.COLOR_AUG and 'train' in self.phase else None
        cprint(f'Loaded HanCo {self.phase} {self.__len__()} sequences', 'red')
        if self.phase == 'train':
            # need train/valid/test HanCo data while training
            # this "self.phase" means WHICH kind of data is choose from HanCo
            cprint(f'  with {len(self.image_aug)} image folders, {len(self.train_seq): > 4} valid training sequences, 8 cameras', 'red')
        elif self.phase == 'valid':
            cprint(f'  with {len(self.image_aug)} image folders, {len(self.valid_seq): > 4} valid validate sequences, 8 cameras', 'red')
        else:
            cprint(f'  with {len(self.test_seq)} valid test sequences, 8 cameras', 'red')
        if writer is not None:
            writer.print_str(f'Loaded HanCo {self.phase} {self.__len__()} sequences')


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
        print(f'Train({len(train_list)}): {train_list[:3] + ["..."] + train_list[-3:]}', end=', ')
        print(f'Test({len(test_list)}): {test_list[:3] + ["..."] + test_list[-3:]}')
        return train_list, test_list

    def _split_train_valid(self, train_seq, ratio=0.02):
        '''
        Split {train_seq} into train/ valid
        by {ratio}:          1-ratio/ ratio

        return sorted train / valid sequence
        '''
        np.random.shuffle(train_seq)
        valid_size = int(len(train_seq) * ratio)
        valid_seq = train_seq[ -valid_size:]
        train_seq = train_seq[:-valid_size]
        return sorted(train_seq), sorted(valid_seq)

    def _get_valid_seq_start_frame(self, valid_seq):
        ''' min seq len = 23 '''
        valid_seq_start = []
        for seq_id in valid_seq:
            max_seq_length = len(os.listdir(
                os.path.join(self.hanco_root, 'rgb', f'{seq_id:04d}', 'cam0')
            ))
            start = np.random.randint(
                max_seq_length // 3, max_seq_length -self.frame_counts +1,
                size=5 * 8
            )   # 5 * seq_count * cam_count:8
            # from (1/3) len to end
            # min case: 6 ~ 23
            valid_seq_start += [start]
        from einops import rearrange
        # shape: (5*8, seq_count)
        valid_seq_start = rearrange(valid_seq_start,
            'Seq (IA Cam) -> (IA Seq Cam)', IA=5)  # IA: Image augment folders
        return list(valid_seq_start)

    def __len__(self):
        if self.phase == 'train':
            return len(self.image_aug) * len(self.train_seq) * 8
        elif self.phase == 'valid':
            return len(self.image_aug) * len(self.valid_seq) * 8
        elif self.phase == 'test':
            return len(self.test_seq) * 8

    def __getitem__(self, idx):
        if self.phase == 'train':
            try:
                aug_id, seq_id, cam_id = self._inverse_compute_index(idx)
                if self.cfg.DATA.CONTRASTIVE:
                    raise NotImplemented('contrastive data audmentation not implemented yet')
                else:
                    return self.get_training_sample(aug_id, seq_id, cam_id)
            except:
                raise Exception(f'--- [Error] at {self.phase}: data[{idx}] --- aug: {aug_id}, seq: {seq_id}, cam: {cam_id}')
        elif self.phase == 'valid':
            aug_id, seq_id, cam_id = self._inverse_compute_index(idx)
            return self.get_training_sample(aug_id, seq_id, cam_id,
                        start=self.valid_seq_start_frame[idx])  # valid_start = shape(IA Seq Cam)
        elif self.phase == 'test':
            seq_id, cam_id = self._inverse_compute_index(idx)
            return self.get_testing_sample(seq_id, cam_id)

    def _compute_index(self, aug_id, seq_id, cam_id):
        ''' aug_id, seq_id, cam_id -> index
        '''
        if self.phase == 'train':
            return aug_id * self.len_train_seq_cam + \
                   self.train_seq.index(seq_id) * 8 + cam_id
        elif self.phase == 'valid':
            return aug_id * self.len_valid_seq_cam + \
                   self.valid_seq.index(seq_id) * 8 + cam_id
        elif self.phase == 'test':
            return self.test_seq.index(seq_id) * 8 + cam_id

    def _inverse_compute_index(self, idx):
        ''' index -> (aug_id), seq_id, cam_id
        '''
        if self.phase == 'train':
            aug_id = int(idx / self.len_train_seq_cam)
            seq_id = self.train_seq[int(idx % self.len_train_seq_cam / 8)] # No. ? in train_seq
            cam_id = idx % 8
            return aug_id, seq_id, cam_id
        elif self.phase == 'valid':
            aug_id = int(idx / self.len_valid_seq_cam)
            seq_id = self.valid_seq[int(idx % self.len_valid_seq_cam / 8)] # No. ? in valid_seq
            cam_id = idx % 8
            return aug_id, seq_id, cam_id
        elif self.phase == 'test':
            seq_id = self.test_seq[int(idx / 8)] # No. ? in train_seq
            cam_id = idx % 8
            return seq_id, cam_id

    def get_training_sample(self, aug_id, seq_id, cam_id, start=None):
        ''' Get HanCo sequence, see details at top - FORMAT
            with frame counts: self.frame_counts

            start: Optional[int], pre-defined start frames
                                  uses in valid-set
        '''
        # _ = f'{seq_id:04d}', f'cam{cam_id}', f'{frame_id:08d}'

        # compute frame counts in seq: seq_id
        max_seq_length = len(os.listdir(
            os.path.join(self.hanco_root, 'rgb', f'{seq_id:04d}', f'cam{cam_id}')
        ))

        # get sequence: [start, ..., start+self.frame_counts -1][ ... ]
        if start is None:
            start = np.random.randint(0, max_seq_length -self.frame_counts +1)
            # len:20, frames:10 -> start(0, 11) -> [0, ..., 9], [10, ..., 19]
        assert start + self.frame_counts <= max_seq_length, \
            f'start frame too large with frame_count={self.frame_counts}, ({start} / {max_seq_length})'

        # read images, masks
        images, masks = [], []
        for frame_id in range(start, start + self.frame_counts):
            img = read_img(self.hanco_root, self.image_aug[aug_id], seq_id, cam_id, frame_id)
            mask = read_mask(self.hanco_root, seq_id, cam_id, frame_id)
            images += [img]
            masks += [mask]

        images, masks = np.stack(images), np.stack(masks)

        # read annots
        Annots = read_annot(self.hanco_root, seq_id, cam_id)
        Roots = Annots['global_t'][start:start + self.frame_counts]
        Verts = Annots['verts'][start:start + self.frame_counts] + Roots
        Joints = Annots['joint'][start:start + self.frame_counts] + Roots
        Intrinsics = Annots['intrinsic'][start:start + self.frame_counts]

        # Complete list of ndarray {roi, mask, joint_cam, joint_img, verts, root, calib}
        roi_list, mask_list, joint_cam_list, joint_img_list, verts_list, root_list, calib_list = [], [], [], [], [], [], []

        for i in range(self.frame_counts):
            # augment for data[i]: image, mask, annots
            img = images[i]
            mask = masks[i]
            vert = Verts[i]

            bbox = self._get_init_bbox_from_mask(mask=mask)
            K, joint_cam = Intrinsics[i], Joints[i] # (3, 3), (21, 3)

            # Copied from FreiHAND.get_training_sample()
            joint_img = projectPoints(joint_cam, K) # (21, 2)
            princpt = K[0:2, 2].astype(np.float32)
            focal = np.array( [K[0, 0], K[1, 1]], dtype=np.float32)

            roi, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, mask = augmentation(img, bbox, self.phase,
                                                                                            exclude_flip=not self.cfg.DATA.HANCO.FLIP,
                                                                                            input_img_shape=(self.cfg.DATA.SIZE, self.cfg.DATA.SIZE),
                                                                                            mask=mask,
                                                                                            base_scale=self.cfg.DATA.HANCO.BASE_SCALE,
                                                                                            scale_factor=self.cfg.DATA.HANCO.SCALE,
                                                                                            rot_factor=self.cfg.DATA.HANCO.ROT,
                                                                                            shift_wh=[bbox[2], bbox[3]],
                                                                                            gaussian_std=self.cfg.DATA.STD)
            if self.color_aug is not None:
                roi = self.color_aug(roi)
            roi = base_transform(roi, self.cfg.DATA.SIZE, mean=self.cfg.DATA.IMG_MEAN, std=self.cfg.DATA.IMG_STD)

            # joints
            joint_img, princpt = augmentation_2d(img, joint_img, princpt, img2bb_trans, do_flip)
            joint_img = joint_img[:, :2] / self.cfg.DATA.SIZE

            # 3D rot
            rot = aug_param[0]
            assert rot == 0, 'should not rotate in hanco dataset'
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
            focal = focal * self.cfg.DATA.SIZE / (bbox[2]*aug_param[1])
            calib = np.array([
                [ focal[0],    0    , princpt[0], 0],
                [    0    , focal[1], princpt[1], 0],
                [    0    ,    0    ,     1     , 0],
                [    0    ,    0    ,     0     , 1]
            ], dtype=np.float64)

            # postprocess root and joint_cam
            root = Roots[i]
            joint_cam -= root
            vert -= root
            joint_cam /= self.cfg.DATA.HANCO.SCALE  # edited from 0.2
            vert /= self.cfg.DATA.HANCO.SCALE  # edited from 0.2

            # add to list
            roi_list += [roi]
            mask_list += [mask]
            joint_cam_list += [joint_cam]
            joint_img_list += [joint_img]
            verts_list += [vert]
            root_list += [root[0]]
            calib_list += [calib]

        # Prepare torch.Tensor from ndarray lists
        roi_tensor = torch.from_numpy(np.stack(roi_list)).float()
        mask_tensor = torch.from_numpy(np.stack(mask_list)).float()
        joint_cam_tensor = torch.from_numpy(np.stack(joint_cam_list)).float()
        joint_img_tensor = torch.from_numpy(np.stack(joint_img_list)).float()
        verts_tensor = torch.from_numpy(np.stack(verts_list)).float()
        root_tensor = torch.from_numpy(np.stack(root_list)).float()
        calib_tensor = torch.from_numpy(np.stack(calib_list)).float()

        ret = {
            'start': start,

            'img': roi_tensor,
            'mask': mask_tensor,

            'joint_cam': joint_cam_tensor,
            'joint_img': joint_img_tensor,
            'verts': verts_tensor,

            'root': root_tensor,

            'calib': calib_tensor,
        }

        return ret

    def _get_init_bbox_from_mask(self, mask=None, img=None):
        ''' only called by get_[training, contrastive, testing]_sample
        '''
        if self.phase in ['train', 'valid']:
            assert mask is not None, 'phase: train|valid, {mask} should not be None ' + f'got: {mask}'
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = list(contours)
            contours.sort(key=cnt_area, reverse=True)
            bbox = cv2.boundingRect(contours[0])
        else:
            assert img is not None, 'phase: train, {img} should not be None ' + f'got: {img}'
            bbox = [img.shape[1]//2-50, img.shape[0]//2-50, 100, 100]  # img: (224, 224), bbox=[62, 62, 100, 100]

        center = [bbox[0]+bbox[2]*0.5, bbox[1]+bbox[3]*0.5]
        w, h = bbox[2], bbox[3]
        bbox = [center[0]-0.5 * max(w, h), center[1]-0.5 * max(w, h), max(w, h), max(w, h)]  # square(left, top, w, h)
        return bbox

    def get_testing_sample(self, seq_id, cam_id):
        ''' Get HanCo sequence, see details at top - FORMAT
            (X) with frame counts: self.frame_counts
            with ALL frames
        '''
        # _ = f'{seq_id:04d}', f'cam{cam_id}', f'{frame_id:08d}'

        # compute frame counts in seq: seq_id
        max_seq_length = len(os.listdir(
            os.path.join(self.hanco_root, 'rgb', f'{seq_id:04d}', f'cam{cam_id}')
        ))

        # read images, masks
        images, masks = [], []
        for frame_id in range(max_seq_length):
            img = read_img(self.hanco_root, 'rgb', seq_id, cam_id, frame_id)
            mask = read_mask(self.hanco_root, seq_id, cam_id, frame_id)
            images += [img]
            masks += [mask]

        images, masks = np.stack(images), np.stack(masks)

        # read annots
        Annots = read_annot(self.hanco_root, seq_id, cam_id)
        Roots = Annots['global_t']
        Verts = Annots['verts'] + Roots
        Joints = Annots['joint'] + Roots
        Intrinsics = Annots['intrinsic']

        # Complete list of ndarray {roi, mask, joint_cam, joint_img, verts, root, calib}
        roi_list, mask_list, joint_cam_list, joint_img_list, verts_list, root_list, calib_list = [], [], [], [], [], [], []

        for i in range(max_seq_length):
            img = images[i]
            K = Intrinsics[i]
            mask = masks[i]  # Ground-Truth
            vert = Verts[i]  # Ground-Truth
            joint_cam = Joints[i]  # Ground-Truth

            bbox = self._get_init_bbox_from_mask(img=images[i])

            joint_img = projectPoints(joint_cam, K) # (21, 2), Ground-Truth

            princpt = K[0:2, 2].astype(np.float32)
            focal = np.array( [K[0, 0], K[1, 1]], dtype=np.float32)

            # aug, also crop {roi, mask} to bbox
            roi, img2bb_trans, bb2img_trans, aug_param, do_flip, scale, mask = augmentation(img, bbox, self.phase,
                                                                                            exclude_flip=not self.cfg.DATA.FREIHAND.FLIP,
                                                                                            input_img_shape=(self.cfg.DATA.SIZE, self.cfg.DATA.SIZE),
                                                                                            mask=mask,
                                                                                            base_scale=self.cfg.DATA.FREIHAND.BASE_SCALE,
                                                                                            scale_factor=self.cfg.DATA.FREIHAND.SCALE,
                                                                                            rot_factor=self.cfg.DATA.FREIHAND.ROT,
                                                                                            shift_wh=[bbox[2], bbox[3]],
                                                                                            gaussian_std=self.cfg.DATA.STD)
            # no color aug
            roi = base_transform(roi, self.cfg.DATA.SIZE, mean=self.cfg.DATA.IMG_MEAN, std=self.cfg.DATA.IMG_STD)

            # update after BBOXING, just like get_training_sample() do
            joint_img, princpt = augmentation_2d(img, joint_img, princpt, img2bb_trans, do_flip)
            joint_img = joint_img[:, :2] / self.cfg.DATA.SIZE

            # No 3D rot

            # K
            focal = focal * self.cfg.DATA.SIZE / (bbox[2]*aug_param[1])  # 放大倍率為：result(roi) / origin(bbox*scale, 擴大 bbox 擷取框框的部份)
            calib = np.array([
                [ focal[0],    0    , princpt[0], 0],
                [    0    , focal[1], princpt[1], 0],
                [    0    ,    0    ,     1     , 0],
                [    0    ,    0    ,     0     , 1]
            ], dtype=np.float64)

            # postprocess root and joint_cam
            root = Roots[i]
            joint_cam -= root
            vert -= root
            joint_cam /= self.cfg.DATA.HANCO.SCALE
            vert /= self.cfg.DATA.HANCO.SCALE

            # add to list
            roi_list += [roi]
            mask_list += [mask]
            joint_cam_list += [joint_cam]
            joint_img_list += [joint_img]
            verts_list += [vert]
            root_list += [root[0]]
            calib_list += [calib]

        # Prepare torch.Tensor from ndarray lists
        roi_tensor  = torch.from_numpy(np.stack(roi_list)).float()
        mask_tensor = torch.from_numpy(np.stack(mask_list)).float()
        joint_cam_tensor = torch.from_numpy(np.stack(joint_cam_list)).float()
        joint_img_tensor = torch.from_numpy(np.stack(joint_img_list)).float()
        verts_tensor = torch.from_numpy(np.stack(verts_list)).float()
        root_tensor = torch.from_numpy(np.stack(root_list)).float()
        calib_tensor = torch.from_numpy(np.stack(calib_list)).float()

        ret = {
            'start': 0,

            'img': roi_tensor,
            'mask': mask_tensor,            # GT

            'joint_cam': joint_cam_tensor,  # GT
            'joint_img': joint_img_tensor,  # GT
            'verts': verts_tensor,          # GT

            'root': root_tensor,            # GT

            'calib': calib_tensor,
        }

        return ret

    def visualization(self, res, idx):
        """ Visualization of correctness
        """
        import matplotlib.pyplot as plt
        from my_research.tools.vis import perspective

        Cols = min(8, self.frame_counts)
        fig = plt.figure(figsize=(18, 8))
        for i in range(Cols):
            img = inv_base_tranmsform(res['img'][i].numpy())
            # joint_img
            if 'joint_img' in res:
                ax = plt.subplot(4, Cols, Cols*0 + i+1)
                vis_joint_img = vc.render_bones_from_uv(np.flip(res['joint_img'][i].numpy()*self.cfg.DATA.SIZE, axis=-1),
                                                        img.copy(), MPIIHandJoints, thickness=2)
                ax.imshow(vis_joint_img)
                ax.set_title('kps2d')
                ax.axis('off')
            # aligned joint_cam
            if 'joint_cam' in res:
                ax = plt.subplot(4, Cols, Cols*1 + i+1)
                xyz = res['joint_cam'][i].numpy()
                root = res['root'][i].numpy()
                xyz = xyz * 0.2 + root
                proj3d = perspective(torch.from_numpy(xyz).permute(1, 0).unsqueeze(0), res['calib'][i].unsqueeze(0))[0].numpy().T
                vis_joint_img = vc.render_bones_from_uv(np.flip(proj3d[:, :2], axis=-1),
                                                        img.copy(), MPIIHandJoints, thickness=2)
                ax.imshow(vis_joint_img)
                ax.set_title('kps3d2d')
                ax.axis('off')
            # aligned verts
            if 'verts' in res:
                ax = plt.subplot(4, Cols, Cols*2 + i+1)
                vert = res['verts'][i].numpy()
                vert = vert * 0.2 + root
                proj_vert = perspective(torch.from_numpy(vert).permute(1, 0).unsqueeze(0), res['calib'][i].unsqueeze(0))[0].numpy().T
                ax.imshow(img)
                plt.plot(proj_vert[:, 0], proj_vert[:, 1], 'o', color='red', markersize=1)
                ax.set_title('verts')
                ax.axis('off')
            # mask
            if 'mask' in res:
                ax = plt.subplot(4, Cols, Cols*3 + i+1)
                mask = res['mask'][i].numpy() * 255
                mask_ = np.concatenate([mask[:, :, None]] + [np.zeros_like(mask[:, :, None])] * 2, 2).astype(np.uint8)
                img_mask = cv2.addWeighted(img, 1, mask_, 0.5, 1)
                ax.imshow(img_mask)
                ax.set_title('mask')
                ax.axis('off')

        if self.phase == 'train':
            aug_id, seq_id, cam_id = self._inverse_compute_index(idx)
            title = f'HanCo  |  folder: {self.image_aug[aug_id]}  |  seq: {seq_id}  |  cam: {cam_id}  |  frame_start: {res["start"]}'
        elif self.phase == 'valid':
            aug_id, seq_id, cam_id = self._inverse_compute_index(idx)
            title = f'HanCo  |  folder: {self.image_aug[aug_id]}  |  seq: {seq_id}  |  cam: {cam_id}  |  frame_start: {res["start"]}'
        elif self.phase == 'test':
            seq_id, cam_id = self._inverse_compute_index(idx)
            title = f'HanCo  |  folder: rgb  |  seq: {seq_id}  |  cam: {cam_id}  |  frame_start: {res["start"]}'

        fig.suptitle(title)
        plt.show()

    def _check_correctness(self, start=0):
        ''' check the correctness of this dataset ( in certain phase )
            full image, un-broken labels
        '''
        from tqdm import tqdm
        print(f'Check correctness of HanCo[{self.phase}]...')

        Error_seqs = []
        for idx in tqdm(range(start, self.__len__())):
            aug_id, seq_id, cam_id = self._inverse_compute_index(idx)
            max_seq_length = len(os.listdir(
                os.path.join(self.hanco_root, self.image_aug[aug_id], f'{seq_id:04d}', f'cam{cam_id}')
            ))
            annots = read_annot(self.hanco_root, seq_id, cam_id)

            global_t = None
            try:
                global_t = annots['global_t']
                _ = annots['verts'], annots['joint'], annots['intrinsic']
            except:
                Error_seqs += [f'label file broken, data[{idx}]'
                               f'aug: {self.image_aug[aug_id]}, seq: {seq_id}, cam: {cam_id}']

            if global_t is not None:
                if len(annots['global_t']) != max_seq_length:
                    Error_seqs += [f'image counts not fit, label: {len(annots["global_t"])} <-> img: {max_seq_length}, '
                                   f'data[{idx}], aug: {self.image_aug[aug_id]}, seq: {seq_id}, cam: {cam_id}']

            if len(Error_seqs) >= 5:
                print(f'TOO MUCH ERRORS, check to data[{idx}]')
                break

        print(f'There are {len(Error_seqs)} errors')
        for Error in Error_seqs:
            print(Error)


    def __exp_stacked_images(self):
        ''' check if images, masks are read successfully
            in TRAIN phase
        '''
        aug_seq_cam_ids = 0, 2, 3
        data = self.__getitem__(self._compute_index(*aug_seq_cam_ids))
        images = data['image']
        masks = data['mask']

        print(f'aug: {self.image_aug[0]}, seq: 2, cam: 3')
        print(f'start at: {data["start"]}')

        import matplotlib.pyplot as plt
        ax = plt.subplot(1, 2, 1)
        ax.imshow(images[0])
        ax.set_title('rgb')
        ax.axis('off')

        ax = plt.subplot(1, 2, 2)
        ax.imshow(masks[0])
        ax.set_title('mask')
        ax.axis('off')

        plt.show()

    def __exp_read_mesh(self):
        ''' check get vert function
            check PASSED
        '''
        seq_id = 2
        cam_id = 1
        frame_id = 5

        _ = f'{seq_id:04d}', f'cam{cam_id}', f'{frame_id:08d}'
        ply_path = os.path.join(self.hanco_root, 'mesh', f'{seq_id:04d}', f'cam{cam_id}', f'{frame_id:08d}.ply')
        seq_path = os.path.join(self.hanco_root, 'numpy_seq', f'{seq_id:04d}', f'cam{cam_id}.npz')

        from utils.read import read_mesh as read_mesh_
        vert_ply = read_mesh_(ply_path).x.numpy()
        np_file = np.load(seq_path)
        vert_seq = np_file['verts'][frame_id]

        root = np_file['global_t'][frame_id]
        vert_ply -= root
        print('maximum difference between vert_ply & vert_seq: ', (vert_ply - vert_seq).max())

    def __exp_find_min_frame_counts(self):
        ''' min frame counts at seq=7, frames=20
        '''
        # _ = f'{seq_id:04d}', f'cam{cam_id}', f'{frame_id:08d}'
        min_frame_counts = 1000
        min_index = 0
        for seq_id in self.train_seq + self.test_seq:
            path = os.path.join(self.hanco_root, 'mesh', f'{seq_id:04d}', 'cam0')
            frame_counts = len(os.listdir(path))
            if min_frame_counts > frame_counts:
                min_frame_counts = frame_counts
                min_index = seq_id

        print(f'min frames: {min_frame_counts}, at index: {min_index}')


if __name__ == '__main__':
    """Test the dataset
    """
    from my_research.main import setup
    from options.cfg_options import CFGOptions

    # args = CFGOptions().parse()
    # args.config_file = 'my_research/configs/mobrecon_rs.yml'
    # cfg = setup(args)
    from my_research.configs.config import get_cfg
    cfg = get_cfg()
    cfg.DATA.COLOR_AUG = False
    cfg.DATA.HANCO.ROT = 0

    dataset = HanCo(cfg, phase='train', frame_counts=8)
    dataset._check_correctness()
    for i in range(len(dataset)):
        data = dataset[i]
    index = dataset._compute_index(0, 2, 3)  # 1st seq in validation: 75
    # index = 100

    print(f'Show dataset[{index}]')
    data = dataset[index] # get 'rgb', seq:2, cam:3

    dataset.visualization(data, index)

    # for i in range(0, len(dataset), len(dataset)//10):
    #     print(i)
    #     data = dataset.__getitem__(i)
    #     dataset.visualization(data, i)

    # idx = 1090
    # data = dataset[idx]
    # dataset.visualization(data, idx)
