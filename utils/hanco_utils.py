from typing import Dict
import numpy as np
import json
import os
import time
import skimage.io as io

__all__ = [
    'projectPoints',
    'read_img',
    'read_mask',
    'read_verts',
]

''' General util functions. '''
# Copied from fh_utils.py
def _assert_exist(path):
    assert os.path.exists(path), f'File does not exists: {path}'

def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


''' Dataset related functions '''
def read_img(hanco_root: str, folder: str, seq_id: int, cam_id: int, frame_id: int) -> np.ndarray:
    ''' read image
        train -> folder in ['rgb', 'rgb_color_auto', ...]
        test  -> folder == 'rgb'
    '''
    img_path = os.path.join(hanco_root, folder, f'{seq_id:04d}', f'cam{cam_id}', f'{frame_id:08d}.jpg')
    _assert_exist(img_path)

    return io.imread(img_path)

def read_mask(hanco_root: str, seq_id: int, cam_id: int, frame_id: int) -> np.ndarray:
    ''' read mask
        folder == 'mask_hand'
    '''
    mask_path = os.path.join(hanco_root, 'mask_hand', f'{seq_id:04d}', f'cam{cam_id}', f'{frame_id:08d}.jpg')
    _assert_exist(mask_path)

    return io.imread(mask_path)

def read_annot(hanco_root: str, seq_id: int, cam_id: int) -> Dict[str, np.ndarray]:
    ''' return { numpy annotations }
        IMPLEMENT ONLY NUMPY_SEQ VERSION
        including: {
            'verts': (#, 778, 3), (Required)
            'joint': (#, 21,  3), (Optional)
            'global_t': (#, 1, 3), (Optional)
            'intrinsic': (#, 3, 3), (Optional)
        }

        Details: see top of mobrecon/datasets/hanco.py
    '''
    if not os.path.exists(os.path.join(hanco_root, 'numpy_seq', '0000', 'cam0.npz')):
        raise NotImplemented('implement numpy_seq version only')

    np_path = os.path.join(hanco_root, 'numpy_seq', f'{seq_id:04d}', f'cam{cam_id}.npz')
    _assert_exist(np_path)

    return np.load(np_path)
