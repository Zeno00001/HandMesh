# Target: check weight file difference or not
# & create joint conf figure

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from typing import OrderedDict

def check(weight1: OrderedDict, weight2: OrderedDict):
    recorder = []
    for k, v in weight1.items():
        print(f'{k:<70}', end='')
        w1 = v
        w2 = weight2[k]
        if torch.all(w1 == w2):
            print('O')
        else:
            print('X ----------------')
            recorder += [k]
    print('Not same:')
    print('\n'.join(recorder))


def show(weight: OrderedDict):
    for k, v in weight.items():
        print(k)

def test_weight():
    old_weight_path = os.path.join('my_research', 'out', 'FreiHAND_Angle', 'checkpoint_last.pt')
    new_weight_path = os.path.join('my_research', 'out', 'FreiHAND_Angle', 'mrc_ds_angle_4_head_train_GCN', 'checkpoints', 'checkpoint_last.pt')

    old_file = torch.load(old_weight_path)['model_state_dict']
    new_file = torch.load(new_weight_path)['model_state_dict']
    check(old_file, new_file)

labels = [
    'W', #0
    'T0', 'T1', 'T2', 'T3', #4
    'I0', 'I1', 'I2', 'I3', #8
    'M0', 'M1', 'M2', 'M3', #12
    'R0', 'R1', 'R2', 'R3', #16
    'L0', 'L1', 'L2', 'L3', #20
]

joint_pos = np.array([
    [397, 691],
    [318, 664], [178, 552], [97, 469], [54, 385],
    [302, 390], [260, 245], [238, 164], [218, 98],
    [383, 396], [383, 238], [377, 121], [373, 55],
    [452, 414], [479, 275], [494, 174], [500, 103],
    [522, 462], [568, 340], [589, 267], [601, 201],
])

# 座標為文字左下角
joint_pos += (-25, 10)

def show_joint_conf():
    # to get confidence value:
    # 1. set seq_id in seq_runner.py> runner.test()
    # 2. set start_frame in seq_runner.py> runner.seq_pred_one_clip()
    # 3. output conf using "breakpoints" before transformer, and output to file
    path = os.path.join('my_research', 'out', 'HanCo_Eval', 'contour_joint.jpg')
    contour = plt.imread(path)
    path = os.path.join('data', 'HanCo', 'rgb', '0026', 'cam0', '00000032.jpg')
    hand = plt.imread(path)
    path = os.path.join('my_research', 'out', 'HanCo_Eval', 'joint_confidence.npy')
    joint = np.load(path)

    i = 0  # frame_i
    draw_contour = contour.copy()
    fig = plt.figure()
    ax = plt.subplot(1, 2, 1)
    ax.imshow(draw_contour)
    for j, j_pos in enumerate(joint_pos):
        j_conf = joint[0, i, j]
        ax.text(j_pos[0], j_pos[1], f'{j_conf:.2f}', size=8, color='b')
    ax.axis('off')
    ax = plt.subplot(1, 2, 2)
    ax.imshow(hand)
    ax.axis('off')

    fig.suptitle('HanCo | folder: rgb | seq: 26 | cam: 0 | frame: 32')
    plt.subplots_adjust(
        left=0.02,
        bottom=0.02,
        right=0.98,
        top=0.95,
        wspace=0.02, 
        hspace=0.02
    )
    plt.show()  # then saved


if __name__ == '__main__':
    show_joint_conf()
