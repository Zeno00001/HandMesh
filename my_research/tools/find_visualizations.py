import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
from os import path as osp
from my_research.datasets.hanco_eval import HanCo_Eval
from matplotlib import pyplot as plt


if __name__ == '__main__':
    """Test the dataset
    """
    from my_research.main import setup
    from options.cfg_options import CFGOptions

    args = CFGOptions().parse()
    # args.exp_name = 'test'
    args.config_file = 'my_research/configs/mobrecon_ds_conf_transformer.yml'
    cfg = setup(args)

    work_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
    work_dir = osp.join(work_dir, 'out', 'HanCo_Eval', 'test')

    dataset = HanCo_Eval(cfg, phase='test', frame_counts=8)
    # dataset.test_seq: testing sequences 

    folder_our = osp.join(work_dir, 'exps_ours_j')
    folder_mob = osp.join(work_dir, 'exps_mob_j')

    # good = [75, 90, 214, 407, 436, 458, 537, 567, 602, 640, 741]\
    sequences = dataset.test_seq
    start_seq = sequences[0]
    start_seq = 466
    for i, seq_id in enumerate(sequences[sequences.index(start_seq):]):
        # if seq_id not in good:
        #     continue

        mpvpe_our_file = np.load(osp.join(folder_our, f'{seq_id:04d}_0.npz'))
        mpvpe_mob_file = np.load(osp.join(folder_mob, f'{seq_id:04d}_0.npz'))

        mpvpe_our_root = mpvpe_our_file['rooted']
        mpvpe_our_no_root = mpvpe_our_file['not_rooted']

        mpvpe_mob_root = mpvpe_mob_file['rooted']
        mpvpe_mob_no_root = mpvpe_mob_file['not_rooted']

        # all are 1-D floating points, representing the error in (mm)
        # plt.plot(mpvpe_our_no_root, 'k-', label='mpvpe ours')
        # plt.plot(mpvpe_mob_no_root, 'b-', label='mpvpe mob')

        plt.title(f'seq: {seq_id:04d} | cam: 0')
        ax = plt.subplot(1, 1, 1)
        ax.set_ylim([-5, 40])

        # camera rooted
        # plt.plot(mpvpe_mob_no_root - mpvpe_our_no_root, 'b-', label='mpvpe, mob-our, larger better')

        # wrist rooted
        plt.plot(mpvpe_mob_root - mpvpe_our_root, 'b-', label='mpvpe_root, mob-our, larger better')

        # plt.legend()
        plt.show()
        # not rooted
        # seq=12, start=6, 36
        # 26, 56, 40 <- all occluded, can't use
        # 31, 6, 22(O)
        # 60, 25 <-- large, strange
        # seq=75, start=27, recorded
        # 89, 30
        # 90, 8
        # 118, 43 -> 50
        # 207, 5, 35, 48 -> 50~
        # 212, 46 -> 100
        # 214, 16 -> 50, 24~44 -> 75
        # 241, 19 -> 60
        # 321, 0~14-> 50, 48 -> 50
        # 407, 40 -> 20
        # 423, 20
        # 436, 38 -> 80
