import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_backbone.build import build_model, build_dataset
from my_backbone.configs.config import get_cfg
from options.cfg_options import CFGOptions
from my_backbone.runner import Runner
import os.path as osp
from utils import utils
from utils.writer import Writer
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    # default_setup(cfg, args)
    return cfg


def main(args):
    # get config
    cfg = setup(args)

    # device
    args.rank = 0
    args.world_size = 1
    args.n_threads = 4
    if -1 in cfg.TRAIN.GPU_ID or not torch.cuda.is_available():
        device = torch.device('cpu')
        print('CPU mode')
    elif len(cfg.TRAIN.GPU_ID) == 1:
        device = torch.device('cuda', cfg.TRAIN.GPU_ID[0])
        print('CUDA ' + str(cfg.TRAIN.GPU_ID) + ' Used')
    else:
        raise Exception('Do not support multi-GPU training')
    cudnn.benchmark = True
    cudnn.deterministic = False  #FIXME

    # print config
    if args.rank == 0:
        print(cfg)
        print(args.exp_name)
    # from my_backbone.models.densestack import DenseStack_BackBone
    exec('from my_backbone.models.{} import {}'.format(cfg.MODEL.NAME.lower(), cfg.MODEL.NAME + '_Backbone'))

    # from my_backbone.datasets.multipledatasets import MultipleDatasets
    exec('from my_backbone.datasets.{} import {}'.format(cfg.TRAIN.DATASET.lower(), cfg.TRAIN.DATASET))

    # from my_backbone.datasets.ge import Ge
    exec('from my_backbone.datasets.{} import {}'.format(cfg.VAL.DATASET.lower(), cfg.VAL.DATASET))

    # dir
    args.work_dir = osp.dirname(osp.realpath(__file__))  # HandMesh/my_backbone
    args.out_dir = osp.join(args.work_dir, 'out', cfg.TRAIN.DATASET, args.exp_name)  # my_backbone/out/Multiple.../
    args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
    args.board_dir = osp.join(args.out_dir, 'board')
    args.eval_dir = osp.join(args.out_dir, cfg.VAL.SAVE_DIR)
    args.test_dir = osp.join(args.out_dir, cfg.TEST.SAVE_DIR)
    try:
        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(args.checkpoints_dir, exist_ok=True)
        os.makedirs(args.board_dir, exist_ok=True)
        os.makedirs(args.eval_dir, exist_ok=True)
        os.makedirs(args.test_dir, exist_ok=True)
    except: pass

    # log
    writer = Writer(args)
    writer.print_str(args)
    writer.print_str(cfg)
    # args.board_dir == 'out/MultipleDatasets/mrc_ds/board/'
    board = SummaryWriter(args.board_dir) if cfg.PHASE == 'train' and args.rank == 0 else None

    # model
    model = build_model(cfg).to(device)

    # optim
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # resume
    if cfg.PHASE == 'train':
        if cfg.MODEL.RESUME:
            if len(cfg.MODEL.RESUME.split('/')) > 1:
                model_path = cfg.MODEL.RESUME
            else:
                model_path = osp.join(args.checkpoints_dir, cfg.MODEL.RESUME)
                if args.exp_name == 'test':
                    model_path = osp.join(args.out_dir, '../densestack_conf', 'checkpoints', cfg.MODEL.RESUME)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch'] + 1
            writer.print_str('Resume from: {}, start epoch: {}'.format(model_path, epoch))
            print('Resume from: {}, start epoch: {}'.format(model_path, epoch))
        else:
            epoch = 0
            writer.print_str('Train from 0 epoch')
    elif cfg.PHASE in ['eval', 'pred', 'demo']:
        epoch = 0
        if len(cfg.MODEL.RESUME.split('/')) > 1:
            model_path = cfg.MODEL.RESUME
        else:
            model_path = osp.join(args.checkpoints_dir, cfg.MODEL.RESUME)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Eval model in {model_path}, with dataset: {cfg.VAL.DATASET}')
    else:
        input('[ERROR] wrong cfg PHASE while loading model')

    # data
    kwargs = {"pin_memory": True, "num_workers": 4, "drop_last": True}  # num_worker: 8
    if cfg.PHASE in ['train',]:
        train_dataset = build_dataset(cfg, 'train', writer=writer)
        train_sampler = None
        train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=(train_sampler is None), sampler=train_sampler, **kwargs)
    else:
        print('Need not trainloader')
        train_loader = None

    if cfg.PHASE in ['train', 'eval']:
        eval_dataset = build_dataset(cfg, 'val', writer=writer)
        eval_sampler = None
        eval_loader = DataLoader(eval_dataset, batch_size=cfg.VAL.BATCH_SIZE, shuffle=False, sampler=eval_sampler, **kwargs)
    else:
        print('Need not eval_loader')
        eval_loader = None

    if cfg.PHASE in ['train', 'pred']:
        test_dataset = build_dataset(cfg, 'test', writer=writer)
        test_loader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, **kwargs)
    else:
        print('Need not testloader')
        test_loader = None

    # model.eval()
    # TESTIMAGES = 10
    # diffs = []
    # for i in range(TESTIMAGES):
    #     data = train_dataset[i]
    #     image = data['img'].view(1, 3, 128, 128).to(device)
    #     ground_truth = data['joint_img']
    #     out = model(image)
    #     predict = out['joint_img'].cpu()[0]

    #     # print(f'GT: {ground_truth}')
    #     # print(f'PD: {predict}')
    #     diff = ((ground_truth - predict)**2).sum(axis=1).sqrt()  # sqrt. sum. (GT - p)^2
    #     print(f'Diff: {diff}')  # Diff: 0.1784597933292389
    #     diffs += [diff.detach().numpy()]
    # import numpy as np
    # diffs = np.stack(diffs)
    # print(f'max: {diffs.max()}, min: {diffs.min()}, mean: {diffs.mean()}')

    # #? EXP
    # exp_on(
    #     model=  model,
    #     datas=   [train_dataset[i] for i in (5, 10, 15, 20, 25, 30)],
    #     device= device
    # )
    # exit()

    # run
    runner = Runner(cfg, args, model, train_loader, eval_loader, test_loader, optimizer, writer, device, board, start_epoch=epoch)
    runner.run()

def exp_on(model, datas, device):
    # train_dataset.dbs[0].visualization(data, 0)
    model.eval()
    with torch.no_grad():
        for data in datas:
            out = model(data['img'][None].to(device))  # .cpu()
            # (#, 21, 2), (#, 21, 1)
            for key in out:
                out[key] = out[key].cpu()
            # exp_conf(data, out)
            exp_heatmap(data, out)

def confidence(**kwargs):
    # return (#, 21)
    distance = ((kwargs['joint_img_pred'] - kwargs['joint_img_gt'])**2).sum(axis=2).sqrt()
    # (#, 21)
    distance = distance.detach()
    conf_gt = 2 - 2 * torch.sigmoid(distance * 30)
    return conf_gt

def exp_conf(data, out):
    from matplotlib import pyplot as plt
    gt = confidence(joint_img_gt=data['joint_img'][None], joint_img_pred=out['joint_img'])
    pred = out['conf']
    gt = gt[0, :]
    pred = pred[0, :, 0]
    print('gt:', gt, 'pred:', pred, sep='\n')
    plt.plot(gt, label='gt')
    plt.plot(pred, label='pred')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

def exp_heatmap(data, out):
    '''
    data: {
        joint_img: [21, 2]
        img: [3, 128, 128]
    }
    out: {
        joint_img: [1, 21, 2]
        conf: [1, 21, 1]
    }
    edit uv2map() function for prefect heatmap width
    '''
    from matplotlib import pyplot as plt
    import numpy as np
    from utils.vis import uv2map, inv_base_tranmsform
    import vctoolkit as vc
    from my_research.tools.kinematics import MPIIHandJoints
    for k in data:
        data[k] = data[k].numpy()
    for k in out:
        out[k] = out[k].numpy()

    uv_gt = data['joint_img'] * 128     # gt
    uv_pd = out['joint_img'][0] * 128   # pred
    # uv_gt, uv_pd = uv_gt.int(), uv_pd.int()

    img = inv_base_tranmsform(data['img'])
    img_size = img.shape[:2]
    c_gt = confidence(joint_img_gt=torch.from_numpy(data['joint_img']),
                      joint_img_pred=torch.from_numpy(out['joint_img'])).numpy() # [#, 21]
    c_gt = c_gt[0, :, None, None]       # gt  : [21, 1, 1]

    c_pd = out['conf'][0, :, None]      # pred: [21, 1, 1]

    def sum_and_thrd(heat):
        # heat: (21, 128, 128)
        heat = heat.sum(axis=0)
        heat[heat>1] = 1
        return heat

    heat_gt = uv2map(uv_gt.astype('int'), img_size)  # (21, 128, 128)
    heat_pd = uv2map(uv_pd.astype('int'), img_size)  # (21, 128, 128)

    # hand img: 1
    ax = plt.subplot(2, 4, 1)
    ax.imshow(img)
    ax.set_title('img')
    ax.axis('off')

    # hand joint: 2
    ax = plt.subplot(2, 4, 2)
    vis_joint_img = vc.render_bones_from_uv(np.flip(uv_gt, axis=-1).copy(),
                                            img.copy(), MPIIHandJoints, thickness=2)
    ax.imshow(vis_joint_img)
    ax.set_title('joint + img gt')
    ax.axis('off')

    # hand joint: 6
    ax = plt.subplot(2, 4, 6)
    vis_joint_img = vc.render_bones_from_uv(np.flip(uv_pd, axis=-1).copy(),
                                            img.copy(), MPIIHandJoints, thickness=2)
    ax.imshow(vis_joint_img)
    ax.set_title('joint + img pred')
    ax.axis('off')

    # heat, pos: gt, conf: gt -> : 3
    ax = plt.subplot(2, 4, 3)
    heat = heat_gt * c_gt
    ax.imshow(sum_and_thrd(heat))
    ax.set_title('heat pos:gt, conf:gt')
    ax.axis('off')

    # heat, pos: gt, conf: pd -> : 4
    ax = plt.subplot(2, 4, 4)
    heat = heat_gt * c_pd
    ax.imshow(sum_and_thrd(heat))
    ax.set_title('heat pos:gt, conf:pd')
    ax.axis('off')

    # heat, pos: pd, conf: gt -> : 7
    ax = plt.subplot(2, 4, 7)
    heat = heat_pd * c_gt
    ax.imshow(sum_and_thrd(heat))
    ax.set_title('heat pos:pd, conf:gt')
    ax.axis('off')

    # heat, pos: pd, conf: pd -> : 8
    ax = plt.subplot(2, 4, 8)
    heat = heat_pd * c_pd
    ax.imshow(sum_and_thrd(heat))
    ax.set_title('heat pos:pd, conf:pd')
    ax.axis('off')

    plt.show()
    print()


if __name__ == "__main__":

    args = CFGOptions().parse()
    #! COMMENTED while train with scripts/
    # args.exp_name = 'test'  # folder
    # args.config_file = 'my_backbone/configs/densestack_conf.yml'
    main(args)
