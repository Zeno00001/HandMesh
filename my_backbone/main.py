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
                # model_path = osp.join(args.out_dir, '../densestack', 'checkpoints', cfg.MODEL.RESUME)
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
    # data = train_dataset[0]
    # image = data['img'].view(1, 3, 128, 128).to(device)
    # ground_truth = data['joint_img']
    # out = model(image)
    # predict = out['joint_img'].cpu()[0]

    # print(f'GT: {ground_truth}')
    # print(f'PD: {predict}')
    # diff = ((ground_truth - predict)**2).sum().sqrt()  # sqrt. sum. (GT - p)^2
    # print(f'Diff: {diff}')  # Diff: 0.1784597933292389

    # exit()

    # run
    runner = Runner(cfg, args, model, train_loader, eval_loader, test_loader, optimizer, writer, device, board, start_epoch=epoch)
    runner.run()


if __name__ == "__main__":

    args = CFGOptions().parse()
    args.exp_name = 'test'  # folder
    args.config_file = 'my_backbone/configs/densestack.yml'
    main(args)
