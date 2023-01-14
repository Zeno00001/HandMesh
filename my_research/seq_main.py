import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_research.build import build_model, build_dataset
from my_research.configs.config import get_cfg
from options.cfg_options import CFGOptions
from my_research.seq_runner import Runner
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
    # from my_research.models.mobrecon_ds import MobRecon_DS
    # there is a python decorator inside
    # register MobRecon_DS automatically
    exec('from my_research.models.{} import {}'.format(cfg.MODEL.NAME.lower(), cfg.MODEL.NAME))

    # from my_research.datasets.hanco import HanCo
    exec('from my_research.datasets.{} import {}'.format(cfg.TRAIN.DATASET.lower(), cfg.TRAIN.DATASET))

    # dir
    args.work_dir = osp.dirname(osp.realpath(__file__))  # HandMesh/my_research
    args.out_dir = osp.join(args.work_dir, 'out', cfg.TRAIN.DATASET, args.exp_name)  # my_research/out/Multiple.../
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
                    model_path = osp.join(args.out_dir, '..', args.check_exp,
                                          'checkpoints', cfg.MODEL.RESUME)
            checkpoint = torch.load(model_path, map_location=device)
            # model.load_state_dict(checkpoint['model_state_dict'])
            missing, unexpected = model.load_state_dict(
                checkpoint['model_state_dict'], strict=False
            )
            print(f'missing    params: {chr(10).join(missing)}')  # chr(10) == '\n'
            print(f'unexpected params: {chr(10).join(unexpected)}')
            try:  # TODO: mini bug: not load optimizer state if MISMATCH
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                pass
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
            if args.exp_name == 'test':
                model_path = osp.join(args.out_dir, '..', args.check_exp,
                                      'checkpoints', cfg.MODEL.RESUME)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Eval model in {model_path}, with dataset: {cfg.VAL.DATASET}')
    else:
        input('[ERROR] wrong cfg PHASE while loading model')

    # data
    kwargs = {"pin_memory": True, "num_workers": 6, "drop_last": True}  # num_worker: 8
    if cfg.PHASE in ['train',]:
        train_dataset = build_dataset(cfg, phase='train', frame_counts=8, writer=writer)
        train_sampler = None
        train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=(train_sampler is None), sampler=train_sampler, **kwargs)
    else:
        print('Need not trainloader')
        train_loader = None

    if cfg.PHASE in ['train', 'eval']:
        eval_dataset = build_dataset(cfg, phase='val', frame_counts=8, writer=writer)
        eval_sampler = None
        eval_loader = DataLoader(eval_dataset, batch_size=cfg.VAL.BATCH_SIZE, shuffle=False, sampler=eval_sampler, **kwargs)
    else:
        print('Need not eval_loader')
        eval_loader = None

    if cfg.PHASE in ['train', 'pred']:
        test_dataset = build_dataset(cfg, phase='test', writer=writer)  # not need to provide frame_counts while testing
        test_loader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, **kwargs)
    else:
        print('Need not testloader')
        test_loader = None

    # check dataset here
    # img00028376 = train_dataset[28376] # mask file broken
    # img19502 = train_dataset[19502]
    # mesh154 = train_dataset[154]
    # mesh25265 = train_dataset[25265]
    # for i in range(len(train_dataset)):
    #     print(f'image: {i}')
    #     img_data = train_dataset[i]

    # for step, data in enumerate(train_loader):
    #     print(f'steps: {step: <5}/{len(train_loader)}')

    # input('Success!!!') # perfect
    # for data in train_loader:
    #     print(data['img'].shape)
    #     break
    # print()
    # input()

    # run
    runner = Runner(cfg, args, model, train_loader, eval_loader, test_loader, optimizer, writer, device, board, start_epoch=epoch)
    runner.run()


if __name__ == "__main__":

    args = CFGOptions().parse()
    # args.exp_name = 'test'
    # args.config_file = 'my_research/configs/mobrecon_ds_conf_transformer.yml'
    # args.check_exp = 'base_enc3_dec3_normtwice_reg_at_enc_DF_AR21Z_weightConf'
    main(args)
