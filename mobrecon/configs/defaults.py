from .config import CfgNode as CN
import os


_C = CN()
_C.VERSION = 0.1
_C.PHASE = 'train'

_C.MODEL = CN()
_C.MODEL.NAME = 'MobRecon_DS'
_C.MODEL.MANO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../template'))
_C.MODEL.RESUME = ''
_C.MODEL.KPTS_NUM = 21
_C.MODEL.LATENT_SIZE = 256

_C.MODEL.SPIRAL = CN()
_C.MODEL.SPIRAL.TYPE = 'Conv'
_C.MODEL.SPIRAL.OUT_CHANNELS = [32, 64, 128, 256]
_C.MODEL.SPIRAL.DOWN_SCALE = [2, 2, 2, 2]
_C.MODEL.SPIRAL.LEN = [9, 9, 9, 9]
_C.MODEL.SPIRAL.DILATION = [1, 1, 1, 1]

_C.DATA = CN()
_C.DATA.SIZE = 128
_C.DATA.STD = 3
_C.DATA.IMG_MEAN = 0.5
_C.DATA.IMG_STD = 0.5
_C.DATA.COLOR_AUG = True
_C.DATA.CONTRASTIVE = False

_C.DATA.FREIHAND = CN()
_C.DATA.FREIHAND.USE = True
_C.DATA.FREIHAND.ROOT = 'data/FreiHAND'
_C.DATA.FREIHAND.ROT = 90
_C.DATA.FREIHAND.SCALE = 0.2
_C.DATA.FREIHAND.BASE_SCALE = 1.3
_C.DATA.FREIHAND.FLIP = False

_C.DATA.GE = CN()
_C.DATA.GE.USE = True
_C.DATA.GE.ROOT = 'data/Ge'
_C.DATA.GE.BASE_SCALE = 1.3

_C.DATA.COMPHAND = CN()
_C.DATA.COMPHAND.USE = True
_C.DATA.COMPHAND.ROOT = 'data/CompHand'
_C.DATA.COMPHAND.ROT = 90
_C.DATA.COMPHAND.SCALE = 0.2
_C.DATA.COMPHAND.BASE_SCALE = 1.3
_C.DATA.COMPHAND.FLIP = False

_C.DATA.HANCO = CN()
_C.DATA.HANCO.USE = True
_C.DATA.HANCO.ROOT = 'data/HanCo'
_C.DATA.HANCO.ROT = 90
_C.DATA.HANCO.SCALE = 0.2
_C.DATA.HANCO.BASE_SCALE = 1.3
_C.DATA.HANCO.FLIP = False

_C.TRAIN = CN()
_C.TRAIN.DATASET = 'FreiHAND'
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_DECAY = 0.1
_C.TRAIN.DECAY_STEP = [30, ]
_C.TRAIN.WARMUP_EPOCHS = 0
_C.TRAIN.WEIGHT_DECAY = 0
_C.TRAIN.EPOCHS = 38
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.GPU_ID = [0, ]

_C.VAL = CN()
_C.VAL.DATASET = 'Ge'
_C.VAL.BATCH_SIZE = 1
_C.VAL.SAVE_DIR = 'eval'
_C.VAL.SAVE_PRED = False

_C.TEST = CN()
_C.TEST.DATASET = 'FreiHAND'
_C.TEST.BATCH_SIZE = 1
_C.TEST.SAVE_DIR = 'test'
_C.TEST.SAVE_PRED = False

