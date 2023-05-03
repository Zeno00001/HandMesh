import os
import numpy as np
import time
import torch
import cv2
import json
from utils.warmup_scheduler import adjust_learning_rate
from utils.vis import inv_base_tranmsform
from utils.zimeval import EvalUtil
from utils.transforms import rigid_align
from my_research.tools.vis import perspective, compute_iou, cnt_area
from my_research.tools.kinematics import mano_to_mpii, MPIIHandJoints
from my_research.tools.registration import registration
import vctoolkit as vc

from einops import rearrange


class Runner(object):
    def __init__(self, cfg, args, model, train_loader, val_loader, test_loader, optimizer, writer, device, board, start_epoch=0):
        super(Runner, self).__init__()
        self.cfg = cfg
        self.args = args
        self.model = model
        face = np.load(os.path.join(cfg.MODEL.MANO_PATH, 'right_faces.npy'))
        self.face = torch.from_numpy(face).long()
        self.j_reg = np.load(os.path.join(self.cfg.MODEL.MANO_PATH, 'j_reg.npy'))
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.max_epochs = cfg.TRAIN.EPOCHS
        self.optimizer = optimizer
        self.writer = writer
        self.device = device
        self.board = board
        self.start_epoch = start_epoch
        self.epoch = max(start_epoch - 1, 0)
        if cfg.PHASE == 'train':
            self.total_step = self.start_epoch * (len(self.train_loader.dataset) // cfg.TRAIN.BATCH_SIZE)
            try:
                self.loss = self.model.loss
            except:
                self.loss = self.model.module.loss
        self.best_val_loss = np.float('inf')
        print('runner init done')

    def run(self):
        if self.cfg.PHASE == 'train':
            if self.val_loader is not None and self.epoch > 0:
                if self.args.exp_name != 'test':
                    self.best_val_loss = self.eval()
                    print(f'Epoch: {self.epoch}, PA-MPJPE: {self.best_val_loss}')  # Reg(vert) <-> gt_joint
            for epoch in range(self.start_epoch, self.max_epochs + 1):
                self.epoch = epoch
                t = time.time()
                if self.args.world_size > 1:
                    self.train_loader.sampler.set_epoch(epoch)
                train_loss = self.train()
                t_duration = time.time() - t
                if self.val_loader is not None:
                    val_loss = self.eval()
                else:
                    val_loss = np.float('inf')

                info = {
                    'current_epoch': self.epoch,
                    'epochs': self.max_epochs,
                    'train_loss': train_loss,
                    'test_loss': val_loss,
                    't_duration': t_duration
                }
                print(f'epoch: {self.epoch: 3} / {self.max_epochs}')

                self.writer.print_info(info)
                if val_loss < self.best_val_loss:
                    self.writer.save_checkpoint(self.model, self.optimizer, None, self.epoch, best=True)
                    self.best_test_loss = val_loss
                self.writer.save_checkpoint(self.model, self.optimizer, None, self.epoch, last=True)
            # self.pred()
        elif self.cfg.PHASE == 'eval':
            self.eval()
        elif self.cfg.PHASE == 'pred':
            self.pred()
        elif self.cfg.PHASE == 'test':
            self.test()  # evaluate on HanCo test set
        elif self.cfg.PHASE == 'demo':
            self.demo()
        else:
            raise Exception('PHASE ERROR')

    def phrase_data(self, data):
        '''
        將 data[each key] 中的每一筆 data 都傳到 .to('GPU') 裡面
        '''
        for key, val in data.items():
            try:
                if isinstance(val, list):
                    data[key] = [d.to(self.device) for d in data[key]]
                else:
                    data[key] = data[key].to(self.device)
            except:
                pass
        return data

    def board_scalar(self, phase, n_iter, lr=None, **kwargs):
        split = '/'
        for key, val in kwargs.items():
            if 'loss' in key:
                if isinstance(val, torch.Tensor):
                    val = val.item()
                self.board.add_scalar(phase + split + key, val, n_iter)
            if lr:
                self.board.add_scalar(phase + split + 'lr', lr, n_iter)

    def draw_results(self, data, out, loss, batch_id, aligned_verts=None):
        img_cv2 = inv_base_tranmsform(data['img'][batch_id].cpu().numpy())[..., :3]
        draw_list = []
        if 'joint_img' in data:
            draw_list.append( vc.render_bones_from_uv(np.flip(data['joint_img'][batch_id, :, :2].cpu().numpy()*self.cfg.DATA.SIZE, axis=-1).copy(),
                                                      img_cv2.copy(), MPIIHandJoints, thickness=2) )
        if 'joint_img' in out:
            try:
                draw_list.append( vc.render_bones_from_uv(np.flip(out['joint_img'][batch_id, :, :2].detach().cpu().numpy()*self.cfg.DATA.SIZE, axis=-1).copy(),
                                                         img_cv2.copy(), MPIIHandJoints, thickness=2) )
            except:
                draw_list.append(img_cv2.copy())
        if 'root' in data:
            root = data['root'][batch_id:batch_id+1, :3]
        else:
            root = torch.FloatTensor([[0, 0, 0.6]]).to(data['img'].device)
        if 'verts' in data:
            vis_verts_gt = img_cv2.copy()
            verts = data['verts'][batch_id:batch_id+1, :, :3] * 0.2 + root
            vp = perspective(verts.permute(0, 2, 1), data['calib'][batch_id:batch_id+1, :4])[0].cpu().numpy().T
            for i in range(vp.shape[0]):
                cv2.circle(vis_verts_gt, (int(vp[i, 0]), int(vp[i, 1])), 1, (255, 0, 0), -1)
            draw_list.append(vis_verts_gt)
        if 'verts' in out:
            try:
                vis_verts_pred = img_cv2.copy()
                if aligned_verts is None:
                    verts = out['verts'][batch_id:batch_id+1, :, :3] * 0.2 + root
                else:
                    verts = aligned_verts
                vp = perspective(verts.permute(0, 2, 1), data['calib'][batch_id:batch_id+1, :4])[0].detach().cpu().numpy().T
                for i in range(vp.shape[0]):
                    cv2.circle(vis_verts_pred, (int(vp[i, 0]), int(vp[i, 1])), 1, (255, 0, 0), -1)
                draw_list.append(vis_verts_pred)
            except:
                draw_list.append(img_cv2.copy())

        return np.concatenate(draw_list, 1)

    def board_img(self, phase, n_iter, data, out, loss, batch_id=0):
        draw = self.draw_results(data, out, loss, batch_id)
        self.board.add_image(phase + '/res', draw.transpose(2, 0, 1), n_iter)

    def train(self):
        self.writer.print_str('TRAINING ..., Epoch {}/{}'.format(self.epoch, self.max_epochs))
        self.model.train()  # set to eval to avoid dropouted attention map
        if self.args.exp_name == 'test':
            self.model.eval()
        total_loss = 0
        forward_time = 0.
        backward_time = 0.
        start_time = time.time()
        for step, data in enumerate(self.train_loader):
            ts = time.time()
            adjust_learning_rate(self.optimizer, self.epoch, step, len(self.train_loader), self.cfg.TRAIN.LR, self.cfg.TRAIN.LR_DECAY, self.cfg.TRAIN.DECAY_STEP, self.cfg.TRAIN.WARMUP_EPOCHS)
            data = self.phrase_data(data)  # to('GPU')
            self.optimizer.zero_grad()
            out = self.model(data['img'])
            # self.draw_eval_results(self._reshape_BF_to_B(data), self._reshape_BF_to_B(out))
            tf = time.time()
            forward_time += tf - ts
            losses = self.loss(verts_pred=out.get('verts'),
                               joint_img_pred=out['joint_img'],
                               joint_conf_pred=out['joint_conf'],  # append conf prediciton
                               joint_3d_pred=out.get('joints'),    # append joint prediction

                               verts_gt=data.get('verts'),
                               joint_img_gt=data['joint_img'],
                               joint_3d_gt=data.get('joint_cam'),  # append joint root-relative

                               face=self.face,
                               aug_param=(None, data.get('aug_param'))[self.epoch>4],
                               bb2img_trans=data.get('bb2img_trans'),
                               size=data['img'].size(2),
                               mask_gt=data.get('mask'),
                               trans_pred=out.get('trans'),
                               alpha_pred=out.get('alpha'),
                               img=data.get('img'))
            loss = losses['loss']
            loss.backward()
            self.optimizer.step()
            tb = time.time()
            backward_time +=  tb - tf

            self.total_step += 1
            total_loss += loss.item()
            if self.board is not None:
                self.board_scalar('train', self.total_step, self.optimizer.param_groups[0]['lr'], **losses)
            if self.total_step % 100 == 0:
                cur_time = time.time()
                duration = cur_time - start_time
                start_time = cur_time
                info = {
                    'train_loss': loss.item(),
                    'l1_loss': losses.get('verts_loss', 0),
                    'epoch': self.epoch,
                    'max_epoch': self.max_epochs,
                    'step': step,
                    'max_step': len(self.train_loader),
                    'total_step': self.total_step,
                    'step_duration': duration,
                    'forward_duration': forward_time,
                    'backward_duration': backward_time,
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                self.writer.print_step_ft(info)
                forward_time = 0.
                backward_time = 0.

        if self.board is not None:
            self.board_img('train', self.epoch, self._reshape_BF_to_B(data), self._reshape_BF_to_B(out), losses)

        return total_loss / len(self.train_loader)

    def _reshape_BF_to_B(self, data):
        '''
        reshape data = {
            'tensor_1' -> (4, 8, 3, 128, 128)
            'tensor_2' -> (4, 8, 4, 4)
            ...
        }

        reshape all tensors from (B, F, ...) into (BF, ...)
        '''
        out = {}
        for key in data:
            shape = data[key].shape
            if len(shape) == 3:
                # (B, F, D)
                out[key] = rearrange(data[key], 'B F D -> (B F) D')
            elif len(shape) == 4:
                # (B, F, J, D)
                out[key] = rearrange(data[key], 'B F J D -> (B F) J D')
            elif len(shape) == 5:
                # (B, F, D, W, H)
                out[key] = rearrange(data[key], 'B F D H W -> (B F) D H W')
            elif len(shape) == 1:
                # case: data['start']
                pass
            else:
                raise RuntimeError(f'len(shape) out of range: {shape}')
        return out

    def draw_eval_results(self, data, out, mpjpe=None, pampjpe=None):
        ''' draw image of shape: (128 * F, 512, 3)
            data, out: (1xF, ...) '''
        imgs = []
        for i in range(8):
            imgs += [self.draw_results(data, out, {}, i)]  # (128, 512, 3)
        imgs = rearrange(imgs, 'F H W D -> (F H) W D')
        from matplotlib import pyplot as plt
        mpjpe = None if mpjpe is None else ' |'.join(f'{e.mean(): 6.2f}' for e in mpjpe)
        pampjpe = None if pampjpe is None else ' |'.join(f'{e.mean(): 6.2f}' for e in pampjpe)
        mpjpe = f'mpjpe: {mpjpe}' if mpjpe is not None else 'mpjpe: Nope'
        pampjpe = f'pampjpe: {pampjpe}' if pampjpe is not None else 'pampjpe: Nope'
        plt.imshow(imgs)
        plt.title(mpjpe + '\n' + pampjpe)
        plt.show()

    def eval(self):
        self.writer.print_str('EVALING ... Epoch {}/{}'.format(self.epoch, self.max_epochs))
        self.model.eval()
        evaluator_2d = EvalUtil()
        evaluator_rel = EvalUtil()
        evaluator_pa = EvalUtil()
        mask_iou = []
        joint_cam_errors = []
        pa_joint_cam_errors = []
        joint_img_errors = []
        with torch.no_grad():
            for step, data in enumerate(self.val_loader):
                if self.board is None and step % 100 == 0:
                    print(step, len(self.val_loader))
                # get data then infernce
                data = self.phrase_data(data)
                out = self.model(data['img'])
                # self.draw_eval_results(self._reshape_BF_to_B(data), self._reshape_BF_to_B(out))
                for frame_id in range(out['verts'].shape[1]):
                    # get vertex pred
                    verts_pred = out['verts'][0][frame_id].cpu().numpy() * 0.2  # batch:0, frame: frame_id
                    joint_cam_pred = mano_to_mpii(np.matmul(self.j_reg, verts_pred)) * 1000.0

                    # get mask pred
                    mask_pred = out.get('mask')  # [frame_id], None Obj
                    if mask_pred is not None:
                        mask_pred = (mask_pred[0] > 0.3).cpu().numpy().astype(np.uint8)
                        mask_pred = cv2.resize(mask_pred, (data['img'].size(3+1), data['img'].size(2+1)))
                    else:
                        mask_pred = np.zeros((data['img'].size(3+1), data['img'].size(2+1)), np.uint8)

                    # get uv pred
                    joint_img_pred = out.get('joint_img')[:, frame_id]
                    if joint_img_pred is not None:
                        joint_img_pred = joint_img_pred[0].cpu().numpy() * data['img'].size(3)  # BCHW -> BFCHW
                    else:
                        joint_img_pred = np.zeros((21, 2), dtype=np.float)

                    # pck
                    joint_cam_gt = data['joint_cam'][0][frame_id].cpu().numpy() * 1000.0
                    joint_cam_align = rigid_align(joint_cam_pred, joint_cam_gt)
                    evaluator_2d.feed(data['joint_img'][0][frame_id].cpu().numpy() * data['img'].size(2+1), joint_img_pred)
                    evaluator_rel.feed(joint_cam_gt, joint_cam_pred)
                    evaluator_pa.feed(joint_cam_gt, joint_cam_align)

                    # error
                    if 'mask_gt' in data.keys():
                        mask_iou.append(compute_iou(mask_pred, cv2.resize(data['mask_gt'][0][frame_id].cpu().numpy(), (data['img'].size(3+1), data['img'].size(2+1)))))
                    else:
                        mask_iou.append(0)
                    joint_cam_errors.append(np.sqrt(np.sum((joint_cam_pred - joint_cam_gt) ** 2, axis=1)))
                    pa_joint_cam_errors.append(np.sqrt(np.sum((joint_cam_gt - joint_cam_align) ** 2, axis=1)))
                    joint_img_errors.append(np.sqrt(np.sum((data['joint_img'][0][frame_id].cpu().numpy()*data['img'].size(2+1) - joint_img_pred) ** 2, axis=1)))
                # self.draw_eval_results(self._reshape_BF_to_B(data), self._reshape_BF_to_B(out), joint_cam_errors[-8:], pa_joint_cam_errors[-8:])

            # get auc
            _1, _2, _3, auc_rel, pck_curve_rel, thresholds2050 = evaluator_rel.get_measures(20, 50, 20)
            _1, _2, _3, auc_pa, pck_curve_pa, _ = evaluator_pa.get_measures(20, 50, 20)
            _1, _2, _3, auc_2d, pck_curve_2d, _ = evaluator_2d.get_measures(0, 30, 20)
            # get error
            miou = np.array(mask_iou).mean()
            mpjpe = np.array(joint_cam_errors).mean()
            pampjpe = np.array(pa_joint_cam_errors).mean()
            print(f'pampjpe STD: {np.array(pa_joint_cam_errors).std()}')
            uve = np.array(joint_img_errors).mean()

            if self.board is not None:
                self.board_scalar('test', self.epoch, **{'auc_loss': auc_rel, 'pa_auc_loss': auc_pa, '2d_auc_loss': auc_2d, 'mIoU_loss': miou, 'uve': uve, 'mpjpe_loss': mpjpe, 'pampjpe_loss': pampjpe})
                self.board_img('test', self.epoch, self._reshape_BF_to_B(data), self._reshape_BF_to_B(out), {})
            elif self.args.world_size < 2:
                print( f'pampjpe: {pampjpe}, mpjpe: {mpjpe}, uve: {uve}, miou: {miou}, auc_rel: {auc_rel}, auc_pa: {auc_pa}, auc_2d: {auc_2d}')
                print('thresholds2050', thresholds2050)
                print('pck_curve_all_pa', pck_curve_pa)
            self.writer.print_str( f'pampjpe: {pampjpe}, mpjpe: {mpjpe}, uve: {uve}, miou: {miou}, auc_rel: {auc_rel}, auc_pa: {auc_pa}, auc_2d: {auc_2d}')

        return pampjpe

    def pred(self):
        self.writer.print_str('PREDICING ... Epoch {}/{}'.format(self.epoch, self.max_epochs))
        self.model.eval()
        xyz_pred_list, verts_pred_list = list(), list()
        with torch.no_grad():
            for step, data in enumerate(self.test_loader):
                if self.board is None and step % 100 == 0:
                    print(step, len(self.test_loader))
                # print(f'Eval on image[{data["idx"].cpu()}]')
                data = self.phrase_data(data)
                out = self.model(data['img'])
                # EXP
                # print(f'input      : {data["img"].size()}')         # (1, 3, 128, 128)
                # print(f'output:')
                # print(f'       vert: {out["verts"].size()}')        # (1, 778, 3)
                # print(f'  joint_img: {out["joint_img"].size()}')    # (1, 21, 2)
                # data['img'].permute((0, 2, 3, 1)).reshape((128, 128, 3))
                # np.save('EXP_pred/img_new.npy', data['img'].permute((0, 2, 3, 1)).reshape((128, 128, 3)).cpu().numpy())
                # np.save('EXP_pred/out_vert_new.npy', out['verts'][0].cpu().numpy())
                # np.save('EXP_pred/out_joint_new.npy', out['joint_img'][0].cpu().numpy())

                # np.save('EXP_pred/regressor_new', self.j_reg)
                # return

                # get verts pred
                verts_pred = out['verts'][0].cpu().numpy() * 0.2 # old line: 195

                # get mask pred
                mask_pred = out.get('mask')
                if mask_pred is not None:
                    mask_pred = (mask_pred[0] > 0.3).cpu().numpy().astype(np.uint8)
                    mask_pred = cv2.resize(mask_pred, (data['img'].size(3), data['img'].size(2)))
                    try:
                        contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        contours.sort(key=cnt_area, reverse=True)
                        poly = contours[0].transpose(1, 0, 2).astype(np.int32)
                    except:
                        poly = None
                else:
                    poly = None

                # get uv pred
                joint_img_pred = out.get('joint_img')
                if joint_img_pred is not None:
                    joint_img_pred = joint_img_pred[0].cpu().numpy() * data['img'].size(2)
                    verts_pred, align_state = registration(verts_pred, joint_img_pred, self.j_reg, data['calib'][0].cpu().numpy(), self.cfg.DATA.SIZE, poly=poly)

                # get joint_cam
                joint_cam_pred = mano_to_mpii(np.matmul(self.j_reg, verts_pred))

                # np.save('EXP_pred/joint_cam_pred_new.npy', joint_cam_pred)
                # np.save('EXP_pred/verts_pred_new.npy', verts_pred)
                # raise Exception('Hello runner')
                # track data
                xyz_pred_list.append(joint_cam_pred)
                verts_pred_list.append(verts_pred)
                if self.cfg.TEST.SAVE_PRED:
                    draw = self.draw_results(data, out, {}, 0, aligned_verts=torch.from_numpy(verts_pred).float()[None, ...])[..., ::-1]
                    cv2.imwrite(os.path.join(self.args.out_dir, self.cfg.TEST.SAVE_DIR, f'{step}.png'), draw)

        # dump results
        xyz_pred_list = [x.tolist() for x in xyz_pred_list]
        verts_pred_list = [x.tolist() for x in verts_pred_list]
        # save to a json
        with open(os.path.join(self.args.out_dir, f'{self.args.exp_name}.json'), 'w') as fo:
            json.dump(
                [
                    xyz_pred_list,
                    verts_pred_list
                ], fo)
        self.writer.print_str('Dumped %d joints and %d verts predictions to %s' % (
            len(xyz_pred_list), len(verts_pred_list), os.path.join(self.args.work_dir, 'out', self.args.exp_name, f'{self.args.exp_name}.json')))

    def test(self):
        ''' evaluate on test set
            all predictions and ground-truth are in the unit of "meter"
        '''
        self.writer.print_str('NEW SeqMode: TESTING ... Epoch {}/{}'.format(self.epoch, self.max_epochs))
        self.model.eval()
        forward_time = registration_time = scoring_time = 0
        with torch.no_grad():
            overall_joint_cam_errors = []
            overall_verts_cam_errors = []
            overall_pa_joint_cam_errors = []
            overall_pa_verts_cam_errors = []

            for step, data in enumerate(self.test_loader):
                if self.board is None and step % 10 == 0:
                    print(step, len(self.test_loader))

                xyz_pred_list = []
                verts_pred_list = []
                image_width = data['img'].size(3) # in (B, F, 3, H, W)
                frame_len = data['img'].size(1)
                seq_id = data['seq_id'].item()

                t = time.time()
                data = self.phrase_data(data)

                # save in scale of meter
                prediction_path = os.path.join(self.args.out_dir, self.cfg.TEST.SAVE_DIR, f'{seq_id:04d}_0.npz') # cam: 0
                if os.path.isfile(prediction_path):
                    # skip if calculated already
                    np_data = np.load(prediction_path)
                    xyz_pred_list = np_data['joint_3d']
                    verts_pred_list = np_data['verts_3d']
                else:
                    # data: (B, F, ...), where B=1 for testing
                    out = self.seq_pred_one_clip(self.model, data['img'])
                    forward_time += time.time() - t
                    t = time.time()
                    # out: {verts: (B=1, F, 778, 3), joint_img: (B=1,Ｆ，　２１，　２)}, and ignoring joint_conf, joints

                    # all frames processing
                    # get verts pred
                    verts_pred = out['verts'][0].cpu().numpy() * 0.2  # into (F, 778, 3)
                    # get mask pred
                    poly = None
                    # get uv pred
                    joint_img_pred = out['joint_img'][0].cpu().numpy() * image_width  # into (F, 21, 2)

                    # get calib data
                    calib_info = data['calib'][0].cpu().numpy()  # into (F, 4, 4)

                    # per frame processing
                    for frame_i in range(frame_len):
                        # frame pick
                        f_verts_pred = verts_pred[frame_i]
                        f_joint_img_pred = joint_img_pred[frame_i]
                        f_calib_info = calib_info[frame_i]

                        # registration
                        f_verts_pred, align_state = registration(f_verts_pred, f_joint_img_pred, self.j_reg, f_calib_info, self.cfg.DATA.SIZE, poly=poly)
                        # get joint_cam
                        f_joint_cam_pred = mano_to_mpii(np.matmul(self.j_reg, f_verts_pred))
                        # track data
                        xyz_pred_list.append(f_joint_cam_pred)
                        verts_pred_list.append(f_verts_pred)

                        # 2D joints drawing
                        if self.cfg.TEST.SAVE_PRED:
                            draw = self.draw_results(data, out, {}, 0, aligned_verts=torch.from_numpy(verts_pred).float()[None, ...])[..., ::-1]
                            cv2.imwrite(os.path.join(self.args.out_dir, self.cfg.TEST.SAVE_DIR, f'{step}.png'), draw)

                    xyz_pred_list   = rearrange(xyz_pred_list, 'F J D -> F J D')  # into one numpy arr
                    verts_pred_list = rearrange(verts_pred_list, 'F V D -> F V D')

                    # save in scale of meter
                    np.savez(prediction_path, joint_3d=xyz_pred_list, verts_3d=verts_pred_list)
                    registration_time += time.time() - t
                    t = time.time()

                # get joint, verts prediction in cam coord \
                #   in {xyz_pred_list}, {verts_pred_list}

                # per frame scoring
                # data['joint_cam']:    B=1 F J D   in GPU  in meter
                # data['verts']:        B=1 F V D   in GPU  in meter
                root_gt_list = data['root'][0].cpu().numpy()
                root_gt_list = rearrange(root_gt_list, 'F D -> F () D')
                xyz_gt_list   = data['joint_cam'][0].cpu().numpy() + root_gt_list
                verts_gt_list = data['verts'][0].cpu().numpy() + root_gt_list
                xyz_pred_list, xyz_gt_list     = xyz_pred_list * 1000, xyz_gt_list * 1000
                verts_pred_list, verts_gt_list = verts_pred_list * 1000, verts_gt_list * 1000
                for frame_i in range(frame_len):
                    # xyz_pred_list:    F J D   in CPU  in millimeter
                    # verts_pred_list:  F V D   in CPU  in millimeter
                    # xyz_gt_list:      F J D   in CPU  in millimeter
                    # verts_gt_list:    F V D   in CPU  in millimeter
                    f_joint_cam_pred = xyz_pred_list[frame_i]
                    f_joint_cam_gt   = xyz_gt_list[frame_i]
                    f_verts_cam_pred = verts_pred_list[frame_i]
                    f_verts_cam_gt   = verts_gt_list[frame_i]
                    f_joint_cam_align = rigid_align(f_joint_cam_pred, f_joint_cam_gt)
                    f_verts_cam_align = rigid_align(f_verts_cam_pred, f_verts_cam_gt)
                    overall_joint_cam_errors.append(
                        np.sqrt(np.sum(np.square(f_joint_cam_gt - f_joint_cam_pred), axis=1))
                    )
                    overall_pa_joint_cam_errors.append(
                        np.sqrt(np.sum(np.square(f_joint_cam_gt - f_joint_cam_align), axis=1))
                    )
                    overall_verts_cam_errors.append(
                        np.sqrt(np.sum(np.square(f_verts_cam_gt - f_verts_cam_pred), axis=1))
                    )
                    overall_pa_verts_cam_errors.append(
                        np.sqrt(np.sum(np.square(f_verts_cam_gt - f_verts_cam_align), axis=1))
                    )

            mpjpe = np.array(overall_joint_cam_errors).mean()
            pampjpe = np.array(overall_pa_joint_cam_errors).mean()
            mpvpe = np.array(overall_verts_cam_errors).mean()
            pampvpe = np.array(overall_pa_verts_cam_errors).mean()

            print(f'MPJPE: {mpjpe} mm, PA-MPJPE: {pampjpe} mm')
            print(f'MPVPE: {mpvpe} mm, PA-MPVPE: {pampvpe} mm')

            score_path = os.path.join(self.args.out_dir, self.cfg.TEST.SAVE_DIR, f'score.txt')
            with open(score_path, 'w') as fo:
                fo.write(f'MPJPE: {mpjpe} mm\n')
                fo.write(f'PA-MPJPE: {pampjpe} mm\n')
                fo.write(f'MPVPE: {mpvpe} mm\n')
                fo.write(f'PA-MPVPE: {pampvpe} mm\n')

            scoring_time += time.time() - t
            print(f'\nForTime: {forward_time}')
            print(f'RegisTime: {registration_time}')
            print(f'ScoreTime: {scoring_time}')

        # end of test()

    def seq_pred_one_clip(self, model, data, win_len=8, win_stride=4):
        '''
        ! this function can handle default window value only for now !

        using {model} to inference all images in {clip_imgs} and output prediction result from {model}
        @ model:        the nn.Module of overall model                 send to GPU
        @ clip_imgs:    the nn.Tensor with shape (1, F, 3, 128, 128)   send to GPU
        @ return:       the dict of nn.Tensor                               in GPU
            out = {
                'verts': (1, F, 778, 3),
                'joint_img': (1, F, 21, 2)
            }

        example
        [ 1 2 3 4 5 6 7 8 9 0 1 2 3 ], with default window
        | v v v v v v - - |                 in first window prediction
                | - - v v v v - - |         in second window prediction
                  | - - - - - v v v |       in last window prediction
        '''
        # model in              : (B=1, F=8, 3, 128, 128)
        # model out['verts']    : (B=1, F=8, 778, 3)
        # model out['joint_img']: (B=1, F=8, 21, 2)
        clip_len = data.shape[1]
        out = {
            'verts': [None] * clip_len,
            'joint_img': [None] * clip_len,
        }
        win_start = 0
        while win_start + win_len < clip_len:
            win_data = data[:, win_start : win_start + 8]  # first ':' to reserve Batch dimension
            win_out = model(win_data)
            win_out_verts = win_out['verts'][0]         # into (F=8, 778, 3)
            win_out_joint_img = win_out['joint_img'][0] # into (F=8, 21, 3)

            if win_start == 0:
                out['verts'][0],     out['verts'][1]     = win_out_verts[0],     win_out_verts[1]
                out['joint_img'][0], out['joint_img'][1] = win_out_joint_img[0], win_out_joint_img[1]
            for i in range(2, 6):
                out['verts'][win_start + i]     = win_out_verts[i]
                out['joint_img'][win_start + i] = win_out_joint_img[i]

            win_start += win_stride

        if out['verts'][-1] is None:
            win_data = data[:, clip_len-8:]
            win_out = model(win_data)
            win_out_verts = win_out['verts'][0]
            win_out_joint_img = win_out['joint_img'][0]
            for i in range(1, 9):
                # [len-1, len-2, ..., len-8]
                if out['verts'][clip_len - i] is not None:
                    break
                # else
                out['verts'][clip_len - i]     = win_out_verts[8 - i]
                out['joint_img'][clip_len - i] = win_out_joint_img[8 - i]

        out['verts']     = rearrange(out['verts'], 'F V D -> () F V D')
        out['joint_img'] = rearrange(out['joint_img'], 'F J D -> () F J D')
        return out

    def set_demo(self, args):
        import pickle
        with open(os.path.join(args.work_dir, '../template/MANO_RIGHT.pkl'), 'rb') as f:
            mano = pickle.load(f, encoding='latin1')
        self.j_regressor = np.zeros([21, 778])
        self.j_regressor[:16] = mano['J_regressor'].toarray()
        for k, v in {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}.items():
            self.j_regressor[k, v] = 1
        self.std = torch.tensor(0.20)

    def demo(self):
        from utils.progress.bar import Bar
        from termcolor import colored
        from utils.vis import registration, map2uv, base_transform
        from utils.draw3d import save_a_image_with_mesh_joints
        from utils.read import save_mesh
        self.set_demo(self.args)

        INFER_LIST = ['01M', '01R', '01U', '03M', '03R', '03U', '05M', '05R', '05U', '07M', '07R', '07U', '09M', '09R', '09U', '21M', '21R', '21U', '23M', '23R', '23U', '25M', '25R', '25U', '27M', '27R', '27U', '29M', '29R', '29U', '41M', '41R', '41U', '43M', '43R', '43U', '45M', '45R', '45U', '47M', '47R', '47U', '49M', '49R', '49U']

        for i in range(len(INFER_LIST)):
            INFER_FOLDER = INFER_LIST[i]
            print(f'Predicting {INFER_FOLDER}')

            args = self.args
            args.size = 128  # NEW APPEND
            self.model.eval()
            # image_fp = os.path.join(args.work_dir, 'images')
            image_fp = os.path.join(args.work_dir, 'images', INFER_FOLDER)
            output_fp = os.path.join(args.out_dir, 'demo', INFER_FOLDER)
            os.makedirs(output_fp, exist_ok=True)
            ''' paths
            input : ~/HandMesh/images/{INFER_FOLDER}/
            output: ~/HandMesh/out/FreiHAND/mrc_ds/demo/{INFER_FOLDER} /
            '''

            # image_files = [os.path.join(image_fp, i) for i in os.listdir(image_fp) if '_img.jpg' in i]
            image_files = [os.path.join(image_fp, e) for e in os.listdir(image_fp) if e.endswith('.jpg')]  # or jpg...
            bar = Bar(colored("DEMO", color='blue'), max=len(image_files))
            with torch.no_grad():
                for step, image_path in enumerate(image_files):
                    # EXP
                    # print('TPYE', type(self.face))
                    # np.save('EXP_demo/face.npy', self.face.cpu().detach().numpy())
                    # return
                    # print(f'Demo on: {image_path}')
                    # image_path = '/home/oscar/Desktop/HandMesh/my_research/images/0_stone/image.jpg'
                    # EXP

                    # image_name = image_path.split('/')[-1].split('_')[0]
                    image_name = os.path.basename(image_path).split('.')[0]  # '0000'
                    image = cv2.imread(image_path)[..., ::-1]
                    image = cv2.resize(image, (args.size, args.size))
                    input = torch.from_numpy(base_transform(image, size=args.size)).unsqueeze(0).to(self.device)

                    # print(f'processing file: {image_path}')
                    _Knpy_file_path = image_path.replace('_img.jpg', '_K.npy')
                    if os.path.isfile(_Knpy_file_path) and _Knpy_file_path.endswith('_K.npy'):  # example images' K
                        K = np.load(_Knpy_file_path)
                    elif os.path.isfile(os.path.join(args.work_dir, 'images', 'default.npy')):  # my images' K
                        K = np.load(os.path.join(args.work_dir, 'images', 'default.npy'))

                    K[0, 0] = K[0, 0] / 224 * args.size
                    K[1, 1] = K[1, 1] / 224 * args.size
                    K[0, 2] = args.size // 2
                    K[1, 2] = args.size // 2

                    out = self.model(input)
                    # print(f'input      : {input.size()}')         # (1, 3, 128, 128)
                    # print(f'output:')
                    # print(f'       vert: {out["verts"].size()}')        # (1, 778, 3)
                    # print(f'  joint_img: {out["joint_img"].size()}')    # (1, 21, 2)
                    # np.save('EXP_demo/in.npy', input.cpu().detach().numpy())
                    # np.save('EXP_demo/vert.npy', out['verts'][0].cpu().detach().numpy())
                    # np.save('EXP_demo/joint.npy', out['joint_img'][0].cpu().detach().numpy())
                    # return
                    # silhouette
                    mask_pred = out.get('verts')
                    if mask_pred is not None:
                        mask_pred = (mask_pred[0] > 0.3).cpu().numpy().astype(np.uint8)
                        mask_pred = cv2.resize(mask_pred, (input.size(3), input.size(2)))
                        try:
                            contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            contours.sort(key=cnt_area, reverse=True)
                            poly = contours[0].transpose(1, 0, 2).astype(np.int32)
                        except:
                            poly = None
                    else:
                        mask_pred = np.zeros([input.size(3), input.size(2)])
                        poly = None
                    # vertex
                    pred = out['verts'][0] if isinstance(out['verts'], list) else out['verts']
                    vertex = (pred[0].cpu() * self.std.cpu()).numpy()
                    uv_pred = out['joint_img']
                    if uv_pred.ndim == 4:
                        uv_point_pred, uv_pred_conf = map2uv(uv_pred.cpu().numpy(), (input.size(2), input.size(3)))
                    else:
                        uv_point_pred, uv_pred_conf = (uv_pred * args.size).cpu().numpy(), [None,]
                    vertex, align_state = registration(vertex, uv_point_pred[0], self.j_regressor, K, args.size, uv_conf=uv_pred_conf[0], poly=poly)

                    vertex2xyz = mano_to_mpii(np.matmul(self.j_regressor, vertex))
                    # np.savetxt(os.path.join(args.out_dir, 'demotext', image_name + '_xyz.txt'), vertex2xyz)
                    # np.savetxt(os.path.join(output_fp, image_name + '_xyz.txt'), vertex2xyz, fmt='%f')
                    np.save(os.path.join(output_fp, image_name + '_xyz.npy'), vertex2xyz)

                    save_a_image_with_mesh_joints(image[..., ::-1], mask_pred, poly, K, vertex, self.face, uv_point_pred[0], vertex2xyz,
                                                os.path.join(output_fp, image_name + '_plot.jpg'))
                    save_mesh(os.path.join(output_fp, image_name + '_mesh.ply'), vertex, self.face)
                    # faces is incorrect

                    bar.suffix = '({batch}/{size})' .format(batch=step+1, size=len(image_files))
                    bar.next()
            bar.finish()
