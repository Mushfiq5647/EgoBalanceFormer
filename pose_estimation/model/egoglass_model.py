from enum import auto
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
from torch.nn import MSELoss

import itertools
from .base_model import BaseModel
from . import network
from utils.loss import LossFuncLimb, LossFuncCosSim, LossFuncMPJPE
from utils.util import batch_compute_similarity_transform_torch
from utils.body_part import heatmap_to_pseudo_limb_mask


class EgoGlassModel(BaseModel):
    def name(self):
        return 'EgoGlass model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.opt = opt
        self.scaler = GradScaler(enabled=opt.use_amp)

        self.loss_names = [
            'heatmap_left', 'heatmap_right',
            'body_part_left', 'body_part_right',
            'heatmap_left_rec', 'heatmap_right_rec',
            'pose', 'cos_sim',
        ]

        if self.isTrain:
            self.visual_names = [
                'input_rgb_left', 'input_rgb_right',
                'pred_heatmap_left', 'pred_heatmap_right',
                'gt_heatmap_left', 'gt_heatmap_right',
                'pred_heatmap_left_rec', 'pred_heatmap_right_rec'
            ]
        else:
            self.visual_names = [
                # 'input_rgb_left', 'input_rgb_right',
                'pred_heatmap_left', 'pred_heatmap_right',
                'gt_heatmap_left', 'gt_heatmap_right',
            ]

        self.visual_pose_names = [
            "pred_pose", "gt_pose"
        ]
       
        if self.isTrain:
            self.model_names = ['HeatMap_left', 'HeatMap_right', 'AutoEncoder']
        else:
            self.model_names = ['HeatMap_left', 'HeatMap_right', 'AutoEncoder']

        self.eval_key = "mpjpe"
        self.cm2mm = 10

        # define the transform network
        self.net_HeatMap_left = network.define_HeatMap(opt, model=opt.model)
        self.net_HeatMap_right = network.define_HeatMap(opt, model=opt.model)
        self.net_AutoEncoder = network.define_AutoEncoder(opt, model=opt.model)

        # define loss functions
        self.lossfunc_MSE = MSELoss()
        self.lossfunc_limb = LossFuncLimb()
        self.lossfunc_cos_sim = LossFuncCosSim()
        self.lossfunc_MPJPE = LossFuncMPJPE()

        if self.isTrain:
            # initialize optimizers
            self.optimizer_HeatMap_left = torch.optim.Adam(
                params=self.net_HeatMap_left.parameters(), 
                lr=opt.lr,
                weight_decay=opt.weight_decay
            )

            self.optimizer_HeatMap_right = torch.optim.Adam(
                params=self.net_HeatMap_right.parameters(), 
                lr=opt.lr,
                weight_decay=opt.weight_decay
            )

            self.optimizer_AutoEncoder = torch.optim.Adam(
                params=self.net_AutoEncoder.parameters(), 
                lr=opt.lr,
                weight_decay=opt.weight_decay
            )

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_HeatMap_left)
            self.optimizers.append(self.optimizer_HeatMap_right)
            self.optimizers.append(self.optimizer_AutoEncoder)
            for optimizer in self.optimizers:
                self.schedulers.append(network.get_scheduler(optimizer, opt))

        # if not self.isTrain or opt.continue_train:
        #     self.load_networks(opt.which_epoch)

    def set_input(self, data):
        self.data = data
        self.input_rgb_left = data['input_rgb_left'].cuda(self.device)
        self.input_rgb_right = data['input_rgb_right'].cuda(self.device)
        self.gt_heatmap_left = data['gt_heatmap_left'].cuda(self.device)
        self.gt_heatmap_right = data['gt_heatmap_right'].cuda(self.device)
        self.gt_pose = data['gt_local_pose'].cuda(self.device)

    def forward(self):
        with autocast(enabled=self.opt.use_amp):
            out_left = self.net_HeatMap_left(self.input_rgb_left)
            out_right = self.net_HeatMap_right(self.input_rgb_right)
            self.pred_heatmap_left = out_left[0] if isinstance(out_left, tuple) else out_left
            self.pred_heatmap_right = out_right[0] if isinstance(out_right, tuple) else out_right
            self.pred_body_part_left = out_left[1] if isinstance(out_left, tuple) and len(out_left) > 1 else None
            self.pred_body_part_right = out_right[1] if isinstance(out_right, tuple) and len(out_right) > 1 else None

            pred_heatmap_cat = torch.cat([self.pred_heatmap_left, self.pred_heatmap_right], dim=1)

            # EgoGlass paper: fuse body part masks with heatmaps before 3D module
            if self.pred_body_part_left is not None and self.pred_body_part_right is not None:
                pred_body_part_cat = torch.cat([self.pred_body_part_left, self.pred_body_part_right], dim=1)
                ae_input = torch.cat([pred_heatmap_cat, pred_body_part_cat], dim=1)
            else:
                ae_input = pred_heatmap_cat

            self.pred_pose, pred_heatmap_rec_cat = self.net_AutoEncoder(ae_input)

            self.pred_heatmap_left_rec, self.pred_heatmap_right_rec = torch.chunk(pred_heatmap_rec_cat, 2, dim=1)
    
    def backward_HeatMap(self):
        with autocast(enabled=self.opt.use_amp):
            loss_heatmap_left = self.lossfunc_MSE(
                self.pred_heatmap_left, self.gt_heatmap_left
            )
            loss_heatmap_right = self.lossfunc_MSE(
                self.pred_heatmap_right, self.gt_heatmap_right
            )
            self.loss_heatmap_left = loss_heatmap_left * self.opt.lambda_heatmap
            self.loss_heatmap_right = loss_heatmap_right * self.opt.lambda_heatmap
            loss_total = self.loss_heatmap_left + self.loss_heatmap_right

            if self.pred_body_part_left is not None and self.pred_body_part_right is not None:
                lambda_bp = getattr(self.opt, 'lambda_body_part', 0.1)
                # Paper Eq.2: Lb = mse(predicted_mask, mask). Pseudo-limb mask from GT heatmaps.
                gt_limb_left = heatmap_to_pseudo_limb_mask(self.gt_heatmap_left)
                gt_limb_right = heatmap_to_pseudo_limb_mask(self.gt_heatmap_right)
                loss_bp_left = self.lossfunc_MSE(self.pred_body_part_left, gt_limb_left)
                loss_bp_right = self.lossfunc_MSE(self.pred_body_part_right, gt_limb_right)
                self.loss_body_part_left = loss_bp_left * lambda_bp
                self.loss_body_part_right = loss_bp_right * lambda_bp
                loss_total = loss_total + self.loss_body_part_left + self.loss_body_part_right
            else:
                self.loss_body_part_left = torch.tensor(0.0, device=self.gt_heatmap_left.device)
                self.loss_body_part_right = torch.tensor(0.0, device=self.gt_heatmap_right.device)

        self.scaler.scale(loss_total).backward(retain_graph=True)

    def backward_AutoEncoder(self):
        with autocast(enabled=self.opt.use_amp):
            loss_pose = self.lossfunc_MPJPE(self.pred_pose, self.gt_pose)
            loss_cos_sim = self.lossfunc_cos_sim(self.pred_pose, self.gt_pose)
            loss_heatmap_left_rec = self.lossfunc_MSE(
                self.pred_heatmap_left_rec, self.pred_heatmap_left.detach()
            )
            loss_heatmap_right_rec = self.lossfunc_MSE(
                self.pred_heatmap_right_rec, self.pred_heatmap_right.detach()
            )

            self.loss_pose = loss_pose * self.opt.lambda_mpjpe
            self.loss_cos_sim = loss_cos_sim * self.opt.lambda_cos_sim * self.opt.lambda_mpjpe
            self.loss_heatmap_left_rec = loss_heatmap_left_rec * self.opt.lambda_heatmap_rec
            self.loss_heatmap_right_rec = loss_heatmap_right_rec * self.opt.lambda_heatmap_rec

            loss_total = self.loss_pose + self.loss_cos_sim + \
                self.loss_heatmap_left_rec + self.loss_heatmap_right_rec

        self.scaler.scale(loss_total).backward()

    def optimize_parameters(self):

        # set model trainable
        self.net_HeatMap_left.train()
        self.net_HeatMap_right.train()
        self.net_AutoEncoder.train()
        
        # set optimizer.zero_grad()
        self.optimizer_HeatMap_left.zero_grad()
        self.optimizer_HeatMap_right.zero_grad()
        self.optimizer_AutoEncoder.zero_grad()

        # forward
        self.forward()

        # backward 
        self.backward_HeatMap()
        self.backward_AutoEncoder()

        # optimizer step
        self.scaler.step(self.optimizer_HeatMap_left)
        self.scaler.step(self.optimizer_HeatMap_right)
        self.scaler.step(self.optimizer_AutoEncoder)

        self.scaler.update()

    def evaluate(self, runnning_average_dict):
        # set evaluation mode
        self.net_HeatMap_left.eval()
        self.net_HeatMap_right.eval()
        self.net_AutoEncoder.eval()

        # forward pass
        out_left = self.net_HeatMap_left(self.input_rgb_left)
        out_right = self.net_HeatMap_right(self.input_rgb_right)
        self.pred_heatmap_left = out_left[0] if isinstance(out_left, tuple) else out_left
        self.pred_heatmap_right = out_right[0] if isinstance(out_right, tuple) else out_right
        self.pred_body_part_left = out_left[1] if isinstance(out_left, tuple) and len(out_left) > 1 else None
        self.pred_body_part_right = out_right[1] if isinstance(out_right, tuple) and len(out_right) > 1 else None
        pred_heatmap_cat = torch.cat([self.pred_heatmap_left, self.pred_heatmap_right], dim=1)
        if self.pred_body_part_left is not None and self.pred_body_part_right is not None:
            pred_body_part_cat = torch.cat([self.pred_body_part_left, self.pred_body_part_right], dim=1)
            ae_input = torch.cat([pred_heatmap_cat, pred_body_part_cat], dim=1)
        else:
            ae_input = pred_heatmap_cat
        self.pred_pose = self.net_AutoEncoder.predict_pose(ae_input)

        S1_hat = batch_compute_similarity_transform_torch(self.pred_pose, self.gt_pose)

        # compute metrics
        for id in range(self.pred_pose.size()[0]):  # batch size
            # calculate mpjpe and p_mpjpe   # cm to mm
            mpjpe = self.lossfunc_MPJPE(self.pred_pose[id], self.gt_pose[id]) * self.cm2mm
            pa_mpjpe = self.lossfunc_MPJPE(S1_hat[id], self.gt_pose[id]) * self.cm2mm

            # update metrics dict
            runnning_average_dict.update(dict(
                mpjpe=mpjpe, 
                pa_mpjpe=pa_mpjpe)
            )

        return runnning_average_dict