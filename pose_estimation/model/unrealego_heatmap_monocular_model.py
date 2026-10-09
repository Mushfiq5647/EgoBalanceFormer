from cProfile import run
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


class UnrealEgoHeatmapMonocularModel(BaseModel):
    def name(self):
        return 'UnrealEgo Heatmap Monocular model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.opt = opt
        self.scaler = GradScaler(enabled=opt.use_amp)

        # Only single heatmap loss for monocular setting
        self.loss_names = [
            'heatmap', 
        ]

        # Only single image and heatmap for monocular setting
        self.visual_names = [
            'input_rgb',
            'pred_heatmap',
            'gt_heatmap',
        ]

        self.visual_pose_names = [
        ]
       
        if self.isTrain:
            self.model_names = ['HeatMap']
        else:
            self.model_names = ['HeatMap']

        self.eval_key = "mse_heatmap"
        self.cm2mm = 10

        # define the transform network for monocular input
        print(f"Monocular model: {opt.model}")
        self.net_HeatMap = network.define_HeatMap_Monocular(opt, model=opt.model)

        if self.isTrain:
            # define loss functions
            self.lossfunc_MSE = MSELoss()

            # initialize optimizers
            self.optimizer_HeatMap = torch.optim.Adam(
                params=self.net_HeatMap.parameters(), 
                lr=opt.lr,
                weight_decay=opt.weight_decay
            )

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_HeatMap)
            for optimizer in self.optimizers:
                self.schedulers.append(network.get_scheduler(optimizer, opt))

        # if not self.isTrain or opt.continue_train:
        #     self.load_networks(opt.which_epoch)

    def set_input(self, data):
        self.data = data
        # For monocular setting, we only need one image and one heatmap
        # You can choose either 'input_rgb' or 'input_rgb_left' depending on your data format
        if 'input_rgb' in data:
            self.input_rgb = data['input_rgb'].cuda(self.device)
        elif 'input_rgb_left' in data:
            # Use left image if only stereo data is available
            self.input_rgb = data['input_rgb_left'].cuda(self.device)
        else:
            raise ValueError("No suitable input image found in data. Expected 'input_rgb' or 'input_rgb_left'")
        
        # Similarly for ground truth heatmap
        if 'gt_heatmap' in data:
            self.gt_heatmap = data['gt_heatmap'].cuda(self.device)
        elif 'gt_heatmap_left' in data:
            # Use left heatmap if only stereo data is available
            self.gt_heatmap = data['gt_heatmap_left'].cuda(self.device)
        else:
            raise ValueError("No suitable ground truth heatmap found in data. Expected 'gt_heatmap' or 'gt_heatmap_left'")

    def forward(self):
        with autocast(enabled=self.opt.use_amp):
            # estimate monocular heatmap
            self.pred_heatmap = self.net_HeatMap(self.input_rgb)

    def backward_HeatMap(self):
        with autocast(enabled=self.opt.use_amp):
            loss_heatmap = self.lossfunc_MSE(
                self.pred_heatmap, self.gt_heatmap
            )
            
            self.loss_heatmap = loss_heatmap * self.opt.lambda_heatmap
            
            loss_total = self.loss_heatmap

        self.scaler.scale(loss_total).backward()

    def optimize_parameters(self):

        # set model trainable
        self.net_HeatMap.train()
        
        # set optimizer.zero_grad()
        self.optimizer_HeatMap.zero_grad()

        # forward
        self.forward()

        # backward 
        self.backward_HeatMap()

        # optimizer step
        self.scaler.step(self.optimizer_HeatMap)

        self.scaler.update()

    def evaluate(self, runnning_average_dict):
        # set evaluation mode
        self.net_HeatMap.eval()

        # forward pass
        self.pred_heatmap = self.net_HeatMap(self.input_rgb)
        
        # compute metrics
        for id in range(self.pred_heatmap.size()[0]):  # batch size
            # calculate mse loss for heatmap
            loss_heatmap_id = self.lossfunc_MSE(
                self.pred_heatmap[id], self.gt_heatmap[id]
            )
            
            mse_heatmap = loss_heatmap_id

            # update metrics dict
            runnning_average_dict.update(dict(
                mse_heatmap=mse_heatmap
                )
            )

        return runnning_average_dict