# -*- coding: utf-8 -*-

import torch.nn as nn
from resnet import *
from utils import *

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class res18feature(nn.Module):
    def __init__(self, args, pretrained=True, num_classes=7, drop_rate=0.4, out_dim=64):
        super(res18feature, self).__init__()

        #'affectnet_baseline/resnet18_msceleb.pth'
        res18 = ResNet(block=BasicBlock, n_blocks=[2, 2, 2, 2], channels=[64, 128, 256, 512], output_dim=1000)
        msceleb_model = torch.load(args.pretrained_backbone_path)
        state_dict = msceleb_model['state_dict']
        res18.load_state_dict(state_dict, strict=False)

        self.drop_rate = drop_rate
        self.out_dim = out_dim
        self.features = nn.Sequential(*list(res18.children())[:-2])

        self.mu = nn.Sequential(
            nn.BatchNorm2d(512, eps=2e-5, affine=False),
            nn.Dropout(p=self.drop_rate),
            Flatten(),
            nn.Linear(512 * 7 * 7, self.out_dim),
            nn.BatchNorm1d(self.out_dim, eps=2e-5))

        self.log_var = nn.Sequential(
            nn.BatchNorm2d(512, eps=2e-5, affine=False),
            nn.Dropout(p=self.drop_rate),
            Flatten(),
            nn.Linear(512 * 7 * 7, self.out_dim),
            nn.BatchNorm1d(self.out_dim, eps=2e-5))

    def forward(self, x, target, phase='train'):

        if phase == 'train':
            x = self.features(x)
            mu = self.mu(x)
            logvar = self.log_var(x)

            mixed_x, y_a, y_b, att1, att2 = mixup_data(mu, target, logvar.exp().mean(dim=1, keepdim=True), use_cuda=True)
            return mixed_x, y_a, y_b, att1, att2
        else:
            x = self.features(x)
            output = self.mu(x)
            return output
