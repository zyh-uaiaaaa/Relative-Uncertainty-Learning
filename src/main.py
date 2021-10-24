# -*- coding: utf-8 -*-

import os
import cv2
import csv
import math
import random
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import torch.utils.data as data
import torch.nn.functional as F

from dataset import RafDataset
from rul import res18feature
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--raf_path', type=str, default='../../raf-basic',help='raf_dataset_path')
parser.add_argument('--pretrained_backbone_path', type=str, default='../../affectnet_baseline/resnet18_msceleb.pth', help='pretrained_backbone_path')
parser.add_argument('--label_path', type=str, default='../../raf-basic/EmoLabel/list_patition_label.txt', help='label_path')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
parser.add_argument('--out_dimension', type=int, default=64, help='feature dimension')
args = parser.parse_args()


def train():
    setup_seed(0)
    res18 = res18feature(args)
    fc = nn.Linear(args.out_dimension, 7)


    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25))
    ])
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = RafDataset(args, phase='train', transform=data_transforms)
    test_dataset = RafDataset(args, phase='test', transform=data_transforms_val)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)

    res18.cuda()
    fc.cuda()

    params = res18.parameters()
    params2 = fc.parameters()


    optimizer = torch.optim.Adam([
        {'params': params},
        {'params': params2, 'lr': 0.002}], lr=0.0002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)



    best_acc = 0
    best_epoch = 0
    for i in range(1, args.epochs + 1):
        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        res18.train()

        for batch_i, (imgs, labels, indexes) in enumerate(train_loader):
            imgs = imgs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            mixed_x, y_a, y_b, att1, att2 = res18(imgs, labels, phase='train')
            outputs = fc(mixed_x)

            criterion = nn.CrossEntropyLoss()
            loss_func = mixup_criterion(y_a, y_b)
            loss = loss_func(criterion, outputs)

            loss.backward()
            optimizer.step()

            iter_cnt += 1
            _, predicts = torch.max(outputs, 1)

            correct_num = torch.eq(predicts, labels).sum()
            correct_sum += correct_num
            running_loss += loss

        scheduler.step()

        running_loss = running_loss / iter_cnt

        acc = correct_sum.float() / float(train_dataset.__len__())
        print('Epoch : %d, train_acc : %.4f, train_loss: %.4f' % (i, acc, running_loss))

        with torch.no_grad():
            res18.eval()

            running_loss = 0.0
            iter_cnt = 0
            correct_sum = 0
            data_num = 0


            for batch_i, (imgs, labels, indexes) in enumerate(test_loader):
                imgs = imgs.cuda()
                labels = labels.cuda()

                outputs = res18(imgs, labels, phase='test')
                outputs = fc(outputs)

                loss = nn.CrossEntropyLoss()(outputs, labels)

                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)

                correct_num = torch.eq(predicts, labels).sum()
                correct_sum += correct_num

                running_loss += loss
                data_num += outputs.size(0)

            running_loss = running_loss / iter_cnt
            test_acc = correct_sum.float() / float(data_num)

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = i
                if best_acc >= 0.888:
                    torch.save({'model_state_dict': res18.state_dict(),
                                'fc_state_dict': fc.state_dict()},
                               "acc_888.pth")
                    print('Model saved.')


            print('Epoch : %d, test_acc : %.4f, test_loss: %.4f' % (i, test_acc, running_loss))

    print('best acc: ', best_acc, 'best epoch: ', best_epoch)


if __name__ == '__main__':
    train()
