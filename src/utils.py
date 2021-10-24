import torch
import cv2
import numpy as np
import random

#add gaussian noise
def add_g(image_array, mean=0.0, var=30):
    std = var ** 0.5
    image_add = image_array + np.random.normal(mean, std, image_array.shape)
    image_add = np.clip(image_add, 0, 255).astype(np.uint8)
    return image_add

#flip image
def filp_image(image_array):
    return cv2.flip(image_array, 1)

#set random seed to ensure the results can be reproduced,
#we simply set the random seed to 0, change the random seed value might get the performance of RUL better,
#but we believe that the random seed parameter should not be finetuned
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

#use uncertainty value as weights to mixup feature
#we find that simply follow the traditional mixup setup
# to get mixup pairs can ensure good performance
def mixup_data(x, y, att, use_cuda=True):
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    att1 = att / (att + att[index])
    att2 = att[index] / (att + att[index])
    mixed_x = att1 * x + att2 * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, att1, att2

#add-up loss
def mixup_criterion(y_a, y_b):
    return lambda criterion, pred:  0.5 *  criterion(pred, y_a) + 0.5 * criterion(pred, y_b)