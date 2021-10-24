# -*- coding: utf-8 -*-
import os
import torch.utils.data as data
import pandas as pd
import random

from utils import *

class RafDataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.raf_path = args.raf_path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        df = pd.read_csv(args.label_path, sep=' ', header=None)

        name_c = 0
        label_c = 1
        if phase == 'train':
            dataset = df[df[name_c].str.startswith('train')]
        else:
            dataset = df[df[name_c].str.startswith('test')]

        # notice the raf-db label starts from 1 while label of other dataset starts from 0
        self.label = dataset.iloc[:, label_c].values - 1
        images_names = dataset.iloc[:, name_c].values
        self.aug_func = [filp_image, add_g]
        self.file_paths = []

        for f in images_names:
            f = f.split(".")[0]
            f += '_aligned.jpg'
            file_name = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(file_name)
            

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx])
        image = image[:, :, ::-1]

        if self.phase == 'train':

            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx