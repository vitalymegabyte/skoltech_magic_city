import random
from glob import glob
from os import listdir, path, makedirs
from secrets import choice

import cv2
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import torch
from imgaug import augmenters as iaa
from torch.utils.data import Dataset

from utils import *


# train_files = sorted(listdir('train_crops'))
# val_files = sorted(listdir('val_crops'))

# class O(Dataset):
#     def __init__(self, files, data_dir='train_crops', masks_dir='train_masks',  aug=True, epoch_size=-1):
#         super().__init__()
#         self.files = files
#         self.data_dir = data_dir
#         self.aug = aug
#         self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)
#         self.masks_dir = masks_dir
#         self.epoch_size = len(self.files)
#         if epoch_size > 0:
#             self.epoch_size = epoch_size

#     def __len__(self):
#         return self.epoch_size

# self = O(train_files)

# idx = 0



class TrainDataset(Dataset):
    def __init__(self, files, data_dir='train_crops', masks_dir='train_masks',  aug=True, epoch_size=-1):
        super().__init__()
        self.files = files
        self.data_dir = data_dir
        self.aug = aug
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)
        self.masks_dir = masks_dir
        self.epoch_size = len(self.files)
        if epoch_size > 0:
            self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size


    def __getitem__(self, idx):
        _idx = idx % len(self.files)

        f_name = self.files[_idx]

        img = cv2.imread(path.join(self.data_dir, f_name), cv2.IMREAD_COLOR)
        msk = cv2.imread(path.join(self.masks_dir, f_name), cv2.IMREAD_UNCHANGED)

        
        if self.aug:
            _p = 0.5
            if random.random() > _p:
                img = img[::-1, ...]
                msk = msk[::-1, ...]

            _p = 0.5
            if random.random() > _p:
                img = img[:, ::-1, :]
                msk = msk[:, ::-1]

            _p = 0.0
            if random.random() > _p:
                _k = random.randrange(4)
                img = np.rot90(img, k=_k, axes=(0,1))
                msk = np.rot90(msk, k=_k, axes=(0,1))

            _p = 0.01
            if random.random() > _p:
                _d = int(img.shape[1] * 0.25)
                rot_pnt =  (img.shape[1] // 2 + random.randint(-_d, _d), img.shape[2] // 2 + random.randint(-_d, _d))
                scale = 1
                if random.random() > 0.01:
                    scale = random.normalvariate(1.0, 0.3)

                angle = 0
                if random.random() > 0.01:
                    angle = random.randint(0, 90) - 45
                if (angle != 0) or (scale != 1):
                    _v = random.choice([0, 255])
                    img = rotate_image(img, angle, scale, rot_pnt, borderValue=(_v, _v, _v))
                    msk = rotate_image(msk, angle, scale, rot_pnt, borderValue=0)

                _p = 0.5
                if random.random() > _p:
                    img = shift_channels(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

                _p = 0.5
                if random.random() > _p:
                    img = change_hsv(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

                _p = 0.92
                if random.random() > _p:
                    img = img[:, :, ::-1]

                _p = 0.5
                if random.random() > _p:
                    if random.random() > 0.66:
                        img = clahe(img)
                    elif random.random() > 0.5:
                        img = gauss_noise(img)
                    elif random.random() > 0:
                        img = cv2.blur(img, (3, 3))
                
                _p = 0.5
                if random.random() > _p:
                    if random.random() > 0.66:
                        img = saturation(img, 0.8 + random.random() * 0.4)
                    elif random.random() > 0.5:
                        img = brightness(img, 0.8 + random.random() * 0.4)
                    elif random.random() > 0:
                        img = contrast(img, 0.8 + random.random() * 0.4)

            _p = 0.9
            if random.random() > _p:
                el_det = self.elastic.to_deterministic()
                for z in range(img.shape[0]):
                    img[z] = el_det.augment_image(img[z])
            

            _p = 0.8
            if random.random() > _p:
                for _i in range(random.randrange(3)):
                    _v = random.choice([0, 255])
                    sz0 = random.randrange(1, int(img.shape[0] * 0.3))
                    sz1 = random.randrange(1, int(img.shape[1] * 0.3))
                    x0 = random.randrange(img.shape[1] - sz1)
                    y0 = random.randrange(img.shape[0] - sz0)
                    img[y0:y0+sz0, x0:x0+sz1, :] = _v
                    msk[y0:y0+sz0, x0:x0+sz1] = 0


        msk = msk[..., np.newaxis] > 127

        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()
        
        msk = torch.from_numpy(msk.transpose((2, 0, 1)).copy()).long()

        sample = {'img': img, 'msk': msk, 'id': f_name}

        return sample
