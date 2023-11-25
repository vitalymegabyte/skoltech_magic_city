import os

from os import path, makedirs, listdir
import sys
import numpy as np

import torch
from torch import nn

from tqdm import tqdm
import timeit
import cv2

from torch.nn import DataParallel

from models import Timm_Unet

from utils import *

input_dir = 'test_dataset_test/images' # 'test'
out_dir = 'sub03' # 'test_out'

input_dir, out_dir = sys.argv[1:]

models_folder = '.' # 'weights'

msk_value = 1 # 255 # 1 
threshold = 0.5 

if __name__ == '__main__':
    t0 = timeit.default_timer()
    
    makedirs(out_dir, exist_ok=True)

    model = Timm_Unet(name='convnextv2_base.fcmae_ft_in22k_in1k', pretrained=None)
    snap_to_load = 'convnextv2_base_wPre_256_e04_full_0_last_5' # 'convnextv2_base_256_e04_full_0_best' # 'convnextv2_base_256_e04_0_best'
    print("=> loading checkpoint '{}'".format(snap_to_load))
    checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
    loaded_dict = checkpoint['state_dict']
    sd = model.state_dict()
    for k in model.state_dict():
        if k in loaded_dict:
            sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict)
    print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
        checkpoint['epoch'], checkpoint['best_score']))
    model = model.eval().cuda()
    model = DataParallel(model).eval().cuda()


    with torch.no_grad():
        for f in tqdm(sorted(listdir(input_dir))):
            img = cv2.imread(path.join(input_dir, f), cv2.IMREAD_COLOR)
            img = preprocess_inputs(img)

            msk = np.zeros((img.shape[:2]), dtype='float')
            cnts = np.zeros((img.shape[:2]), dtype='uint')       

            sz = 256
            stp = 128 # 64

            batch_size = 32

            crops = []
            coords = []

            y0 = 0
            while y0 + sz <= img.shape[0]:
                x0 = 0
                while x0 + sz <= img.shape[1]:

                    crops.append(img[y0:y0+sz, x0:x0+sz, :])
                    coords.append((y0, x0))

                    if x0 + sz == img.shape[1]:
                        break
                    if x0 + stp + sz <= img.shape[1]:
                        x0 += stp
                    else:
                        x0 = img.shape[1] - sz

                if y0 + sz == img.shape[0]:
                    break
                if y0 + stp + sz <= img.shape[0]:
                    y0 += stp
                else:
                    y0 = img.shape[0] - sz


            with torch.cuda.amp.autocast():
                i = 0
                while i < len(crops):
                    imgs = np.asarray(crops[i:min(i+batch_size, len(crops))])
                    batch_coords = coords[i:min(i+batch_size, len(crops))]

                    imgs = imgs.transpose((0, 3, 1, 2)).copy()

                    for _tta in range(8):
                        _i = _tta // 2
                        _flip = False
                        if _tta % 2 == 1:
                            _flip = True

                        if _i == 0:
                            inp = imgs.copy()
                        elif _i == 1:
                            inp = np.rot90(imgs, k=1, axes=(2,3)).copy()
                        elif _i == 2:
                            inp = np.rot90(imgs, k=2, axes=(2,3)).copy()
                        elif _i == 3:
                            inp = np.rot90(imgs, k=3, axes=(2,3)).copy()

                        if _flip:
                            inp = inp[:, :, :, ::-1].copy()

                        inp = torch.from_numpy(inp).float().cuda()                   
                    # with torch.no_grad():                      
                        out = model(inp)
                        msk_pred = torch.sigmoid(out).cpu().numpy()
                        
                        if _flip:
                            msk_pred = msk_pred[:, :, :, ::-1].copy()

                        if _i == 1:
                            msk_pred = np.rot90(msk_pred, k=4-1, axes=(2,3)).copy()
                        elif _i == 2:
                            msk_pred = np.rot90(msk_pred, k=4-2, axes=(2,3)).copy()
                        elif _i == 3:
                            msk_pred = np.rot90(msk_pred, k=4-3, axes=(2,3)).copy()

                        for j in range(len(batch_coords)):
                            y0, x0 = batch_coords[j]
                            msk[y0:y0+sz, x0:x0+sz] += msk_pred[j, 0]
                            cnts[y0:y0+sz, x0:x0+sz] += 1

                    i += batch_size

            # break
            msk = msk / cnts

            msk = ((msk > threshold) * msk_value).astype('uint8')

            cv2.imwrite(path.join(out_dir , f.replace('image', 'mask')), msk, [cv2.IMWRITE_PNG_COMPRESSION, 4])
            
        

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))