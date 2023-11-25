import os

from os import path, makedirs, listdir
import sys
import numpy as np

from tqdm import tqdm
import timeit
import cv2


from utils import *

out_dir = 'test_out'
masks_dir = 'train_updated_titiles/masks'

if __name__ == '__main__':
    t0 = timeit.default_timer()
    
    dices = []
    ious = []

    for f in sorted(listdir(out_dir)):
        pred = cv2.imread(path.join(out_dir, f), cv2.IMREAD_UNCHANGED)
        truth = cv2.imread(path.join(masks_dir, f.replace('image', 'mask')), cv2.IMREAD_UNCHANGED)
        
        _dice = dice(truth > 0, pred > 0)
        _iou = iou(truth > 0, pred > 0)

        dices.append(_dice)
        ious.append(_iou)

        print(f'{f} f1/dice: {_dice} iou: {_iou}')

    print(f'ALL: f1/dice: {np.mean(dices)} iou: {np.mean(ious)}')
        

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))