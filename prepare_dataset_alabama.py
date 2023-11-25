import os
import numpy as np
import cv2
from tqdm import tqdm

os.makedirs('pretrain_crops', exist_ok=True)
os.makedirs('pretrain_masks', exist_ok=True)


images_path = 'alabama/image/'
masks_path = 'alabama/mask/'

sz = 256 # 768
stp = 64 # 384

for f in tqdm(sorted(os.listdir(images_path))):
    if '.png' in f:
        img = cv2.imread(images_path + f, cv2.IMREAD_COLOR)
        msk = cv2.imread(masks_path + f.replace('_image_', '_mask_'), cv2.IMREAD_UNCHANGED)

        img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_NEAREST)
        msk = 255 - cv2.resize(msk, (512, 512), interpolation = cv2.INTER_NEAREST)

        msk = 255 * (msk > 0)
        msk = msk.astype('uint8')

        save_dir = 'pretrain_crops/'
        save_dir_masks = 'pretrain_masks/'

        f_name = f.split('.png')[0]

        y0 = 0
        while y0 + sz <= img.shape[0]:
            x0 = 0
            while x0 + sz <= img.shape[1]:
                cv2.imwrite(save_dir + f'{f_name}_{x0}_{y0}.png', img[y0:y0+sz, x0:x0+sz, :], [cv2.IMWRITE_PNG_COMPRESSION, 5])
                cv2.imwrite(save_dir_masks + f'{f_name}_{x0}_{y0}.png', msk[y0:y0+sz, x0:x0+sz], [cv2.IMWRITE_PNG_COMPRESSION, 5])

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