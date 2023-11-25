import rasterio
import numpy as np
import os
import matplotlib.pyplot as plt
from spacenetutilities.labeltools import coreLabelTools as cLT
import geopandas as gpd
import cv2
from tqdm import tqdm

# BGR_8BIT_THRESHOLD = 1500
# FOLDER_LIST = ['Atlanta_nadir16_catid_1030010002649200', 'Atlanta_nadir8_catid_10300100023BC100', 'Atlanta_nadir10_catid_1030010003CAF100',
#     'Atlanta_nadir7_catid_1030010003D22F00', 'Atlanta_nadir13_catid_1030010002B7D800', 'Atlanta_nadir10_catid_1030010003993E00']
# def convert_to_8bit_bgr(four_channel_im, threshold):
#     three_channel_im = four_channel_im[:, :, 0:3]  # remove 4th channel
#     # next, clip to threshold
#     np.clip(three_channel_im, None, threshold, out=three_channel_im)
#     # finally, rescale to 8-bit range with threshold value scaled to 255
#     three_channel_im = np.floor_divide(three_channel_im,
#                                        threshold/255).astype('uint8')
#     return three_channel_im

# def pan_to_bgr(src_path):

#     im_reader = rasterio.open(os.path.join(src_path))
#     img = np.empty((im_reader.height,
#                     im_reader.width,
#                     im_reader.count))
#     for band in range(im_reader.count):
#         img[:, :, band] = im_reader.read(band+1)
#     bgr_im = convert_to_8bit_bgr(img, BGR_8BIT_THRESHOLD)
#     bgr_im = bgr_im[:, :, ::-1]
#     return bgr_im

# for folder in FOLDER_LIST:
#     PATH_FILES = f'spacenet/{folder}/Pan-Sharpen'

#     files = os.listdir(PATH_FILES)
#     for file in tqdm(files):
#         im = pan_to_bgr(os.path.join(PATH_FILES, file))
#         geo_file = 'target_spacenet/geojson/spacenet-buildings/spacenet-buildings_' + '_'.join(file.split('_')[-2:]).split('.')[0] + '.geojson'
#         dest_img =  os.path.join('spacenet_train_crops', file)
#         dest_msk =  os.path.join('spacenet_train_masks', file)
#         cv2.imwrite(dest_img, im)
#         cLT.createRasterFromGeoJson( geo_file,os.path.join(PATH_FILES, file),
#                                         dest_msk)

os.makedirs('pretrain_crops', exist_ok=True)
os.makedirs('pretrain_masks', exist_ok=True)


images_path = 'spacenet_train_crops/'
masks_path = 'spacenet_train_masks/'

sz = 256 # 768
stp = 64 # 384

for f in tqdm(sorted(os.listdir(images_path))):
    if '.tif' in f:
        img = cv2.imread(images_path + f, cv2.IMREAD_COLOR)[:, :, ::-1]
        msk = cv2.imread(masks_path + f.replace('_image_', '_mask_'), cv2.IMREAD_UNCHANGED)

        img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_NEAREST)
        msk = cv2.resize(msk, (512, 512), interpolation = cv2.INTER_NEAREST)

        msk = 255 * (msk > 0)
        msk = msk.astype('uint8')

        save_dir = 'pretrain_crops/'
        save_dir_masks = 'pretrain_masks/'

        f_name = f.split('.tif')[0]

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