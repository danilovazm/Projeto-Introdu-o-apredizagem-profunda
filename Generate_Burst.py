import os
from PIL import Image
import numpy as np
import h5py
from matplotlib import pyplot as plt
from pytvision.transforms import functional as F
import cv2
from PIL import Image
import imageio
import glob

https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv

def noisy(image):
      row,col= image.shape
      mean = 0
      var = 5
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col))
      gauss = gauss.reshape(row,col)
      noisy = image + gauss
      return noisy

files = os.listdir('../Zurich/train/huawei_raw_noisy')
count = 0
x = 60
y = 60
h = 64
w = 64
for file in files:
    img = Image.open('../Zurich/train/huawei_raw_noisy/' + file )
    Im = np.asarray(img)
    os.mkdir('../Zurich/train/huawei_raw_noisy_burst/' + file[:-4])
    for i in range(8):
        mat_r, mat_t, mat_w= F.get_geometric_random_transform(Im.shape, degree=0.5, translation=0.005, warp=0.01)
        img_translate = F.applay_geometrical_transform(Im, mat_r, mat_t, mat_w, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        crop_img_translate = img_translate[y:y+h, x:x+w]
        im = Image.fromarray(np.squeeze(crop_img_translate, axis=2))
        im.save('../Zurich/train/huawei_raw_noisy_burst/' + file[:-4] + '/' + str(i) + '.png')
    gt = imageio.imread('../Zurich/train/canon/' + file[:-4] + '.jpg', pilmode='RGB')
    crop_gt = gt[y:y+h, x:x+w, :]
    imageio.imwrite('../Zurich/train/huawei_raw_noisy_burst/' + file[:-4] + '/gt' + '.jpg', crop_gt)
