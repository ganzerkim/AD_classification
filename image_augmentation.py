# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:54:03 2020

@author: MIT-DGMIF
"""
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2 as cv
import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import numpy as np
import matplotlib.pyplot as plt


datagen = ImageDataGenerator(rotation_range=0, width_shift_range=0.2, height_shift_range=0.2, rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=False, fill_mode = 'nearest')

img = load_img('C:\\Users\\MIT-DGMIF\\Desktop\\exam\\data\\1.jpg')  # PIL 이미지
x = img_to_array(img)  # (3, 150, 150) 크기의 NumPy 배열
  # (1, 3, 150, 150) 크기의 NumPy 배열

x = cv.resize(x, dsize = (128, 128), interpolation = cv.INTER_LINEAR)
x = x.reshape((1,) + x.shape)
# 아래 .flow() 함수는 임의 변환된 이미지를 배치 단위로 생성해서
# 지정된 `preview/` 폴더에 저장합니다.
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='C:\\Users\\MIT-DGMIF\\Desktop\\exam\\gen', save_prefix='AD', save_format='jpg'):
    i += 1
    if i > 20:
        break  # 이미지 20장을 생성하고 마칩니다
    
    
imgdir = "E:\\Dataset\\alzheimers-dataset-4-class-of-images\\Alzheimer_s Dataset\\train\\ModerateDemented"
# if you want file of a specific extension (.png):
filelist = [f for f in glob.glob(imgdir + "**/*.jpg", recursive=True)]
savedir = "E:\\Dataset\\alzheimers-dataset-4-class-of-images\\Alzheimer_s Dataset\\train\\ModerateDemented_AUG"

datagen = ImageDataGenerator(rotation_range=0, width_shift_range=0.2, height_shift_range=0.2, rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=False, fill_mode = 'nearest')
frame_num = 1
for file in filelist:
    img = load_img(file)
    x = img_to_array(img)
    x = cv.resize(x, dsize = (128, 128), interpolation = cv.INTER_LINEAR)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=savedir, save_prefix='AUG', save_format='png'):
        i += 1
        if i > 30:
            break  # 이미지 20장을 생성하고 마칩니다
    print(frame_num)
    frame_num += 1
    
    
    
imgdir = "E:\\Dataset\\alzheimers-dataset-4-class-of-images\\Alzheimer_s Dataset\\train\\VeryMildDemented"
# if you want file of a specific extension (.png):
filelist = [f for f in glob.glob(imgdir + "**/*.jpg", recursive=True)]
savedir = "E:\\Dataset\\alzheimers-dataset-4-class-of-images\\Alzheimer_s Dataset\\train\\VeryMildDemented(128)"

#start_pos = start_x, start_y = (0, 0)
#cropped_image_size = w, h = (100, 100)
frame_num = 1
img_size = 128
for file in filelist:
    img = Image.open(file)
    resize_image = img.resize((img_size, img_size))
    name = os.path.basename(file)
    name = os.path.splitext(name)[0]
    save_to= os.path.join(savedir, name+"_resize(" + str(int(img_size)) +").png")
    resize_image.save(save_to.format(frame_num), "png", quality=100 )
    print(frame_num)
    frame_num += 1
