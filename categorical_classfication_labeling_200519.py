# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:30:38 2019

@author: mit
"""

from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
import cv2 as cv

# 분류 대상 카테고리 선택하기 
base_dir = "E:\\Dataset\\alzheimers-dataset-4-class-of-images\\Alzheimer_s Dataset\\train"
categories = ["MildDemented_t","ModerateDemented_t","NonDemented_t", "VeryMildDemented_t"]
nb_classes = len(categories)

# 이미지 크기 지정 
image_w = 128
image_h = 128
pixels = image_w * image_h

# 이미지 데이터 읽어 들이기 
X = []
Y = []
for idx, cat in enumerate(categories):
    # 레이블 지정 
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    # 이미지 
    image_dir = base_dir + "\\" + cat
    files = glob.glob(image_dir + "\\*.png")
    for i, f in enumerate(files):
        img = Image.open(f).convert('L') 
        
        data = np.asarray(img)      # numpy 배열로 변환
        data = cv.normalize(data.astype(np.float64), None, 0, 1, cv.NORM_MINMAX)
        X.append(data)
        Y.append(label)
        if i % 10 == 0:
            print(i, "\n", data)
X = np.array(X)
Y = np.array(Y)

# 학습 전용 데이터와 테스트 전용 데이터 구분 

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state=42, test_size=0.1, shuffle = True)
xy = (X_train, X_val, Y_train, Y_val)

print('>>> data 저장중 ...')
base_path = 'E:\\Dataset\\alzheimers-dataset-4-class-of-images\\Alzheimer_s Dataset\\train'

dataset_path = os.path.join(base_path, 'dataset')
if not(os.path.exists(dataset_path)):
    os.mkdir(dataset_path)
    
np.save('E:\\Dataset\\alzheimers-dataset-4-class-of-images\\Alzheimer_s Dataset\\train\\dataset\\X_train.npy', X_train)
np.save('E:\\Dataset\\alzheimers-dataset-4-class-of-images\\Alzheimer_s Dataset\\train\\dataset\\Y_train.npy', Y_train)
np.save('E:\\Dataset\\alzheimers-dataset-4-class-of-images\\Alzheimer_s Dataset\\train\\dataset\\X_val.npy', X_val)
np.save('E:\\Dataset\\alzheimers-dataset-4-class-of-images\\Alzheimer_s Dataset\\train\\dataset\\Y_val.npy', Y_val)

print("ok,", len(Y))