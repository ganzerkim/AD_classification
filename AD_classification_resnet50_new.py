# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 09:14:16 2019

@author: MG
"""
import os, glob
import sys
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from keras import utils, models, layers, optimizers
from keras.models import Model, load_model, Sequential

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras import initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import GlobalAveragePooling2D, ZeroPadding2D, Add

from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from keras.applications.resnet50 import ResNet50

import tensorflow as tf
import keras

from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img
from keras.utils import multi_gpu_model

#%%
base_path = 'E:\\Dataset\\alzheimers-dataset-4-class-of-images\\Alzheimer_s Dataset\\'

train_images = np.load(base_path + 'train\\dataset\\X_train.npy')
train_labels = np.load(base_path + 'train\\dataset\\Y_train.npy')
val_images = np.load(base_path + 'train\\dataset\\X_val.npy')
val_labels = np.load(base_path + 'train\\dataset\\Y_val.npy')
test_images = np.load(base_path + 'test\\dataset\\X_test.npy')
test_labels = np.load(base_path + 'test\\dataset\\Y_test.npy')

"""
test_path = os.path.join('C:\\Users\\MG\\Desktop\\H&E New dataset\\H&Edataset\\testimage')
test_img_path = os.path.join(test_path)
test_img = [cv.imread(test_img_path + '\\' + s) for s in os.listdir(test_img_path)]
"""
print(train_images.shape, train_labels.shape)
print(test_images, test_labels.shape)
#%%
class_names = ["MildDemented","ModerateDemented","NonDemented", "VeryMildDemented"]

plt.figure(figsize = (10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i][:,:,0], cmap = plt.cm.bone)
    plt.xlabel(class_names[np.argmax(train_labels[i])])
plt.show()


#%%
batch_size = 16
num_classes = len(class_names)
epochs = 50

#train_images = train_images.astype('float32')
#train_images = train_images / 255

#test_images = test_images.astype('float32')
#test_images = test_images / 255

#train_labels = utils.to_categorical(train_labels, num_classes)
#test_labels = utils.to_categorical(test_labels, num_classes)

#%%

# 모델 구축하기
#binary_crossentropy가 더 높게 나올 가능성????
'''
weight = 'E:\\Dataset\\rsna-intracranial-hemorrhage-detection\\brain_diagnosis\\4-Model\\dataset\\models\\model_weights_resnet.h5'
conv_base = ResNet50(weights = weight, include_top = False, input_shape = (128, 128, 3), pooling = 'max')
conv_base.trainable = True

model = Sequential()
model.add(conv_base)
#model.add(Flatten())
model.add(Dense(1024, activation = 'sigmoid', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(64, activation = 'sigmoid', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.2))
model.add(Dense(4, activation = 'sigmoid'))

model.compile(optimizer = optimizers.Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
model.summary()
#sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
'''

inputs = Input(shape = (128, 128, 3))

def conv1_layer(x):    
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x) 
    return x   
  
def conv2_layer(x):         
    x = MaxPooling2D((3, 3), 2)(x)      
    shortcut = x
 
    for i in range(3):
        if (i == 0):
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)            
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            
            shortcut = x
 
        else:
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])   
            x = Activation('relu')(x)  
 
            shortcut = x        
    
    return x
  
def conv3_layer(x):        
    shortcut = x    
    
    for i in range(4):     
        if(i == 0):            
            x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)    
 
            shortcut = x              
        
        else:
            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])     
            x = Activation('relu')(x)
 
            shortcut = x      
            
    return x
  
def conv4_layer(x):
    shortcut = x        
  
    for i in range(6):     
        if(i == 0):            
            x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)
 
            shortcut = x               
        
        else:
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)
 
            shortcut = x      
 
    return x
  
def conv5_layer(x):
    shortcut = x    
  
    for i in range(3):     
        if(i == 0):            
            x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])  
            x = Activation('relu')(x)      
 
            shortcut = x               
        
        else:
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)           
            
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)       
 
            shortcut = x                  
 
    return x
 
x = conv1_layer(inputs)
x = conv2_layer(x)
x = conv3_layer(x)
x = conv4_layer(x)
x = conv5_layer(x)
 
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation = 'relu', kernel_regularizer=regularizers.l2(0.001))(x)
outputs = Dense(4, activation='sigmoid')(x)
model = Model(inputs, outputs)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy',  metrics=['accuracy'])
from keras.utils import plot_model
plot_model(model, to_file = 'resnet50.png')

#%%
#low_Validation accuratcy => data suffle!!!!!!!!!!!!!
hdf5_file = "E:\\Dataset\\alzheimers-dataset-4-class-of-images\\Alzheimer_s Dataset\\train\\results\\weight_AD_clf_200520_binary.hdf5"

if os.path.exists(hdf5_file):
    # 기존에 학습된 모델 불러들이기
    model.load_weights(hdf5_file)
else:
    # 학습한 모델이 없으면 파일로 저장
    #early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)
    #history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=100, batch_size=32, callbacks=[
    #ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-07)])
    history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=100, batch_size=32, callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)])    
    #history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, batch_size=32, callbacks=[early_stopping])
    model.save_weights(hdf5_file)
        
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    ax[0, 0].set_title('loss')
    ax[0, 0].plot(history.history['loss'], 'r')
    ax[0, 1].set_title('acc')
    ax[0, 1].plot(history.history['acc'], 'b')

    ax[1, 0].set_title('val_loss')
    ax[1, 0].plot(history.history['val_loss'], 'r--')
    ax[1, 1].set_title('val__accuracy')
    ax[1, 1].plot(history.history['val_acc'], 'b--')
    
#%%
# 모델 평가하기 

score = model.evaluate(test_images, test_labels)
print('loss=', score[0])        # loss
print('accuracy=', score[1])    # acc

from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import cv2 as cv

# 분류 대상 카테고리 선택하기 

categories = ["MildDemented","ModerateDemented","NonDemented", "VeryMildDemented"]
nb_classes = len(categories)
class_names =['negative', 'positive']
# 이미지 크기 지정 
image_w = 128
image_h = 128
pixels = image_w * image_h * 3

predictions = model.predict(test_images)
pred =  predictions.astype("float16")

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    #norm_img = cv.normalize(img.astype(np.uint8), None, 0, 255, cv.NORM_MINMAX)
    plt.imshow(img, cmap=plt.cm.bone)
    
    predicted_label = []
    for idx in range(len(predictions_array)):
        predicted_label.append(round(predictions_array[idx]))
    predicted_label = np.array(predicted_label).astype(np.uint8)
    
    if list(predicted_label) == list(true_label):
        color = 'blue'
    else:
        color = 'red'
    pred_bool = predicted_label == 1
    label_bool = true_label == 1    
    categor = np.array(categories)
    
    plt.xlabel("{} {}% ({})".format(str(categor[pred_bool]), str(100 * (predictions_array)), str(categor[label_bool])), color = color, fontsize = 7)
    
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(nb_classes), predictions_array, color = "#777777")
    plt.ylim([0, 1])
    #predicted_label = []
    #for idx in range(len(predictions_array)):
    #    predicted_label.append(round(predictions_array[idx]))
    #predicted_label = np.array(predicted_label).astype(np.uint8)
    
    thisplot[np.argmax(predictions_array)].set_color('green')
    thisplot[np.argmax(true_label)].set_color('orange')
    
num_rows = 3
num_cols = 3
start_point = 300
num_images = num_rows * num_cols
plt.figure(figsize = (2 * 2 * num_cols, 2 * num_rows))

for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i + start_point, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i + start_point, predictions, test_labels)
plt.show()