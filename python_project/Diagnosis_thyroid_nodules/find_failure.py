# -*- coding: utf-8 -*-
"""
Created on Thu May  3 09:47:43 2018

@author: 叶晨
"""
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, list_pictures
from keras.models import load_model
import os
import numpy as np
import shutil

#%%
def getAllImages(path):
    #f.endswith（）  限制文件类型
    #f.endswith('.jpg')|f.endswith('.png')  改成这句可以读取jpg/png格式的文件
    #注意 返回的是绝对路径
   return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]
#%%
img_1_path = r'F:\picture\Preprocess_imageset\predict\1\train\1'
img_path_1 = r'F:\picture\Preprocess_imageset\predict\1\fail-train\1'  #最后放分类失败图片的文件夹
cnnf = load_model('cnnf_full_ac=91.47-91.47.h5')
img_1_list = getAllImages(img_1_path)
nb_img_1 = len(img_1_list)


#%% 用于患有结节  -1
for i in range(nb_img_1):
    img_1 = load_img(img_1_list[i], target_size=(200, 200))
    img_1_array = img_to_array(img_1) / 255  #转换为array数组并标准化
    shape = img_1_array.shape
    img_1_array = img_1_array.reshape((1,shape[0],shape[1],shape[2]))
    a = cnnf.predict(img_1_array).reshape((2,))
    if a[1]<0.5:
        shutil.copy(img_1_list[i], img_path_1)
#%% 用于健康甲状腺  -0
for i in range(nb_img_1):
    img_1 = load_img(img_1_list[i], target_size=(200, 200))
    img_1_array = img_to_array(img_1) / 255  #转换为array数组并标准化
    shape = img_1_array.shape
    img_1_array = img_1_array.reshape((1,shape[0],shape[1],shape[2]))
    a = cnnf.predict(img_1_array).reshape((2,))
    if a[1]>0.5:
        shutil.copy(img_1_list[i], img_path_1)