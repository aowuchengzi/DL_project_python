# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 14:43:25 2018

@author: 15339
"""
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
#%%
#读取path路径下的 png文件
def getAllImages(path):
    #f.endswith（）  限制文件类型
    #f.endswith('.jpg')|f.endswith('.png')  改成这句可以读取jpg/png格式的文件
    #注意 返回的是绝对路径
   return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]
#%%
folder_path = r'test1\label\PixelLabelData'
save_folder_path = r'test1\label\label_picture'
#%%
for path in getAllImages(folder_path):
    #读图
    file_name = os.path.split(path)
    img = Image.open(path)
    x = np.asarray(img)
    x = x * 255
    img_1 = Image.fromarray(x, 'L')
    save_path = os.path.join(save_folder_path, file_name[1])
    img_1.save(save_path)   #保存到指定文件夹。