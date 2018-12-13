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
def getAllImages(path, file_type = '.png'):
    #f.endswith（）  限制文件类型
    #注意 返回的是绝对路径
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(file_type)]
#%%
folder_path = r'图像标签\PixelLabelData'
save_folder_path = r'图像标签\label'
#%%
def visual_mask(folder_path, save_folder_path, file_type = '.png'):
    '''
    把matlab生成的0-1掩模可视化
    folder_path:目标文件夹
    save_folder_path:处理后的保存文件夹
    file_type：目标文件夹中图片类型
    '''
    for path in getAllImages(folder_path, file_type):
        #读图
        file_name = os.path.split(path)
        img = Image.open(path)
        x = np.asarray(img)
        x = x * 255
        img_1 = Image.fromarray(x, 'L')
        save_path = os.path.join(save_folder_path, file_name[1])
        img_1.save(save_path)   #保存到指定文件夹。