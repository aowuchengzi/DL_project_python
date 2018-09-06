# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 10:03:30 2018

@author: 15339
"""

import os
import cv2
import numpy as np
from PIL import Image as image
#%%
folder_path = r'E:\15339\my_keras\opencv-thyroid\image'
save_folder_path = r'E:\15339\my_keras\opencv-thyroid\mask'
#%%
#读取path路径下的 png文件
def getAllImages(path):
    #f.endswith（）  限制文件类型
    #f.endswith('.jpg')|f.endswith('.png')  改成这句可以读取jpg/png格式的文件
    #注意 返回的是绝对路径
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]
#%%
for path in getAllImages(folder_path):
    #读图
    file_name = os.path.split(path)
    save_path = os.path.join(save_folder_path, file_name[1])
    img = cv2.imread(path)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh=cv2.threshold(gray,165,255,cv2.THRESH_BINARY)
    cv2.imwrite(save_path, thresh)   #保存到指定文件夹。








#%%
img=cv2.imread('1.jpg')
cv2.imshow('img',img)
#cv2.waitKey(0)
cv2.destroyAllWindows()
#将图像转化为灰度图像
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
#阈值化处理
ret,thresh=cv2.threshold(gray,160,255,cv2.THRESH_BINARY)
cv2.imshow('thresh',thresh)
cv2.imwrite('1-mask.jpg', thresh)
