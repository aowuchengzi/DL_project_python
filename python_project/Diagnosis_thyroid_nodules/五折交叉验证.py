# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 21:28:54 2018

@author: 叶晨
"""

#%%
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
img_a_path = r'F:\picture\cut_a\all'
img_b_path = r'F:\picture\cut_b\all'
img_patha = r'F:\picture\cut_a'
img_pathb = r'F:\picture\cut_b'

img_a_list = getAllImages(img_a_path)
img_b_list = getAllImages(img_b_path)

np.random.shuffle(img_a_list)
np.random.shuffle(img_b_list)

print(len(img_a_list))
print(len(img_b_list))
#%%
num_a = int(len(img_a_list)/5)
num_b = int(len(img_b_list)/5)
#list_img = img_a_list[1*num_a:1*num_a + num_a]

#shutil.copy(img_a_list[0], r'F:\picture\cut_a\1')
#%%
def getImglist5(img_list):
    num_a = int(len(img_list)/5)
    list_img = []
    for i in range(5):
        if i<4:
            list_img.append(img_list[i*num_a:i*num_a + num_a])
        else:
            list_img.append(img_list[i*num_a:])
    return list_img
#%%
aaa = getImglist5(img_a_list)
bbb = getImglist5(img_b_list)
#%%
def copyTo5folder(img_list_5, img_path):    
    for i in range(5):
        for a in img_list_5[i]:
            shutil.copy(a, os.path.join(img_path, str(i)))
    return
#%%
copyTo5folder(aaa, img_patha)
copyTo5folder(bbb, img_pathb)
        
        
        
        
        