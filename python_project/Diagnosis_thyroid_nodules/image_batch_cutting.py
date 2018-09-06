# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:08:34 2018

@author: 叶晨
"""
#%%
from PIL import Image
import matplotlib.pyplot as plt
import os
#%%
#读取path路径下的 png文件
def getAllImages(path):
    #f.endswith（）  限制文件类型
    #f.endswith('.jpg')|f.endswith('.png')  改成这句可以读取jpg/png格式的文件
    #注意 返回的是绝对路径
   return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]

#%%
 #从图中剪裁出200*200的图片来，（左，上，右，下）的坐标模式
box_1 = (80,70,430,370)#偏大情况
box_2 = (90,100,330,340)#偏左情况

box_3 = (150,180,350,380)#偏下
box_0 = (140,110,360,330)#一般情况
box_00 = (140,110,360,330)
#ab = getAllImages(folder_path)
#sf = os.path.split(ab[0])
#%%

folder_path = r'F:\picture\待处理-b\1'
save_folder_path = r'F:\picture\cut_b\1'
#%%

for path in getAllImages(folder_path):
    #读图
    file_name = os.path.split(path)
    img = Image.open(path)
    
    #从图中剪裁出200*200的图片来，（左，上，右，下）的坐标模式
    roi=img.crop(box_1)
    save_path = os.path.join(save_folder_path, file_name[1])
    roi.save(save_path)   #保存到指定文件夹。
    






#%%
img=Image.open(r'F:\picture\cut_test\a1.png')  #打开图像
#plt.figure("beauty")
#plt.subplot(1,2,1), plt.title('origin')
#plt.imshow(img),plt.axis('off')

box=(150,120,350,320)
roi=img.crop(box)
roi.save(r'F:\picture\cut_test\a1-1.png')
#plt.subplot(1,2,2), plt.title('roi')
#plt.imshow(roi),plt.axis('off')
#plt.show()
#%%
import os
#读取path路径下的 png文件
def getAllImages(path):
    #f.endswith（）  限制文件类型
    #f.endswith('.jpg')|f.endswith('.png')  改成这句可以读取jpg/png格式的文件
    #注意 返回的是绝对路径
   return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]
#%%

import pylab as plb
import PIL.Image as Image
#循环读图
for path in getAllImages(r'F:\picture\cut_test'):
    #读图
    img = Image.open(path)
    #显示
#    plb.imshow(img)
    #设置裁剪点（4个）
    corner = plb.ginput(4)
    #顺时针取点求解
    left = (corner[0][0] + corner[3][0])/2
    top = (corner[1][1] + corner[0][1])/2
    reight = (corner[1][0] + corner[2][0])/2
    under = (corner[3][1] + corner[2][1])/2
    print(left,top,reight,under)
    #box = [left,top,reight,under]
    #box中的数必须是 int 否则会报错
    box = [int(left),int(top),int(reight),int(under)]
    #裁剪
    img2 = img.crop(box)
    #显示裁剪后的效果
    #plb.imshow(img2)
    #plb.show()
    #储存为原来路径覆盖原文件
    img2.save(path)
#plb.show()