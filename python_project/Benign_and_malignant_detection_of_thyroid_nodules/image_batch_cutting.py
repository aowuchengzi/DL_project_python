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
   return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

#%%
 #把图像中的病人信息和多余信息裁剪掉，（左，上，右，下）的坐标模式
box_1 = (0,200,1172,1300)

#%%

folder_path = r'总数据'
save_folder_path = r'初步裁剪1'
#%%

for path in getAllImages(folder_path):
    #读图
    file_name = os.path.split(path)
    img = Image.open(path)
    
    #从图中剪裁出图片来，（左，上，右，下）的坐标模式
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
