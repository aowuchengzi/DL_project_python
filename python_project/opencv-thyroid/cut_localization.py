# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 18:55:40 2018

@author: 叶晨
"""
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model, load_model
from PIL import Image
import matplotlib.pyplot as plt
import os
#%%
folder_path = r'E:\15339\my_keras\opencv-thyroid\mask'
folder_path_2 = r'E:\15339\my_keras\opencv-thyroid\image'
save_folder_path = r'E:\15339\my_keras\opencv-thyroid\mask_cutted'
save_folder_path_2 = r'E:\15339\my_keras\opencv-thyroid\image_cutted'
model1 = load_model('model1_mse=0.00008.h5')
#%%
#读取path路径下的 png文件
def getAllImages(path):
    #f.endswith（）  限制文件类型
    #f.endswith('.jpg')|f.endswith('.png')  改成这句可以读取jpg/png格式的文件
    #注意 返回的是绝对路径
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.png')]

def getAllImages_double(path_1, path_2):
    #f.endswith（）  限制文件类型
    #f.endswith('.jpg')|f.endswith('.png')  改成这句可以读取jpg/png格式的文件
    #注意 返回的是绝对路径
    return [os.path.join(path_1,f) for f in os.listdir(path_1) if f.endswith('.png')], [os.path.join(path_2,f) for f in os.listdir(path_2) if f.endswith('.png')] 
            

def getbox(img_path):
    #利用训练好的模型
    #获取一张图片的裁剪bndbox
    #img_path为绝对路径
    img = load_img(img_path, target_size=(200, 200))
    img_array = img_to_array(img) / 255  #转换为array数组并标准化
    shape = img_array.shape
    img_array = img_array.reshape((1,shape[0],shape[1],shape[2]))
    bndbox = model1.predict(img_array).reshape((4,))
    return bndbox
 #%%   
for path in getAllImages(folder_path):
    #读图
    file_name = os.path.split(path)
    img = Image.open(path)
    bndbox = getbox(path)*512
    bndbox = tuple(int(i) for i in bndbox)
    #从图中剪裁出甲状腺区域图片来，（左，上，右，下）的坐标模式
    roi=img.crop(bndbox)
    save_path = os.path.join(save_folder_path, file_name[1])
    roi.save(save_path)   #保存到指定文件夹。

#%%

for path in getAllImages(folder_path):
    #读图
    file_name = os.path.split(path)
    image_path = os.path.join(folder_path_2, file_name[1])
    img = Image.open(image_path)
    i =+ 1
    bndbox = getbox(path)*512
    bndbox = tuple(int(i) for i in bndbox)
    #从图中剪裁出甲状腺区域图片来，（左，上，右，下）的坐标模式
    roi=img.crop(bndbox)
    save_path = os.path.join(save_folder_path_2, file_name[1])
    roi.save(save_path)   #保存到指定文件夹。






#%%    
#img_list = getAllImages(folder_path)
#nb_img_1 = len(img_list)
#a = getbox(img_list[0])*512
#b = tuple(int(i) for i in a)
#box_3 = (150,180,350,380)
#%%





