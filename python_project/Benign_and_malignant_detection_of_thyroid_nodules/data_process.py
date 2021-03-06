# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 10:38:55 2018

@author: 叶晨
"""
from PIL import Image
import numpy as np
import os
import shutil
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.models import Model, load_model
#%%
def getAllImages(path, file_type='.png'):
    #读取path路径下的 png文件
    #f.endswith（）  限制文件类型
    #注意 返回的是绝对路径
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(file_type)]

def getAllfiles(path):
    #读取path路径下的 png文件
    #f.endswith（）  限制文件类型
    #注意 返回的是绝对路径
    return [os.path.join(path,f) for f in os.listdir(path)]
#%%
def visual_mask(folder_path, save_folder_path, file_type='.png'):
    #把matlab生成的0-1掩模可视化
    '''
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
#%%        
def image_resize(folder_path, save_folder_path, target_size=(512,512),  file_type='.png'):
    # 图片改变尺寸并保存
    for path in getAllImages(folder_path, file_type):
        #读图
        file_name = os.path.split(path)
        img = Image.open(path)
        img = img.resize(target_size)
        save_path = os.path.join(save_folder_path, file_name[1])
        img.save(save_path)   #保存到指定文件夹。
#%%
def Imglist_divide(img_list, divide_num, shuffle=True, seed=None):
    #把图片列表随机分成N份
    np.random.seed(seed)
    num_a = int(len(img_list)/5)
    img_list_list = []
    if shuffle:
        np.random.shuffle(img_list)
    for i in range(divide_num):
        if i<divide_num-1:
            img_list_list.append(img_list[i*num_a:i*num_a + num_a])
        else:
            img_list_list.append(img_list[i*num_a:])
    return img_list_list

def copyTofolder(img_list_list, img_path, divide_num):   
    #把 img_list_list里的N份图片复制到 img_path 文件夹下的N个文件夹里
    for i in range(divide_num):
        for a in img_list_list[i]:
            shutil.copy(a, os.path.join(img_path, str(i)))
    return
#%%
def n_fold_cross_validation(img_path, save_path, divide_num, img_type='png', shuffle=True, seed=None):
    #把 img path 文件夹里的图片做N折交叉验证的准备工作。
    '''
    img path：图片文件夹
    save path：保存目录
    divide num ：多少折  ,一般 十折或者 五折
    img type ：图片类型，默认png
    '''
    save_path = os.path.join(save_path, str(divide_num)+'折交叉验证')
    save_path_all = os.path.join(save_path, 'all')
    os.mkdir(save_path)
    os.mkdir(save_path_all)
    for i in range(divide_num):
        os.mkdir(os.path.join(save_path_all, str(i)))
        os.mkdir(os.path.join(save_path, str(i)))  # 在保存目录下生成N个文件夹用于保存图片
        for j in range(2):
            os.mkdir(os.path.join(os.path.join(save_path, str(i)), str(j)))
            
    img_list = getAllImages(img_path, file_type=img_type)
    img_list_list = Imglist_divide(img_list, divide_num, shuffle, seed)
    copyTofolder(img_list_list, save_path_all, divide_num)
    
    for i in range(divide_num):
        a1 = []
        a0 = []
        for j in range(divide_num):
            if i==j:
                a0 = img_list_list[j]
            else:
                a1 = a1 + img_list_list[j]
                
        for a in a0:
            shutil.copy(a, os.path.join(os.path.join(save_path, str(i)), '0'))
        for a in a1:
            shutil.copy(a, os.path.join(os.path.join(save_path, str(i)), '1'))
#%%
def img_cut(img_path, save_folder_path, box, img_type='.png'):
    #图像的裁剪
    '''
    img_path: 图片路径
    save_path： 保存目录，和原图片目录不同
    box： 裁剪区域，（x1，y1，x2，y2）的坐标模式
    '''
    file_name = os.path.split(img_path) 
    img1 = Image.open(img_path) #读图
    roi = img1.crop(box) #从图中剪裁出图片来，（左，上，右，下）的坐标模式
    save_path = os.path.join(save_folder_path, file_name[1])
    roi.save(save_path)   #保存到指定文件夹。       
     
def img_batch_cut(img_path, save_path, box, img_type='.png'):
    #图像的批量裁剪
    '''
    img_path: 图片目录
    save_path： 保存目录，和原图片目录不同
    box： 裁剪区域，（x1，y1，x2，y2）的坐标模式
    '''
    for path in getAllImages(img_path, file_type=img_type):
        file_name = os.path.split(path) 
        img = Image.open(path) #读图
        roi=img.crop(box) #从图中剪裁出图片来，（左，上，右，下）的坐标模式
        save_path_ = os.path.join(save_path, file_name[1])
        roi.save(save_path_)   #保存到指定文件夹。
        
#%%
def getpredict(img_path, model, save_folder_path, target_size=(256,256)):
    #利用训练好的模型
    #获取一张图片的mask
    '''
    img_path:图像目录
    model:模型对象
    save_folder_path:mask保存路径
    '''
    
    for path in getAllfiles(img_path):
        file_name = os.path.split(path)
        img = load_img(path, color_mode='grayscale', target_size=target_size, interpolation='bilinear')
        img_array = img_to_array(img) / 255  #转换为array数组并标准化
        shape = img_array.shape
        img_array = img_array.reshape((1,shape[0],shape[1],shape[2]))
        img = model.predict(img_array)
        img = img.reshape((256,256,1))*255
        img = array_to_img(img)
        save_path = os.path.join(save_folder_path, r'mask_' + file_name[1])
        img.save(save_path) 
    return print(r'生成完毕')
#%%
def mask_to_roi(img_path, mask_path, save_path, target_size=(1024,1024)):
    #利用mask和原图片生成roi图片
    '''
    img_path: 图片目录
    mask_path： 生成的掩模图片位置
    save_path： 保存目录，请设置成不同于原图片目录
    '''
    img_path_list = getAllfiles(img_path)
    mask_path_list = getAllfiles(mask_path)
    for i in range(len(img_path_list)):
        file_name = os.path.split(img_path_list[i])
        img = load_img(img_path_list[i], color_mode='grayscale', target_size=target_size, interpolation='bilinear')
        mask = load_img(mask_path_list[i], color_mode='grayscale', target_size=target_size, interpolation='bilinear')
        image_array = img_to_array(img)
        mask_array = img_to_array(mask)/255
        roi_array = np.multiply(image_array, mask_array)
        roi = array_to_img(roi_array)
        roi_save_path = os.path.join(save_path, r'roi_' + file_name[1])
        roi.save(roi_save_path)
    return print(r'roi生成完毕')

def mask_to_disroi(img_path, mask_path, save_path, target_size=(1024,1024)):
    #利用mask和原图片生成反roi图片
    '''
    img_path: 图片目录
    mask_path： 生成的掩模图片位置
    save_path： 保存目录，请设置成不同于原图片目录
    '''
    img_path_list = getAllfiles(img_path)
    mask_path_list = getAllfiles(mask_path)
    for i in range(len(img_path_list)):
        file_name = os.path.split(img_path_list[i])
        img = load_img(img_path_list[i], color_mode='grayscale', target_size=target_size, interpolation='bilinear')
        mask = load_img(mask_path_list[i], color_mode='grayscale', target_size=target_size, interpolation='bilinear')
        image_array = img_to_array(img)
        mask_array = 1 - img_to_array(mask)/255
        roi_array = np.multiply(image_array, mask_array)
        roi = array_to_img(roi_array)
        roi_save_path = os.path.join(save_path, r'disroi_' + file_name[1])
        roi.save(roi_save_path)
    return print(r'反roi生成完毕')