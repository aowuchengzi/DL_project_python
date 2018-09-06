# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:24:51 2018

@author: 叶晨
"""
import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from xml_to_dict import xmlfile_to_dict, get_bndbox_from_dict
#%%
def get_data(path):
    data_dir = path
    img_dir = os.path.join(data_dir,"JPEGImages")
    label_dir = os.path.join(data_dir,"Annotations")
    img_list = []
    label_list = []
    for file in os.listdir(img_dir):
        file_dir = os.path.join(img_dir, file)
        img_pil = load_img(file_dir, target_size=(200,200))
        img_array = img_to_array(img_pil)
        img_list.append(img_array)
        
    for file in os.listdir(label_dir):
        file_dir = os.path.join(label_dir, file)
        label_dict = xmlfile_to_dict(file_dir)
        label_array = get_bndbox_from_dict(label_dict)
        label_list.append(label_array)
        
        
    img_list_array = np.array(img_list)
    label_list_array = np.array(label_list)
    
    return img_list_array, label_list_array
    
    
 
#%%        
#a, b = get_data(r'F:\picture\localization\image_output')
