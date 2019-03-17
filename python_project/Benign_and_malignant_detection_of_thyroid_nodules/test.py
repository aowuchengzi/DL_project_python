# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 10:25:55 2018

@author: 15339
"""

import numpy as np
from keras.applications.densenet import DenseNet121
from keras.applications.resnet50 import ResNet50
#%%

model = DenseNet121(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000)
model2 = ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000)
#%%
from keras.utils import plot_model
plot_model(model, to_file='densenet_model.png', show_shapes=True)

plot_model(model2, to_file='resnet_model.png', show_shapes=True)


#%%
#mask = array([[[0., 1.],
#               [0., 1.],
#               [0., 1.]],
#
#              [[0., 1.],
#               [0., 1.],
#               [0., 1.]],
#
#              [[0., 1.],
#               [0., 1.],
#               [0., 1.]]])
#new_mask = np.zeros((3, 3) + (2, ))
#new_mask
#mask == 0
#mask[1,1]=3
##%%
#for i in range(2):
#   
#    new_mask[mask == i,i] = 1
#new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) 
#mask = new_mask
#%%
from model4 import *
model_4 = model_4()
#%%
from keras.utils import plot_model
plot_model(model, to_file='model10.png', show_shapes=True)
#%%
from data_process import *
from keras.models import Model, load_model
model = load_model('model_4-0.99.h5')
folder_path = r'data/thyroid/test'
save_folder_path = r'test1'
#%%
for path in getAllImages(folder_path,file_type='.jpg'):
    #读图
    file_name = os.path.split(path)
    img = getpredict(path, model)
#    img_ = img_.reshape((256,256,1))*255
#    img = array_to_img(img_)
    save_path = os.path.join(save_folder_path, file_name[1])
    img.save(save_path)   #保存到指定文件夹。
    
    #%%
import pickle
with open('trainHistoryDict', 'wb') as file_pi: 
    pickle.dump(history.history, file_pi) 
     
     
#%%
with open('trainHistoryDict-dice/cnn1-orig&roi-0-200', 'rb') as fp:
    d = pickle.load(fp)
    