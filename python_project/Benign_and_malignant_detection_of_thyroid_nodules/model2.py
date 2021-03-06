# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 20:20:32 2018

@author: 15339
"""

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def model_2(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1_256 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1_256 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_256)
    pool1_128 = MaxPooling2D(pool_size=(2, 2))(conv1_256)
    conv2_128 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1_128)
    conv2_128 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_128)
    pool2_64 = MaxPooling2D(pool_size=(2, 2))(conv2_128)
    conv3_64 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2_64)
    conv3_64 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_64)
    pool3_32 = MaxPooling2D(pool_size=(2, 2))(conv3_64)
    conv4_32 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3_32)
    conv4_32 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_32)
    drop4_32 = Dropout(0.5)(conv4_32)
    pool4_16 = MaxPooling2D(pool_size=(2, 2))(drop4_32)

    conv5_16 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4_16)
    conv5_16 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5_16)
    drop5_16 = Dropout(0.5)(conv5_16)

    conv5_16 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop5_16)
    up6_32 = UpSampling2D(size = (2,2))(conv5_16)
    merge6_32 = concatenate([drop4_32, up6_32],axis=3)
    conv6_32 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6_32)
    conv6_32 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6_32)

    conv6_32 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6_32)
    up7_64 = UpSampling2D(size = (2,2))(conv6_32)
    merge7_64 = concatenate([conv3_64, up7_64],axis=3)
    conv7_64 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7_64)
    conv7_64 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7_64)

    conv7_64 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7_64)
    up8_128 = UpSampling2D(size = (2,2))(conv7_64)
    merge8_128 = concatenate([conv2_128,up8_128],axis=3)
    conv8_128 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8_128)
    conv8_128 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8_128)

    conv8_128= Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8_128)
    up9_256 = UpSampling2D(size = (2,2))(conv8_128)
    merge9_256 = concatenate([conv1_256,up9_256],axis=3)
    conv9_256 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9_256)
    conv9_256 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9_256)
    conv9_256 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9_256)
    conv10_256 = Conv2D(1, 1, activation = 'sigmoid')(conv9_256)

    model = Model(inputs = inputs, outputs = conv10_256)

    model.compile(optimizer = Adam(lr = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


