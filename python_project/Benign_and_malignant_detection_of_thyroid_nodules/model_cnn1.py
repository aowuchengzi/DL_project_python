# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:59:32 2019

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
from keras import backend as K



def dense_block(x, blocks, name):
    
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def conv_block(x, growth_rate, name):
    
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False, kernel_initializer = 'he_normal', name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False, kernel_initializer = 'he_normal', name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def cnn1(pretrained_weights = None,input_size = (256,256,1)):
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    inputs = Input(input_size)
    
    x1 = Conv2D(32, 3, use_bias=False, padding = 'same', kernel_initializer = 'he_normal', name='conv1_conv')(inputs)
    x2 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_0_bn')(x1)
    x2 = Activation('relu', name='conv1_0_relu')(x2)
    x2 = Conv2D(64, 1, use_bias=False, kernel_initializer = 'he_normal', name='conv1_0_conv')(x2)
    x2 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_1_bn')(x2)
    x2 = Activation('relu', name='conv1_1_relu')(x2)
    x2 = Conv2D(32, 3, use_bias=False, kernel_initializer = 'he_normal', padding = 'same', name='conv1_1_conv')(x2)
    x = Concatenate(axis=bn_axis, name='conv1_concat')([x1, x2])
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='pool1_bn')(x)
    x_1 = Activation('relu', name='pool1_relu')(x) #256*256*64
    x = MaxPooling2D(strides=2, name='pool1_pool')(x_1)
    #以上为 第一层 卷积层 尺寸：256>>128 通道：1>>64
    x = dense_block(x, 2, name='conv2') 
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='pool2_bn')(x)
    x_2 = Activation('relu', name='pool2_relu')(x) #128*128*128
    x = MaxPooling2D(strides=2, name='pool2_pool')(x_2)
    #以上为 第二层 卷积层 尺寸：128>>64 通道：64>>128
    x = dense_block(x, 4, name='conv3') 
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='pool3_bn')(x)
    x_3 = Activation('relu', name='pool3_relu')(x) #64*64*256
    x = MaxPooling2D(strides=2, name='pool3_pool')(x_3)
    #以上为 第三层 卷积层 尺寸：64>>32 通道：128>>256
    x = dense_block(x, 8, name='conv4') 
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='pool4_bn')(x)
    x_4 = Activation('relu', name='pool4_relu')(x) #32*32*512
    x = MaxPooling2D(strides=2, name='pool4_pool')(x_4)
    #以上为 第四层 卷积层 尺寸：32>>16 通道：256>>512
    x = dense_block(x, 16, name='conv5') 
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='pool5_bn')(x)
    x = Activation('relu', name='pool5_relu')(x) #16*16*1024
    
    x = AveragePooling2D(16, name='avg_pool')(x)
    outputs_2 = Flatten(name='flatten')(x)
    outputs_1 = Dense(2, activation='softmax', name='fc2')(outputs_2)
    
    model = Model(inputs = inputs, outputs = outputs_1)

    model.compile(optimizer = Adam(lr = 1e-3), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
    return model