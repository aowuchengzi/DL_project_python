# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 18:59:51 2019

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
from loss_function import dice_coef_loss, dice_coef

def dense_block_1(x, blocks, growth_rate, name):
    
    for i in range(blocks):
        x = conv_block(x, growth_rate, name=name + '_block' + str(i + 1))
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

def rblock(inputs, filters, name, scale=0.1):    
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    residual = Conv2D(filters, kernel_size=1, strides=1, padding='same', kernel_initializer = 'he_normal', name=name + '_conv')(inputs)
    residual = BatchNormalization(axis=bn_axis,epsilon=1.001e-5, name=name + '_bn')(residual)
    residual = Lambda(lambda x: x*scale, name=name + '_lambda')(residual)
    res = Add(name=name + '_add')([inputs, residual])
    res = Activation('relu', name=name + '_relu')(res)
    return res 

def upsampleConv(skip, x, filters1, filters2, name):
    x = concatenate([skip, x], axis=bn_axis, name=name+'_concat') 
    x = Conv2D(filters1, 3, use_bias=False, kernel_initializer = 'he_normal', name=name+'_0_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name+'_0_bn')(x)
    x = Activation('relu', name=name+'_0_relu')(x) 
    x = Conv2D(filters2, 3, use_bias=False, padding = 'same', kernel_initializer = 'he_normal', name=name+'_1_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name+'_1_bn')(x)
    x = Activation('relu', name=name+'_1_relu')(x) 
    x = UpSampling2D(size = (2,2), name=name+'_upsample')(x) 
    return x

def model_10(pretrained_weights = None,input_size = (256,256,1)):
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    inputs = Input(input_size)
    # 第一层
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
    pool1_128 = MaxPooling2D(pool_size=(2, 2))(x_1)
    # 第一层
    # 第二层
    x = dense_block_1(pool1_128, 2, 32, name='conv2') 
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='pool2_bn')(x)
    x_2 = Activation('relu', name='pool2_relu')(x) #128*128*128
    x = MaxPooling2D(strides=2, name='pool2_pool')(x_2)
    # 第二层
    # 第三层
    x = dense_block_1(x, 2, 64, name='conv3') 
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='pool3_bn')(x)
    x_3 = Activation('relu', name='pool3_relu')(x) #64*64*256
    x = MaxPooling2D(strides=2, name='pool3_pool')(x_3)
    # 第三层
    # 第四层
    x = dense_block_1(x, 4, 64, name='conv4') 
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='pool4_bn')(x)
    x_4 = Activation('relu', name='pool4_relu')(x) #32*32*512
    x = MaxPooling2D(strides=2, name='pool4_pool')(x_4)
    # 第四层
    # 第五层
    x = dense_block_1(x, 4, 128, name='conv5') 
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='pool5_bn')(x)
    x = Activation('relu', name='pool5_relu')(x) #16*16*1024

    
    conv5_16 = Conv2D(512, 2, padding = 'same', kernel_initializer = 'he_normal')(x)
    conv5_16 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv5_16)
    conv5_16 = Activation('relu')(conv5_16)
    up6_32 = UpSampling2D(size = (2,2))(conv5_16)
    
    skip_4 = rblock(x_4, 512, name='rblock_skip4')
    skip_3 = rblock(x_3, 256, name='rblock_skip3')
    skip_2 = rblock(x_2, 128, name='rblock_skip2')
    skip_1 = rblock(x_1, 64, name='rblock_skip1')
    
    merge6_32 = concatenate([skip_4, up6_32],axis=3)
    conv6_32 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(merge6_32)
    conv6_32 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv6_32)
    conv6_32 = Activation('relu')(conv6_32)
    conv6_32 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6_32)
    conv6_32 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv6_32)
    conv6_32 = Activation('relu')(conv6_32)
    conv6_32 = Conv2D(256, 2, padding = 'same', kernel_initializer = 'he_normal')(conv6_32)
    conv6_32 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv6_32)
    conv6_32 = Activation('relu')(conv6_32)
    up7_64 = UpSampling2D(size = (2,2))(conv6_32)
    
    merge7_64 = concatenate([skip_3, up7_64],axis=3)
    conv7_64 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7_64)
    conv7_64 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv7_64)
    conv7_64 = Activation('relu')(conv7_64)
    conv7_64 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7_64)
    conv7_64 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv7_64)
    conv7_64 = Activation('relu')(conv7_64)
    conv7_64 = Conv2D(128, 2, padding = 'same', kernel_initializer = 'he_normal')(conv7_64)
    conv7_64 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv7_64)
    conv7_64 = Activation('relu')(conv7_64)
    up8_128 = UpSampling2D(size = (2,2))(conv7_64)
    
    merge8_128 = concatenate([skip_2,up8_128],axis=3)
    conv8_128 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8_128)
    conv8_128 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv8_128)
    conv8_128 = Activation('relu')(conv8_128)
    conv8_128 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8_128)
    conv8_128 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv8_128)
    conv8_128 = Activation('relu')(conv8_128)
    conv8_128= Conv2D(64, 2, padding = 'same', kernel_initializer = 'he_normal')(conv8_128)
    conv8_128 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv8_128)
    conv8_128 = Activation('relu')(conv8_128)
    up9_256 = UpSampling2D(size = (2,2))(conv8_128)
    
    merge9_256 = concatenate([skip_1,up9_256],axis=3)
    conv9_256 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9_256)
    conv9_256 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv9_256)
    conv9_256 = Activation('relu')(conv9_256)
    conv9_256 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9_256)
    conv9_256 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv9_256)
    conv9_256 = Activation('relu')(conv9_256)
    conv9_256 = Conv2D(2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv9_256)
    conv9_256 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv9_256)
    conv9_256 = Activation('relu')(conv9_256)
    conv10_256 = Conv2D(1, 1, activation = 'sigmoid')(conv9_256)

    model = Model(inputs = inputs, outputs = conv10_256)

    model.compile(optimizer = Adam(lr = 0.0001), loss = dice_coef_loss, metrics = ['accuracy', dice_coef])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model