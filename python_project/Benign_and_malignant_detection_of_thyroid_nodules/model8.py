# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 18:38:34 2019

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

def model_8(pretrained_weights = None,input_size = (256,256,1)):
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    inputs = Input(input_size)
    # 第一层
#    conv1_256 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
#    conv1_256 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv1_256)
#    conv1_256 = Activation('relu')(conv1_256)
#    conv1_256 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1_256)
#    conv1_256 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv1_256)
#    conv1_256 = Activation('relu')(conv1_256)
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
#    conv2_128 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1_128)
#    conv2_128 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv2_128)
#    conv2_128 = Activation('relu')(conv2_128)
#    conv2_128 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2_128)
#    conv2_128 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv2_128)
#    conv2_128 = Activation('relu')(conv2_128)
#    pool2_64 = MaxPooling2D(pool_size=(2, 2))(conv2_128)
    x = dense_block(pool1_128, 2, name='conv2') 
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='pool2_bn')(x)
    x_2 = Activation('relu', name='pool2_relu')(x) #128*128*128
    x = MaxPooling2D(strides=2, name='pool2_pool')(x_2)
    # 第二层
    # 第三层
#    conv3_64 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
#    conv3_64 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv3_64)
#    conv3_64 = Activation('relu')(conv3_64)
#    conv3_64 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3_64)
#    conv3_64 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv3_64)
#    conv3_64 = Activation('relu')(conv3_64)
#    pool3_32 = MaxPooling2D(pool_size=(2, 2))(conv3_64)
    x = dense_block(x, 4, name='conv3') 
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='pool3_bn')(x)
    x_3 = Activation('relu', name='pool3_relu')(x) #64*64*256
    x = MaxPooling2D(strides=2, name='pool3_pool')(x_3)
    # 第三层
    # 第四层
#    conv4_32 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3_32)
#    conv4_32 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv4_32)
#    conv4_32 = Activation('relu')(conv4_32)
#    conv4_32 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4_32)
#    conv4_32 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv4_32)
#    conv4_32 = Activation('relu')(conv4_32)
#    pool4_16 = MaxPooling2D(pool_size=(2, 2))(conv4_32)
    x = dense_block(x, 8, name='conv4') 
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='pool4_bn')(x)
    x_4 = Activation('relu', name='pool4_relu')(x) #32*32*512
    x = MaxPooling2D(strides=2, name='pool4_pool')(x_4)
    # 第四层
    # 第五层
    conv5_16 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
    conv5_16 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv5_16)
    conv5_16 = Activation('relu')(conv5_16)
    conv5_16 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5_16)
    conv5_16 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv5_16)
    conv5_16 = Activation('relu')(conv5_16)
#    x = dense_block(x, 16, name='conv5') 
#    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='pool5_bn')(x)
#    x = Activation('relu', name='pool5_relu')(x) #16*16*1024
    # 第五层
    conv5_16 = Conv2D(512, 2, padding = 'same', kernel_initializer = 'he_normal')(conv5_16)
    conv5_16 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(conv5_16)
    conv5_16 = Activation('relu')(conv5_16)
    up6_32 = UpSampling2D(size = (2,2))(conv5_16)
    
    merge6_32 = concatenate([x_4, up6_32],axis=3)
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
    
    merge7_64 = concatenate([x_3, up7_64],axis=3)
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
    
    merge8_128 = concatenate([x_2,up8_128],axis=3)
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
    
    merge9_256 = concatenate([x_1,up9_256],axis=3)
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

    model.compile(optimizer = Adam(lr = 0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model