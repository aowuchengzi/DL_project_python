# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 13:56:05 2018

@author: 15339
"""

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
    x = Conv2D(filters1, 1, use_bias=False, kernel_initializer = 'he_normal', name=name+'_0_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name+'_0_bn')(x)
    x = Activation('relu', name=name+'_0_relu')(x) 
    x = Conv2D(filters2, 3, use_bias=False, padding = 'same', kernel_initializer = 'he_normal', name=name+'_1_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name+'_1_bn')(x)
    x = Activation('relu', name=name+'_1_relu')(x) 
    x = UpSampling2D(size = (2,2), name=name+'_upsample')(x) 
    return x

def denseunet(pretrained_weights = None, input_size = (256,256,1)):
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
    #以上为 第五层 卷积层 尺寸：16>>16 通道：512>>1024

    x = Conv2D(512, 1, use_bias=False, kernel_initializer = 'he_normal', name='mid_down')(x) #16*16*512
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='mid_bn')(x)
    x = Activation('relu', name='mid_relu')(x) #16*16*512
    x = UpSampling2D(size = (2,2), name='mid_upsample')(x) #32*32*512
    
    skip_4 = rblock(x_4, 512, name='rblock_skip4')
    skip_3 = rblock(x_3, 256, name='rblock_skip3')
    skip_2 = rblock(x_2, 128, name='rblock_skip2')
    skip_1 = rblock(x_1, 64, name='rblock_skip1')
    
    x = upsampleConv(skip_4, x, 512, 256, name='conv6')
    x = upsampleConv(skip_3, x, 256, 128, name='conv7')
    x = upsampleConv(skip_2, x, 128, 64, name='conv8')
    
    x = concatenate([skip_1, x], axis=bn_axis, name='conv9_concat') #128*128*256
    x = Conv2D(64, 1, use_bias=False, kernel_initializer = 'he_normal', name='conv9_0_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv9_0_bn')(x)
    x = Activation('relu', name='conv9_0_relu')(x) #128*128*128
    x = Conv2D(64, 3, use_bias=False, padding = 'same', kernel_initializer = 'he_normal', name='conv9_1_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv9_1_bn')(x)
    x = Activation('relu', name='conv9_1_relu')(x) #128*128*64
    x = Conv2D(2, 3, use_bias=False, padding = 'same', kernel_initializer = 'he_normal', name='conv9_2_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv9_2_bn')(x)
    x = Activation('relu', name='conv9_2_relu')(x) #128*128*64
    outputs = Conv2D(1, 1, activation = 'sigmoid')(x)
    
    model = Model(inputs = inputs, outputs = outputs)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
    
    
