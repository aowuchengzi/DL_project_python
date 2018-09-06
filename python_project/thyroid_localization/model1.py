# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 19:46:39 2018

@author: 叶晨
"""

import numpy as np
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import backend as K
from load_data import get_data
#%%
data_dir = r'G:\picture\localization\image_output'
img_width, img_height = 200, 200
nb_train_samples = 977
nb_validation_samples = 1406
epochs = 10
batch_size = 20
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#%%
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
#%%
(x_train, y_train) = get_data(data_dir)#获得训练数据和标签
train_datagen = ImageDataGenerator(rescale=1.0 / 255)#数据标准化
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size, )#批训练数据生成器
#%%
inputs = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
x = BatchNormalization(axis=-1)(x)

x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
x = BatchNormalization(axis=-1)(x)

x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
x = BatchNormalization(axis=-1)(x)

x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
x = BatchNormalization(axis=-1)(x)

x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization(axis=-1)(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization(axis=-1)(x)
outputs = Dense(4, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=adam,
              loss='mean_squared_error',
              metrics=['mse', 'acc'])
#%%
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
auto_save = ModelCheckpoint(filepath="model1-1-weights-{epoch:02d}-{acc:.4f}-{val_acc:.4f}.h5",
                            verbose=1,
                            monitor="train_loss",
                            save_best_only=True,
                            save_weights_only= True,
                            mode='min')
                            
reducelr = ReduceLROnPlateau(monitor='val_loss',
                             factor=0.5,
                             patience=5,
                             cooldown=5,
                             min_lr=0.000001)
#%%
model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs
#        validation_data=validation_generator,
#        validation_steps=nb_validation_samples // batch_size+1,
#        callbacks=[auto_save]
#        callbacks=[reducelr,auto_save]
        )
#%%
model.save_weights('model1_weights_Epoch=30_mse=0.00039.h5')
#%%
model.save('model1_mse=0.0004.h5')
