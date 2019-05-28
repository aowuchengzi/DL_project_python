# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:06:45 2019

@author: 15339
"""
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from data_process import *
from data import *
from model_cnn1 import *
from model_cnn2 import *
#%%
#图像尺寸设定
img_width, img_height = 256, 256
train_data_dir = r'data\thyroid_B&M\train'
validation_data_dir = r'data\thyroid_B&M\validation'
nb_train_samples = 1610
nb_validation_samples = 402
epochs = 100
batch_size = 16
#%%
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

#%%
train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,              #数据标准化
        rotation_range=0.2,              #随机转动的角度
        zoom_range=0.1,                 #随机缩放的幅度
        shear_range=0.1,                 #剪切强度（逆时针方向的剪切变换角度）
        horizontal_flip=True)            #水平翻转

test_datagen = ImageDataGenerator(rescale=1. /255)


train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        class_mode='categorical',
        batch_size=batch_size,
        seed=1,
        color_mode="grayscale"
        )

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width,img_height),
        class_mode='categorical',
        batch_size=batch_size,
        seed=1,
        color_mode="grayscale",
        shuffle=False,
        )

auto_save = ModelCheckpoint(filepath="cnn1-w-{epoch:02d}-{acc:.4f}-{val_acc:.4f}.h5",
                            verbose=1,
                            monitor="val_acc",
                            save_best_only=True,
                            save_weights_only= True,
                            mode='max',
                            )
#%%
model = cnn1()
#%%
history = model.fit_generator(
                            train_generator,
                            steps_per_epoch=nb_train_samples // batch_size,
                            epochs=epochs,
                            validation_data=validation_generator,
                            validation_steps=nb_validation_samples // batch_size+1,
                            callbacks=[auto_save]
                            )

#%%
K.set_value(model.optimizer.lr, 0.0001)
#%%
model.save_weights('权重保存\cnn1-roi_w_Epoch=100.h5')
model.save('model1-1_full_ac=90.97-90.97.h5')
#%%
import pickle
with open('trainHistoryDict-cnn/cnn1-200-300', 'wb') as file_pi: 
     pickle.dump(history.history, file_pi) 

























