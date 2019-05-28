# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:04:35 2019

@author: 15339
"""

import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.utils import to_categorical
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from data_process import *
from data import *
from model_cnn1 import *
from model_cnn2 import *
#%%
base_model1 = cnn2(pretrained_weights=r'cnn2-4-w-133-0.9947-0.9406.h5')
img = r'data\thyroid_B&M_roi\validation-4'
test_datagen = ImageDataGenerator(rescale=1. /255)
generator = test_datagen.flow_from_directory(
        img,
        target_size=(256,256),
        class_mode='categorical',
        color_mode = 'grayscale',
        batch_size=1,
        seed=1,
        shuffle=False
        )

predict_cnn2 = base_model1.predict_generator(generator, verbose=1, steps=402)
#%%
base_model2 = cnn2(pretrained_weights=r'cnn2-w-197-1.0000-0.9602.h5')



#%%
base_model1 = cnn1(pretrained_weights=r'cnn1-4-w-06-1.0000-0.9604.h5')
base_model2 = cnn2(pretrained_weights=r'cnn2-4-w-133-0.9947-0.9406.h5')
model1 = Model(inputs = base_model1.input, outputs = base_model1.get_layer('flatten').output)
model2 = Model(inputs = base_model2.input, outputs = base_model2.get_layer('flatten').output)

imgT_1 = r'data\thyroid_B&M\train-4'
imgT_2 = r'data\thyroid_B&M_roi\train-4'
imgV_1 = r'data\thyroid_B&M\validation-4'
imgV_2 = r'data\thyroid_B&M_roi\validation-4'
#%%
'''
融合CNN-1和CNN-2 的提取的特征
'''
test_datagen = ImageDataGenerator(rescale=1. /255)
imgT_1_generator = test_datagen.flow_from_directory(
        imgT_1,
        target_size=(256,256),
        class_mode='categorical',
        color_mode = 'grayscale',
        batch_size=1,
        seed=1,
        shuffle=False
        )
imgT_2_generator = test_datagen.flow_from_directory(
        imgT_2,
        target_size=(256,256),
        class_mode='categorical',
        color_mode = 'grayscale',
        batch_size=1,
        seed=1,
        shuffle=False
        )
imgV_1_generator = test_datagen.flow_from_directory(
        imgV_1,
        target_size=(256,256),
        class_mode='categorical',
        color_mode = 'grayscale',
        batch_size=1,
        seed=1,
        shuffle=False
        )
imgV_2_generator = test_datagen.flow_from_directory(
        imgV_2,
        target_size=(256,256),
        class_mode='categorical',
        color_mode = 'grayscale',
        batch_size=1,
        seed=1,
        shuffle=False
        )
#%%
resultsT_1 = model1.predict_generator(imgT_1_generator, verbose=1, steps=1610)
resultsT_2 = model2.predict_generator(imgT_2_generator, verbose=1, steps=1610)
resultsT = np.concatenate((resultsT_1, resultsT_2), axis=1)

resultsV_1 = model1.predict_generator(imgV_1_generator, verbose=1, steps=402)
resultsV_2 = model2.predict_generator(imgV_2_generator, verbose=1, steps=402)
resultsV = np.concatenate((resultsV_1, resultsV_2), axis=1)



yT = np.array([] +[1]*1137 + [0]*473)
yT_ = to_categorical(yT, 2)
yV = np.array([] +[1]*284 + [0]*118)
yV_ = to_categorical(yV, 2)
indices = np.random.permutation(resultsT.shape[0])
x_train = resultsT[indices]
y_train = yT_[indices]
x_valid = resultsV
y_valid = yV_
np.save(r'data\cnnf\x_train_4.npy', x_train)
np.save(r'data\cnnf\y_train_4.npy', y_train)
np.save(r'data\cnnf\x_valid_4.npy', x_valid)
np.save(r'data\cnnf\y_valid_4.npy', y_valid)
#%%
x_train = np.load(r'data\cnnf\x_train_3.npy')
y_train = np.load(r'data\cnnf\y_train_3.npy')
x_valid = np.load(r'data\cnnf\x_valid_3.npy')
y_valid = np.load(r'data\cnnf\y_valid_3.npy')
epochs = 20
batch_size = 64
inputs = Input(shape=(1536,))
x = Dense(2)(inputs)
#x = Dense(2)(x)
#x = Dropout(0.1)(x)
outputs = Activation('softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

auto_save = ModelCheckpoint(filepath="cnnf-w-{epoch:02d}-{acc:.4f}-{val_acc:.4f}.h5",
                            verbose=1,
                            monitor="val_acc",
                            save_best_only=True,
                            save_weights_only= True,
                            mode='max',
                            )
#%%
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_valid, y_valid),
                    callbacks=[auto_save])
#%%
x_valid = np.load(r'data\cnnf\x_valid_4.npy')
inputs = Input(shape=(1536,))
x = Dense(2)(inputs)
outputs = Activation('softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.load_weights(r'cnnf-4-w-09-0.9975-0.9726.h5')
predict_cnnf = model.predict(x_valid, verbose=1, steps=1)

#%%
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Input, BatchNormalization
import numpy as np

#%%
model_vgg16 = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
x = BatchNormalization(axis=-1)(model_vgg16.output)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization(axis=-1)(x)
x = Dropout(0.1)(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization(axis=-1)(x)
predictions = Dense(2, activation='softmax')(x)
model_1 = Model(inputs=model_vgg16.input, outputs=predictions, name='pre_trained_model')
model_1.load_weights(r'vgg-w-24-0.9517-0.9279.h5')
img = r'data\thyroid_B&M\validation-3'
test_datagen = ImageDataGenerator(rescale=1. /255)
generator = test_datagen.flow_from_directory(
        img,
        target_size=(224,224),
        class_mode='categorical',
        color_mode = 'rgb',
        batch_size=1,
        seed=1,
        shuffle=False
        )

predict_VggNet = model_1.predict_generator(generator, verbose=1, steps=402)
#%%
model_2 = InceptionV3(weights='InceptionV3-w-76-0.9869-0.9378.h5', input_shape=(299, 299, 3), classes=2)
img = r'data\thyroid_B&M\validation-3'
test_datagen = ImageDataGenerator(rescale=1. /255)
generator = test_datagen.flow_from_directory(
        img,
        target_size=(299,299),
        class_mode='categorical',
        color_mode = 'rgb',
        batch_size=1,
        seed=1,
        shuffle=False
        )

predict_InceptionV3 = model_2.predict_generator(generator, verbose=1, steps=402)
#%%
model_3 = ResNet50(weights='ResNet50-w-86-0.9848-0.9328.h5', input_shape=(224, 224, 3), classes=2)
img = r'data\thyroid_B&M\validation-3'
test_datagen = ImageDataGenerator(rescale=1. /255)
generator = test_datagen.flow_from_directory(
        img,
        target_size=(224,224),
        class_mode='categorical',
        color_mode = 'rgb',
        batch_size=1,
        seed=1,
        shuffle=False
        )

predict_ResNet50 = model_3.predict_generator(generator, verbose=1, steps=402)
