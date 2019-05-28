# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:12:40 2019

@author: 15339
"""

from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Input, BatchNormalization
from keras.callbacks import ModelCheckpoint

train_data_dir = r'data\thyroid_B&M\train-3'
validation_data_dir = r'data\thyroid_B&M\validation-3'
nb_train_samples = 1610
nb_validation_samples = 402
batch_size = 16
#%%
#model_1 = VGG16(weights=None, input_shape=(224, 224, 3), classes=2)
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
#model_1.compile(optimizer=Adam(lr = 1e-3), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model_1.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss = 'categorical_crossentropy', metrics = ['accuracy'])
train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,              #数据标准化
        rotation_range=0.2,              #随机转动的角度
        zoom_range=0.1,                 #随机缩放的幅度
        shear_range=0.1,                 #剪切强度（逆时针方向的剪切变换角度）
        horizontal_flip=True)            #水平翻转

test_datagen = ImageDataGenerator(rescale=1. /255)
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=batch_size,
        seed=1,
        color_mode="rgb"
        )

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=batch_size,
        seed=1,
        color_mode="rgb",
        shuffle=False,
        )

auto_save = ModelCheckpoint(filepath="vgg-w-{epoch:02d}-{acc:.4f}-{val_acc:.4f}.h5",
                            verbose=1,
                            monitor="val_acc",
                            save_best_only=True,
                            save_weights_only= True,
                            mode='max',
                            )

history = model_1.fit_generator(
                            train_generator,
                            steps_per_epoch=nb_train_samples // batch_size,
                            epochs=100,
                            validation_data=validation_generator,
                            validation_steps=nb_validation_samples // batch_size+1,
                            callbacks=[auto_save]
                            )
#%%
model_2 = InceptionV3(weights=None, input_shape=(299, 299, 3),classes=2)
model_2.compile(optimizer=Adam(lr = 1e-3), loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,              #数据标准化
        rotation_range=0.2,              #随机转动的角度
        zoom_range=0.1,                 #随机缩放的幅度
        shear_range=0.1,                 #剪切强度（逆时针方向的剪切变换角度）
        horizontal_flip=True)            #水平翻转

test_datagen = ImageDataGenerator(rescale=1. /255)


train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(299, 299),
        class_mode='categorical',
        batch_size=batch_size,
        seed=1,
        color_mode="rgb"
        )

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(299, 299),
        class_mode='categorical',
        batch_size=batch_size,
        seed=1,
        color_mode="rgb",
        shuffle=False,
        )

auto_save = ModelCheckpoint(filepath="InceptionV3-w-{epoch:02d}-{acc:.4f}-{val_acc:.4f}.h5",
                            verbose=1,
                            monitor="val_acc",
                            save_best_only=True,
                            save_weights_only= True,
                            mode='max',
                            )

history = model_1.fit_generator(
                            train_generator,
                            steps_per_epoch=nb_train_samples // batch_size,
                            epochs=100,
                            validation_data=validation_generator,
                            validation_steps=nb_validation_samples // batch_size+1,
                            callbacks=[auto_save]
                            )
#%%
model_3 = ResNet50(weights=None, input_shape=(224, 224, 3),classes=2)
model_3.compile(optimizer=Adam(lr = 1e-3), loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,              #数据标准化
        rotation_range=0.2,              #随机转动的角度
        zoom_range=0.1,                 #随机缩放的幅度
        shear_range=0.1,                 #剪切强度（逆时针方向的剪切变换角度）
        horizontal_flip=True)            #水平翻转

test_datagen = ImageDataGenerator(rescale=1. /255)


train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=batch_size,
        seed=1,
        color_mode="rgb"
        )

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=batch_size,
        seed=1,
        color_mode="rgb",
        shuffle=False,
        )

auto_save = ModelCheckpoint(filepath="ResNet50-w-{epoch:02d}-{acc:.4f}-{val_acc:.4f}.h5",
                            verbose=1,
                            monitor="val_acc",
                            save_best_only=True,
                            save_weights_only= True,
                            mode='max',
                            )

history = model_1.fit_generator(
                            train_generator,
                            steps_per_epoch=nb_train_samples // batch_size,
                            epochs=100,
                            validation_data=validation_generator,
                            validation_steps=nb_validation_samples // batch_size+1,
                            callbacks=[auto_save]
                            )