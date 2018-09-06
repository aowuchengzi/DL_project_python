# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:32:50 2018

@author: 叶晨
"""

#%%
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, Input, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, Input
from keras.layers import concatenate
from keras.optimizers import Adam
from keras import backend as K
#%%
#图像尺寸设定
img_width, img_height = 200, 200

#train_data_dir = r'F:\picture\train'
#validation_data_dir = r'F:\picture\validation'
train_data_dir = r'F:\picture\Preprocess_imageset\1\train'
validation_data_dir = r'F:\picture\Preprocess_imageset\1\validation'
nb_train_samples = 5552
nb_validation_samples = 1406
epochs = 5
batch_size = 20
adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#%%
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#%%
train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,              #数据标准化
        rotation_range=20,              #随机转动的角度
        zoom_range=0.1,                 #随机缩放的幅度
        shear_range=0.1,                 #剪切强度（逆时针方向的剪切变换角度）
        horizontal_flip=True)            #水平翻转

test_datagen = ImageDataGenerator(rescale=1. /255)


train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        class_mode='categorical',
        batch_size=batch_size,
        seed=1)

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width,img_height),
        class_mode='categorical',
        batch_size=batch_size,
        seed=1,
        shuffle=False
        )
#%%
cnn1 = load_model('model1-1_full_ac=90.97-90.97.h5')
cnn2 = load_model('model2-1_full_ac=89.24-88.62.h5')
#%%
cnn1 = load_model('model1-2_full_ac=91.52-90.49.h5')
cnn2 = load_model('model2-2_full_ac=87.68-87.61.h5')
#%%
cnn1 = load_model('model1-3_full_ac=91.74-91.14.h5')
cnn2 = load_model('model2-3_full_ac=90.12-90.20.h5')
#%%
cnn1 = load_model('model1-4_full_ac=91.44-90.42.h5')
cnn2 = load_model('model2-4_full_ac=90.33-90.42.h5')
#%%
cnn1 = load_model('model1-5_full_ac=91.29-91.14.h5')
cnn2 = load_model('model2-5_full_ac=86.63-86.31.h5')
#%%

cnn2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
cnn1.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#%%
x = Input(shape=input_shape)
x1 = cnn1(inputs=x)
x2 = cnn2(inputs=x)
x3 = concatenate(inputs=[x1,x2], axis=-1)
predictions = Dense(2, activation='softmax')(x3)

cnnf = Model(inputs=x, outputs=predictions, name='cnnf')
#%%
cnnf.load_weights('cnnf-1-weights-05-0.9682-0.9147.h5')
#%%
for layer in cnnf.layers[:4]:
    layer.trainable = False
for layer in cnnf.layers[4:]:
    layer.trainable = True
#%%
cnnf.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#%%
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
auto_save = ModelCheckpoint(filepath="cnnf-1-weights-{epoch:02d}-{acc:.4f}-{val_acc:.4f}.h5",
                            verbose=1,
                            monitor="val_acc",
                            save_best_only=True,
                            save_weights_only= True,
                            mode='max',
                            )
                                             
reducelr = ReduceLROnPlateau(monitor='val_loss',
                             factor=0.5,
                             patience=5,
                             cooldown=5,
                             min_lr=0.000001)
#%%
cnnf.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size+1,
        callbacks=[auto_save]
#        callbacks=[reducelr,auto_save]
        )
#%%
cnnf.evaluate_generator(
        validation_generator,
        steps=nb_validation_samples // batch_size+1
        
        )
#%%
cnn1.evaluate_generator(
        validation_generator,
        steps=nb_validation_samples // batch_size+1
        
        )
#%%
cnn2.evaluate_generator(
        validation_generator,
        steps=nb_validation_samples // batch_size+1
        
        )
#%%
cnnf.save('cnnf_full_ac=91.47-91.47.h5')
#%%
from keras.utils import plot_model
plot_model(cnnf, to_file='cnnf.png', show_shapes=True, show_layer_names=True)  
#%%
cnn1 = Sequential()

cnn1.add(Conv2D(32, (3, 3), input_shape=input_shape))
cnn1.add(Activation('relu'))
cnn1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
cnn1.add(BatchNormalization(axis=-1))

cnn1.add(Conv2D(64, (3, 3)))
cnn1.add(Activation('relu'))
cnn1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
cnn1.add(BatchNormalization(axis=-1))

cnn1.add(Conv2D(64, (3, 3)))
cnn1.add(Activation('relu'))
cnn1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
cnn1.add(BatchNormalization(axis=-1))

cnn1.add(Conv2D(128, (3, 3)))
cnn1.add(Activation('relu'))
cnn1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
cnn1.add(BatchNormalization(axis=-1))

cnn1.add(Flatten())
cnn1.add(Dense(64))
cnn1.add(Activation('relu'))
cnn1.add(BatchNormalization(axis=-1))
cnn1.add(Dropout(0.5))
cnn1.add(Dense(64))
cnn1.add(Activation('relu'))
cnn1.add(BatchNormalization(axis=-1))
#cnn1.add(Dropout(0.3))
cnn1.add(Dense(2))
cnn1.add(Activation('softmax'))



