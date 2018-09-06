# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 18:42:37 2018

@author: 叶晨
"""

#%%
from keras.applications.densenet import DenseNet
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input, BatchNormalization
from keras.optimizers import Adam
#%%
#图像尺寸设定
img_width, img_height = 221, 221

#train_data_dir = r'F:\picture\train'
#validation_data_dir = r'F:\picture\validation'
train_data_dir = r'F:\picture\Preprocess_imageset\train'
validation_data_dir = r'F:\picture\Preprocess_imageset\validation'
nb_train_samples = 6086
nb_validation_samples = 267
epochs = 10
batch_size = 20
#%%
model_densenet121 = DenseNet(blocks=[6, 12, 24, 16],
                             include_top=False,
#                             weights='imagenet',
                             weights=None,
                             input_tensor=None,
                             input_shape=(img_width, img_height, 3),
                             pooling='max',
                             classes=1000
                             )
#print('ImageNet_weights_loaded')
#%%
from keras.utils import plot_model
plot_model(model_densenet121, to_file='model4.png', show_shapes=True, show_layer_names=True)  
#%%
#建立一个全连接网络用于分类

predictions = Dense(2, activation='softmax')(model_densenet121.output)

model = Model(inputs=model_densenet121.input, outputs=predictions, name='pre_trained_model')
#%%
model.load_weights('model4_weights.h5')
#%%
from keras.utils import plot_model
plot_model(model, to_file='model4-1.png', show_shapes=True, show_layer_names=True)  
#%%
len(model.layers)
len(model_densenet121.layers)
model.layers[-6:]
#%%
for layer in model.layers[:-30]:
    layer.trainable = False
#%%
for layer in model.layers[-1:]:
    layer.trainable = True
#%%
for layer in model.layers[:]:
    layer.trainable = True
#%%
#%%
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-5, momentum=0.9),
              metrics=['accuracy'])
#%%
#%%
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#%%

train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,              #数据标准化
        rotation_range=20,              #随机转动的角度
        zoom_range=0.1,                 #随机缩放的幅度
        shear_range=0.2,                 #剪切强度（逆时针方向的剪切变换角度）
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
        seed=1)
#%%
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
auto_save = ModelCheckpoint(filepath="model4-weights-{epoch:02d}-{acc:.2f}-{val_acc:.2f}.h5",
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
model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[auto_save]
#        callbacks=[reducelr,auto_save]
        )
#%%
model.save_weights('model4_weights.h5')
#%%
model.save('model2_full_Epoch=5_ac=.h5')

#%%
model.evaluate_generator(
        validation_generator,
        steps=125
        )