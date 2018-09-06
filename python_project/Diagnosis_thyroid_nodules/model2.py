# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 21:03:16 2018

@author: 叶晨
"""

#%%
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input, BatchNormalization
from keras.optimizers import Adam
#%%
#图像尺寸设定
img_width, img_height = 200, 200

#train_data_dir = r'F:\picture\train'
#validation_data_dir = r'F:\picture\validation'
train_data_dir = r'F:\picture\Preprocess_imageset\1\train'
validation_data_dir = r'F:\picture\Preprocess_imageset\1\validation'
nb_train_samples = 5552
nb_validation_samples = 1406
epochs = 10
batch_size = 20


#%%
#载入VGG16模型。并且载入预训练过的权重
model_vgg16 = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
print('Model loaded.')
#%%
model_vgg16 = applications.VGG16(weights=None, include_top=False, input_shape=(img_width, img_height, 3))
print('Model loaded no weights.')
#%%
#建立一个在VGG16模型顶上的全连接网络。
x = BatchNormalization(axis=-1)(model_vgg16.output)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization(axis=-1)(x)
x = Dropout(0.1)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization(axis=-1)(x)

predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=model_vgg16.input, outputs=predictions, name='pre_trained_model')
#%%
model.load_weights('model2-1-weights-02-0.9172-0.8924.h5')
#%%
for layer in model.layers[:19]:
    layer.trainable = False
for layer in model.layers[19:]:
    layer.trainable = True
#%%
from keras.utils import plot_model
plot_model(model, to_file='model2.png', show_shapes=True, show_layer_names=True)  
#%%
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-5, momentum=0.9),
              metrics=['accuracy'])
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
        shuffle=False,
        seed=1)
#%%
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
auto_save = ModelCheckpoint(filepath="model2-1-weights-{epoch:02d}-{acc:.4f}-{val_acc:.4f}.h5",
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
model.save_weights('model2-3-weights-06-0.9219-0.9275.h5')
#%%
model.save('model2-1_full_ac=89.24-88.62.h5')

#%%
model.evaluate_generator(
        validation_generator,
        steps=nb_validation_samples // batch_size+1
        )
