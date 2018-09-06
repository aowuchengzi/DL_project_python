# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 21:46:47 2018

@author: 叶晨
"""

#%%
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.initializers import Constant
#%%
#图像尺寸设定
img_width, img_height = 200, 200

#train_data_dir = r'F:\picture\train'
#validation_data_dir = r'F:\picture\validation'
train_data_dir = r'F:\picture\Preprocess_imageset\train'
validation_data_dir = r'F:\picture\Preprocess_imageset\validation'
nb_train_samples = 6086
nb_validation_samples = 267
epochs = 30
batch_size = 32


#%%
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

#%%

PReLU_ = PReLU(alpha_initializer=Constant(value=0.25), shared_axes=[1,2,3])
#%%

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
#model.add(Conv2D(32, (3, 3)))
#model.add(Activation('relu'))
#model.add(LeakyReLU(alpha=0.3))
model.add(PReLU(alpha_initializer=Constant(value=0.25), shared_axes=[1,2,3]))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization(axis=-1))

model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(LeakyReLU(alpha=0.3))
model.add(PReLU(alpha_initializer=Constant(value=0.25), shared_axes=[1,2,3]))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(BatchNormalization(axis=-1))

model.add(Conv2D(128, (3, 3)))
#model.add(Activation('relu'))
#model.add(LeakyReLU(alpha=0.3))
model.add(PReLU(alpha_initializer=Constant(value=0.25), shared_axes=[1,2,3]))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization(axis=-1))

model.add(Conv2D(128, (3, 3)))
#model.add(Activation('relu'))
#model.add(LeakyReLU(alpha=0.3))
model.add(PReLU(alpha_initializer=Constant(value=0.25), shared_axes=[1,2,3]))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization(axis=-1))

model.add(Flatten())
#model.add(Dense(128))
##model.add(Activation('relu'))
#model.add(LeakyReLU(alpha=0.3))
#model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Activation('relu'))
#model.add(LeakyReLU(alpha=0.3))

model.add(BatchNormalization(axis=-1))
#model.add(Dropout(0.3))
model.add(Dense(2))
model.add(Activation('softmax'))
#%%
model.load_weights('model3.weights.h5')
#%%
from keras.utils import plot_model
plot_model(model, to_file='model3-1.png', show_shapes=True, show_layer_names=True)  

#%%
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#%%

train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,              #数据标准化
        rotation_range=30,              #随机转动的角度
        zoom_range=0.1,                 #随机缩放的幅度
        shear_range=0.2,                 #剪切强度（逆时针方向的剪切变换角度）
        horizontal_flip=True)            #水平翻转

test_datagen = ImageDataGenerator(rescale=1. /255)


train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        class_mode='categorical',
        batch_size=batch_size,
        seed=1,
        color_mode="grayscale")

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width,img_height),
        class_mode='categorical',
        batch_size=batch_size,
        seed=1,
        color_mode="grayscale")
#%%
#model.load_model('model3_full_.h5')
#%%
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
auto_save = ModelCheckpoint(filepath="model3_1-weights-improvement-{epoch:02d}-{val_acc:.2f}.h5",
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
model.save_weights('model2_weight_Epoch=295_ac=87_layer19.h5')
#%%
model.save('model2_full_Epoch=5_ac=.h5')

#%%
model.evaluate_generator(
        validation_generator,
        steps=125
        )





