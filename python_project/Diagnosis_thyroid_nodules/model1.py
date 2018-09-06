# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 18:32:28 2018

@author: 叶晨
"""
#%%
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
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
epochs = 30
batch_size = 32


#%%
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#%%
train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,              #数据标准化
        rotation_range=30,              #随机转动的角度
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
#        color_mode="grayscale"
        )

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width,img_height),
        class_mode='categorical',
        batch_size=batch_size,
        seed=1,
#        color_mode="grayscale",
        shuffle=False,
        )
#%%
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization(axis=-1))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization(axis=-1))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization(axis=-1))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(BatchNormalization(axis=-1))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
#model.add(Dropout(0.3))
model.add(Dense(2))
model.add(Activation('softmax'))
#%%
model.load_weights('model1-1-weights-20-0.9620-0.9097.h5')

#%%
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#%%
model.load_model('model1-4_full_ac=91.44.h5')

#%%
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
auto_save = ModelCheckpoint(filepath="model1-1-weights-{epoch:02d}-{acc:.4f}-{val_acc:.4f}.h5",
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
        validation_steps=nb_validation_samples // batch_size+1,
        callbacks=[auto_save]
#        callbacks=[reducelr,auto_save]
        )
#%%
model.save_weights('model2_weight_Epoch=295_ac=87_layer19.h5')
#%%
model.save('model1-1_full_ac=90.97-90.97.h5')

#%%
from keras.utils import plot_model
plot_model(model, to_file='model1.png', show_shapes=True, show_layer_names=True)  
#%%
model.evaluate_generator(
        validation_generator,
        steps=nb_validation_samples // batch_size+1
        )
        