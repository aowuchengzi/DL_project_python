# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 09:02:15 2018

@author: 叶晨
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:32:50 2018

@author: 叶晨
"""

#%%
from keras import applications
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, list_pictures
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, Input, BatchNormalization
from keras.optimizers import Adam
#%%
#图像尺寸设定
img_width, img_height = 200, 200


validation_ = r'F:\picture\Preprocess_imageset\1\validation'
validation_0 = r'F:\picture\Preprocess_imageset\predict\1\0'
validation_1 = r'F:\picture\Preprocess_imageset\predict\1\1'
validation_1_fail = r'F:\picture\Preprocess_imageset\predict\1\fail'
nb_validation_ = 1406
nb_validation_0 = 680 
nb_validation_1 = 726
epochs = 5
batch_size = 1
#%%
test_datagen = ImageDataGenerator(rescale=1. /255)
#%%
validation_generator = test_datagen.flow_from_directory(
        validation_1,
        target_size=(img_width,img_height),
        class_mode='categorical',
        batch_size=batch_size,
        seed=1,
        shuffle=False
        )
#%%
validation_generator = test_datagen.flow_from_directory(
        validation_,
        target_size=(img_width,img_height),
        class_mode='categorical',
        batch_size=batch_size,
        seed=1,
        shuffle=False
        )
#%%
cnn1 = load_model('model1-1_full_ac=90.97-90.97.h5')
cnn2 = load_model('model2-1_full_ac=89.24-88.62.h5')
cnnf = load_model('cnnf_full_ac=91.47-91.47.h5')

#%%
#%%
a3 = cnnf.predict_generator(
                           validation_generator,
                           verbose=1,
                           steps=nb_validation_ // batch_size,
                           )

#%%
a1 = cnn1.predict_generator(
                           validation_generator,
                           verbose=1,
                           steps=nb_validation_ // batch_size,
                           )
#%%
a2 = cnn2.predict_generator(
                           validation_generator,
                           verbose=1,
                           steps=nb_validation_ // batch_size,
                           )        
#%%
a3_1 = cnnf.predict_generator(
                           validation_generator,
                           verbose=1,
#                           steps=nb_validation_1 // batch_size,
                           steps=1,
                           )






































#%%
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
cnn2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
cnn1.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
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
b = cnn1.predict_generator(
                           validation_generator,
                           verbose=1,
                           steps=nb_validation_1
                       
                           )
        





