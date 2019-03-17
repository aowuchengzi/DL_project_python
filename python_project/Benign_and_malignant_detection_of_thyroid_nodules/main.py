from data import *
#%%
from model import *
from model2 import *
from model3 import *
from model4 import *
from model5 import *
from model6 import *
from model7 import *
from model8 import *
from model9 import *
from model10 import *
from denseunet import *
from data_process import *
#%%
train_data_dir = r'data/thyroid/train'
validation_data_dir = r'data/thyroid/validation'
nb_train_samples = 400
nb_validation_samples = 100
epochs = 200
batch_size = 2
#%%
train_gen_args = dict(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
        )

validate_gen_args = dict(fill_mode='nearest')

trainGene = trainGenerator(
        batch_size,
        'data/thyroid/train',
        'image','label',
        train_gen_args,
#        save_to_dir = r'data\thyroid\train\aug',
        save_to_dir = None
        )
validateGene = trainGenerator(
        batch_size,
        'data/thyroid/validation',
        'image','label',
        validate_gen_args,
        save_to_dir = None
        )
#%%
model_1 = unet() # 原版 U-Net
model_2 = model_2() # 调换了一下 上采样层 和 降通道卷积层 的位置
model_3 = model_3() # 在 model2 的基础上 替换了 特征图传递 方式
model_4 = model_4() # 在 model2 的基础上 增加了 BN层
model_5 = model_5() # 在 model4 的基础上 把第一层卷积层用 dense_block 取代
model_6 = model_6() # 在 model4 的基础上 把第一和 第二层 层卷积层用 dense_block 取代
model_7 = model_7() # 在 model4 的基础上 把第一到第5层 层卷积层用 dense_block 取代
model_8 = model_8() # 在 model4 的基础上 把第一到第4层 层卷积层用 dense_block 取代
model_9 = model_9()
model_10 = model_10()
#dense-unet = denseunet()
#%%
auto_save = ModelCheckpoint(filepath="model6-1-weights-{epoch:02d}-{acc:.4f}-{val_acc:.4f}.h5",
                            verbose=1,
                            monitor="val_acc",
                            save_best_only=True,
                            save_weights_only= True,
                            mode='max',
                            )
#%%
history = model_1.fit_generator(trainGene,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validateGene,
        validation_steps=nb_validation_samples // batch_size,
#        callbacks=[auto_save]
        )


#%%
import pickle
with open('trainHistoryDict-dice/model1-0-200', 'wb') as file_pi: 
     pickle.dump(history.history, file_pi) 
     #%%
model_10.save_weights('权重保存\model10-3-weights-200.h5')
model_10.load_weights('权重保存\model10-1-w-171-0.9905.h5')
model_10.save('model_10-94.32.h5')
#%%
from keras.utils import plot_model
plot_model(model, to_file='模型可视化\model_cnn1.png', show_shapes=True)

#%%
model_10 = model_10()
model_10.load_weights('权重保存\model10-1-w-171-0.9905.h5')

model = model_10
image_path = r'总数据\总数据_分类实验用\1024x1024\良性'
mask_path = r'总数据\总数据_分类实验用\1024x1024\mask\良性'
roi_save_path = r'总数据\总数据_分类实验用\1024x1024\roi\良性'
disroi_save_path = r'总数据\总数据_分类实验用\1024x1024\disroi\良性'
getpredict(image_path, model, save_folder_path=mask_path)
mask_to_roi(image_path, mask_path, roi_save_path)
mask_to_disroi(image_path, mask_path, disroi_save_path)



#%%
image_path = r'总数据\总数据_分类实验用\1172x1100\良性'
saved_path = r'总数据\总数据_分类实验用\1024x1024\良性'
box = (74,0,1098,1024)
img_batch_cut(image_path, saved_path, box, img_type='.jpg')

#%%
image_path = r'总数据\总数据_分类实验用\1024x1024\roi\恶性'
save_path = r'总数据\总数据_分类实验用\1024x1024\roi'
n_fold_cross_validation(image_path, save_path, 5, img_type='jpg', shuffle=True, seed=1)


















