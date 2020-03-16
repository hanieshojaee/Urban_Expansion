import os
import numpy as np
import h5py
import random
import tensorflow.keras as keras
from keras import Input
from keras.layers import UpSampling2D
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model, load_model
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from PIL import Image
from skimage.io import imread, imshow, imread_collection, concatenate_images
# from miou import MeanIoU
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


## Read data(new method)
Im0_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\AllDrivers\Im0.h5', 'r')
Im0  = np.array(Im0_hf.get('Im0')).astype(np.float32)
print(Im0.shape)
print("----------------------")
Im1_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\AllDrivers\Im1.h5', 'r')
Im1  = np.array(Im1_hf.get('Im1')).astype(np.float32)
print(Im1.shape)
print("----------------------")
Im2_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\AllDrivers\Im2.h5', 'r')
Im2  = np.array(Im2_hf.get('Im2')).astype(np.float32)
print(Im2.shape)
print("----------------------")
Im3_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\AllDrivers\Im3.h5', 'r')
Im3  = np.array(Im3_hf.get('Im3')).astype(np.float32)
print(Im3.shape)
print("----------------------")
Im4_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\AllDrivers\Im4.h5', 'r')
Im4  = np.array(Im4_hf.get('Im4')).astype(np.float32)
print(Im4.shape)
print("----------------------")



# Images_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\Images.h5', 'r')
# print(Images_hf.keys())
# Images = np.array(Images_hf.get('Images'))
# Images = np.expand_dims(Images, axis=3)
# print(Images.shape)
#
# Labels_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\Labels.h5', 'r')
# print(Labels_hf.keys())
# Labels = np.array(Labels_hf.get('Labels'))
# print(Labels.shape)


## Convert Label to One Hot Vector
# print(Labels.shape)
# Labels = keras.utils.to_categorical(Labels, num_classes = 2)
# print(Labels.shape)

## Read Image and Label from HDFfile
# Image_Label_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\Image_Label.h5', 'r')
# Image_Label = np.array(Image_Label_hf.get('Image_Label'))
# print(Image_Label.shape)

# Label_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\Label.h5', 'r')
# Label = np.array(Label_hf.get('Label'))
# print(Label.shape)


## Data Augmentation
def augment_data(dataset, augementation_factor=1, use_random_rotation=True,
                 use_random_shear=True, use_random_shift=True, use_random_zoom=True):
    augmented_image = []

    for num in range(0, dataset.shape[0]):
        for i in range(0, augementation_factor):

            # original image:
            augmented_image.append(dataset[num])

            rotation = [-180, -90, 90, 180]
            if use_random_rotation:
                augmented_image.append(
                    tf.contrib.keras.preprocessing.image.random_rotation(dataset[num], random.choice(rotation),
                                                                         row_axis=0, col_axis=1, channel_axis=2))

            # shear = [0.1, 0.2, 0.3, 0.4, 0.5]
            # if use_random_shear:
            #     augmented_image.append(tf.contrib.keras.preprocessing.image.random_shear(dataset[num], random.choice(rotation), row_axis=0, col_axis=1, channel_axis=2))

            # if use_random_shift:
            #     augmented_image.append(tf.contrib.keras.preprocessing.image.random_shift(dataset[num], 5, 5, row_axis=0, col_axis=1, channel_axis=2))

            # if use_random_zoom:
            #     augmented_image.append(tf.contrib.keras.preprocessing.image.random_zoom(dataset[num], (0.05, 0.45), row_axis=0, col_axis=1, channel_axis=2))

    return np.array(augmented_image)

print(Im0.shape)
Im0 = np.expand_dims(Im0, axis=0)
print(Im0.shape)
augmented_Im0 = augment_data(Im0, augementation_factor = 4)
print(augmented_Im0.shape)
print('---------------------------------')
print(Im1.shape)
Im1 = np.expand_dims(Im1, axis=0)
print(Im1.shape)
augmented_Im1 = augment_data(Im1, augementation_factor = 4)
print(augmented_Im1.shape)
print('---------------------------------')
print(Im2.shape)
Im2 = np.expand_dims(Im2, axis=0)
print(Im2.shape)
augmented_Im2 = augment_data(Im2, augementation_factor = 4)
print(augmented_Im2.shape)
print('---------------------------------')
print(Im3.shape)
Im3 = np.expand_dims(Im3, axis=0)
augmented_Im3 = augment_data(Im3, augementation_factor = 4)
print(augmented_Im3.shape)
print('---------------------------------')
print(Im4.shape)
Im4 = np.expand_dims(Im4, axis=0)
print(Im4.shape)
augmented_Im4 = augment_data(Im4, augementation_factor = 4)
print(augmented_Im4.shape)
print('---------------------------------')

images = np.vstack([augmented_Im0, augmented_Im1, augmented_Im2, augmented_Im3, augmented_Im4])

print(images.shape)
np.random.shuffle(images)
print(images.shape)

## Seperate label from Image
Images = images[:, :, :, 0:9]
print(Images.shape)
Labels = images[:, :, :, 9:10]
print(Labels.shape)

## Convert Label Channel to Int
print(Images.dtype)
print(Labels.dtype)
Labels = Labels.astype(int)
print(Labels.dtype)

## Write/Read Labels and Images Numpy To/From HDF5
Images_hf  = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\AllDrivers\Augmented\Images.h5', 'w')
print(Images.shape)
Images_hf.create_dataset('Images', data = Images)
Images_hf.close()

Labels_hf  = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\AllDrivers\Augmented\Labels.h5', 'w')
print(Labels.shape)
Labels_hf.create_dataset('Labels', data = Labels)
Labels_hf.close()

Images_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\AllDrivers\Augmented\Images.h5', 'r')
print(Images_hf.keys())
Images = np.array(Images_hf.get('Images'))
print(Images.shape)
print(np.amin(Images))
print(np.amax(Images))

Labels_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\AllDrivers\Augmented\Labels.h5', 'r')
print(Labels_hf.keys())
Labels = np.array(Labels_hf.get('Labels'))
print(Labels.shape)
print(np.amin(Labels))
print(np.amax(Labels))

## Convert Label to One Hot Vector
Labels = keras.utils.to_categorical(Labels, num_classes=2)
print(Labels.shape)

# class MeanIoU(tf.keras.metrics.MeanIoU):
#     def __call__(self, y_true, y_pred, sample_weight=None):
#         y_pred = tf.argmax(y_pred, axis=-1)
#         return super().__call__(y_true, y_pred, sample_weight=sample_weight)

# tf.metrics.MeanIoU
## Define IoU Metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        # y_pred_ = tf.cast(y_pred > t, tf.int32)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# ## Define Custom Loss Function
class_weights = np.array([1, 1])
weights = K.variable(class_weights)
def weighted_categorical_crossentropy(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis = -1, keepdims = True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calculate loss and weight loss
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss

## Set parameters
Img_Width    = 256
Img_Height   = 256
Img_Channels = 9
Num_Classes  = 2

## Define Inputs and Targets Dim
inputs  = Input((Img_Height, Img_Width, Img_Channels))
# print(inputs.shape)
targets = Input((Img_Height, Img_Width, Num_Classes))
# print(targets.shape)

## Model Architecture

c1 = Conv2D(16, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (inputs)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(64, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (p2)
c3 = Dropout(0.2)(c3)
c3 = Conv2D(64, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(128, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (p3)
c4 = Dropout(0.2)(c4)
c4 = Conv2D(128, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c4)
p4 = MaxPooling2D(pool_size = (2, 2))(c4)

c5 = Conv2D(256, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (p4)
c5 = Dropout(0.3)(c5)
c5 = Conv2D(256, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides = (2, 2), padding = 'same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides = (2, 2), padding = 'same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides = (2, 2), padding = 'same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides = (2, 2), padding = 'same') (c8)
u9 = concatenate([u9, c1], axis = 3)
c9 = Conv2D(16, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c9)

outputs = Conv2D(2, (1, 1), activation = 'softmax') (c9)
print(outputs.shape)
# print(outputs.shape)

# miou_metric = MeanIoU(2)

model = Model(inputs = [inputs], outputs = [outputs])
model.compile(optimizer='adam', loss=weighted_categorical_crossentropy,   metrics=[mean_iou])
model.summary()

##
model_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\UnetModel\model_epochs_05.h5', 'w')
model_hf.close()
earlystopper = EarlyStopping(patience = 5, verbose = 1)
checkpointer = ModelCheckpoint(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\UnetModel\model_epochs_05.h5', verbose = 1, save_best_only=True)
results      = model.fit(Images, Labels, validation_split = 0.10, batch_size = 8, epochs = 5, callbacks = [earlystopper, checkpointer])



# ###
model = load_model(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\UnetModel\model_epochs_05.h5',
    custom_objects={'mean_iou': mean_iou, 'weighted_categorical_crossentropy': weighted_categorical_crossentropy})
# Img1 = np.expand_dims(np.array(Image.open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Drivers\All_Images_Shuffle\_0027.tif')), axis=0)
# Img2 = np.expand_dims(np.array(Image.open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Drivers\All_Images_Shuffle\_0028.tif')), axis=0)
# Img3 = np.expand_dims(np.array(Image.open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Drivers\All_Images_Shuffle\_0023.tif')), axis=0)
# Img4 = np.expand_dims(np.array(Image.open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Drivers\All_Images_Shuffle\_0030.tif')), axis=0)
# Img5 = np.expand_dims(np.array(Image.open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Drivers\All_Images_Shuffle\_0031.tif')), axis=0)
# Img6 = np.expand_dims(np.array(Image.open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Drivers\All_Images_Shuffle\_0032.tif')), axis=0)
# Img7 = np.expand_dims(np.array(Image.open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Drivers\All_Images_Shuffle\_0033.tif')), axis=0)
# Img8 = np.expand_dims(np.array(Image.open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Drivers\All_Images_Shuffle\_0034.tif')), axis=0)
# Img = np.expand_dims(np.vstack([Img1, Img2, Img3, Img4, Img5, Img6, Img7, Img8]), axis=3)
# print(Img.shape)
predict_prob = model.predict(Images[0:5, :, :, :])
print(predict_prob.shape)
predict_class = predict_prob.argmax(axis = -1)
print(predict_class.shape)
#
# predict_prob = model.predict(Images[0:20, :, :, :])
# print(predict_prob.shape)
# predict_class = predict_prob.argmax(axis = -1)
# print(predict_class.shape)
#
# num = 9
# print(Images[0, :, :, :].shape)
imshow(np.reshape(Labels[0,:,:], (256,256)))
plt.show()
#
print(predict_class[0].shape)
imshow(predict_class[0])
plt.show()
# #
# res3 = list(set(i for j in np.array(np.reshape(predict_class[0, :, :], (256, 256))) for i in j))
# print(res3)

# model_path = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\UnetModel\model_epochs_05.h5'
# input_image_dir = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Drivers\Chips\All_Images'
# output_image_dir = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Predict_2004To2014\Chips_Predicted'
#
# for root, dirs, files in os.walk(input_image_dir):
#     if not files: continue
#
#     for f in files:
#         pth = os.path.join(root, f)
#         out_pth = os.path.join(output_image_dir, f.split('.')[0] + '.tif')
#         predict_prob = model.predict(f)
#         print(predict_prob.shape)
#         predict_class = predict_prob.argmax(axis=-1)
#         print(predict_class.shape)
#         print('saved result to ' + out_pth)