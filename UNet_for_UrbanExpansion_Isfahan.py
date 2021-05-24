## Chip Size = 256  --> 2 Classes (NAN:2)
# ## CPU:
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf
import os
import numpy as np
import gdal
import h5py
import random
import tensorflow.keras as keras
from keras import Input
from keras.layers import UpSampling2D, Add
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model, load_model
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image
from skimage.io import imread, imshow
from keras.optimizers import Adam
from keras.layers import Activation
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

Labels = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Final_Isfahan\Labels_1994_2004(0,1,2)_ExclusionsAsNochange.tif')
Labels = Labels.GetRasterBand(1).ReadAsArray().astype(np.float32)
print(Labels.shape)
print(np.amin(Labels))
print(np.amax(Labels))
print('---------------------------------')
DEM = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Ascii\dem.asc')
DEM = DEM.GetRasterBand(1).ReadAsArray().astype(np.float32)
print(DEM.shape)
print(np.amin(DEM))
print(np.amax(DEM))
print('---------------------------------')
Distance_From_Built = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Ascii\dist_built.asc')
Distance_From_Built = Distance_From_Built.GetRasterBand(1).ReadAsArray().astype(np.float32)
print(Distance_From_Built.shape)
print(np.amin(Distance_From_Built))
print(np.amax(Distance_From_Built))
print('---------------------------------')
Distance_From_Crop = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Ascii\dist_crop.asc')
Distance_From_Crop = Distance_From_Crop.GetRasterBand(1).ReadAsArray().astype(np.float32)
print(Distance_From_Crop.shape)
print(np.amin(Distance_From_Crop))
print(np.amax(Distance_From_Crop))
print('---------------------------------')
Distance_From_Marsh = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Ascii\dist_marsh.asc')
Distance_From_Marsh = Distance_From_Marsh.GetRasterBand(1).ReadAsArray().astype(np.float32)
print(Distance_From_Marsh.shape)
print(np.amin(Distance_From_Marsh))
print(np.amax(Distance_From_Marsh))
print('---------------------------------')
Distance_From_Open = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Ascii\dist_open.asc')
Distance_From_Open = Distance_From_Open.GetRasterBand(1).ReadAsArray().astype(np.float32)
print(Distance_From_Open.shape)
print(np.amin(Distance_From_Open))
print(np.amax(Distance_From_Open))
print('---------------------------------')
Distance_From_Roads = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Ascii\dist_road.asc')
Distance_From_Roads = Distance_From_Roads.GetRasterBand(1).ReadAsArray().astype(np.float32)
print(Distance_From_Roads.shape)
print(np.amin(Distance_From_Roads))
print(np.amax(Distance_From_Roads))
print('---------------------------------')
Easting = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Ascii\easting.asc')
Easting = Easting.GetRasterBand(1).ReadAsArray().astype(np.float32)
print(Easting.shape)
print(np.amin(Easting))
print(np.amax(Easting))
print('---------------------------------')
Northing = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Ascii\northing.asc')
Northing = Northing.GetRasterBand(1).ReadAsArray().astype(np.float32)
print(Northing.shape)
print(np.amin(Northing))
print(np.amax(Northing))
print('---------------------------------')
##
Stacked_Drivers = np.stack([DEM[:, :], Distance_From_Built[:, :],Distance_From_Crop[:, :], Distance_From_Marsh[:, :],
                            Distance_From_Open[:, :], Distance_From_Roads[:, :], Easting[:, :], Northing[:,:], Labels[:, :]], axis=2)
print(Stacked_Drivers.shape, np.amin(Stacked_Drivers), np.amax(Stacked_Drivers))

## dataset : B
Stacked_Drivers_train1 = Stacked_Drivers[0:500, 500:]
Stacked_Drivers_train2 = Stacked_Drivers[500:, :]

Stacked_Drivers_test = Stacked_Drivers[0:500, 0:500]

print(Stacked_Drivers_train1.shape, np.amin(Stacked_Drivers_train1), np.amax(Stacked_Drivers_train1))
print(Stacked_Drivers_train2.shape, np.amin(Stacked_Drivers_train2), np.amax(Stacked_Drivers_train2))
print(Stacked_Drivers_test.shape, np.amin(Stacked_Drivers_test), np.amax(Stacked_Drivers_test))

files = 'train1'
globals()[str(files)] = []
for i in range(128, Stacked_Drivers_train1.shape[0] - 128, 32):
    for j in range(128, Stacked_Drivers_train1.shape[1] - 128, 32):
        if 1 in Stacked_Drivers_train1[i - 128: i + 128, j - 128: j + 128, -1]:
            if (2 not in Stacked_Drivers_train1[i - 128: i + 128, j - 128: j + 128, -1]) :  ##  Nan == -9999
                # unique, counts = np.unique(Stacked_Drivers_train1[i - 128: i + 128, j - 128: j + 128, -1], return_counts=True)
                # freq = np.asarray((unique, counts)).T
                # if (freq[np.where(unique == 1), 1] > 32*32/2):
                img = Stacked_Drivers_train1[i - 128: i + 128, j - 128: j + 128, :]
                globals()[str(files)].append(img)

files = 'train2'
globals()[str(files)] = []
for i in range(128, Stacked_Drivers_train2.shape[0] - 128, 32):
    for j in range(128, Stacked_Drivers_train2.shape[1] - 128, 32):
        if 1 in Stacked_Drivers_train2[i - 128: i + 128, j - 128: j + 128, -1]:
            if (2 not in Stacked_Drivers_train2[i - 128: i + 128, j - 128: j + 128, -1]) :  ## Exclusion & Nan == 2
                # unique, counts = np.unique(Stacked_Drivers_train2[i - 128: i + 128, j - 128: j + 128, -1], return_counts=True)
                # freq = np.asarray((unique, counts)).T
                # if (freq[np.where(unique == 1), 1] > 32*32/2):
                img = Stacked_Drivers_train2[i - 128: i + 128, j - 128: j + 128, :]
                Change_min_0 = np.amin(img[:, :, 0])
                Change_max_0 = np.amax(img[:, :, 0])
                Change_min = np.amin(img[:, :, 1])
                Change_max = np.amax(img[:, :, 1])
                Change_min_2 = np.amin(img[:, :, 2])
                Change_max_2 = np.amax(img[:, :, 2])
                Change_min_3 = np.amin(img[:, :, 3])
                Change_max_3 = np.amax(img[:, :, 3])
                Change_min_4 = np.amin(img[:, :, 4])
                Change_max_4 = np.amax(img[:, :, 4])
                Change_min_5 = np.amin(img[:, :, 5])
                Change_max_5 = np.amax(img[:, :, 5])
                Change_min_6 = np.amin(img[:, :, 6])
                Change_max_6 = np.amax(img[:, :, 6])
                if (Change_max_0 != Change_min_0 and Change_max != Change_min and Change_max_2 != Change_min_2 and
                        Change_max_3 != Change_min_3 and Change_max_4 != Change_min_4 and Change_max_5 != Change_min_5 and Change_max_6 != Change_min_6):
                    globals()[str(files)].append(img)

files = 'test'
globals()[str(files)] = []
for i in range(128, Stacked_Drivers_test.shape[0] - 128, 16):
    for j in range(128, Stacked_Drivers_test.shape[1] - 128, 16):
        if 1 in Stacked_Drivers_test[i - 128: i + 128, j - 128: j + 128, -1]:
            if (2 not in Stacked_Drivers_test[i - 128: i + 128, j - 128: j + 128, -1]) :  ## Exclusion & Nan == 2
                globals()[str(files)].append(img)

train1 = np.array(train1)
train2 = np.array(train2)

Change_Chips_train = np.concatenate([train1, train2], axis=0)
print(Change_Chips_train.shape, np.amin(Change_Chips_train), np.amax(Change_Chips_train))
Change_Chips_test = np.array(test)
print(Change_Chips_test.shape, np.amin(Change_Chips_test), np.amax(Change_Chips_test))

## Local MinMax([0,1]) --> (each image)
for i in range(Change_Chips_train.shape[0]):
    for j in range((Change_Chips_train.shape[-1]) - 1):
        Change_Chips_train_min = np.amin(Change_Chips_train[i, :, :, j])
        Change_Chips_train_max = np.amax(Change_Chips_train[i, :, :, j])
        Change_Chips_train[i, :, :, j] = (Change_Chips_train[i, :, :, j] - Change_Chips_train_min) / (Change_Chips_train_max - Change_Chips_train_min)
        Change_Chips_train[i, :, :, j] = np.round_(Change_Chips_train[i, :, :, j], decimals=4)
print(Change_Chips_train.shape, np.amax(Change_Chips_train), np.amin(Change_Chips_train))
print('---------------------------------')
## Local MinMax([0,1]) --> (each image)
for i in range(Change_Chips_test.shape[0]):
    for j in range((Change_Chips_test.shape[-1]) - 1):
        Change_Chips_test_min = np.amin(Change_Chips_test[i, :, :, j])
        Change_Chips_test_max = np.amax(Change_Chips_test[i, :, :, j])
        Change_Chips_test[i, :, :, j] = (Change_Chips_test[i, :, :, j] - Change_Chips_test_min) / (Change_Chips_test_max - Change_Chips_test_min)
        Change_Chips_test[i, :, :, j] = np.round_(Change_Chips_test[i, :, :, j], decimals=4)
print(Change_Chips_test.shape, np.amax(Change_Chips_test), np.amin(Change_Chips_test))
print('---------------------------------')

np.random.shuffle(Change_Chips_train)
np.random.shuffle(Change_Chips_test)
print(Change_Chips_train.shape, np.amax(Change_Chips_train), np.amin(Change_Chips_train))
print(Change_Chips_test.shape, np.amax(Change_Chips_test), np.amin(Change_Chips_test))

## Split data into Train and Test
X_train = Change_Chips_train[:, :, :, 0:8]
Y_train = Change_Chips_train[:, :, :, 8:9]

X_test = Change_Chips_test[:, :, :, 0:8]
Y_test = Change_Chips_test[:, :, :, 8:9]

##Convert Label Channel to Int
print(Y_train.dtype)
Y_train = Y_train.astype(int)
print(Y_train.dtype)

print(Y_test.dtype)
Y_test = Y_test.astype(int)
print(Y_test.dtype)

## Freq
unique, counts = np.unique(Y_train, return_counts=True)
freq_Train = np.asarray((unique, counts)).T
print(freq_Train)

unique, counts = np.unique(Y_test, return_counts=True)
freq_test = np.asarray((unique, counts)).T
print(freq_test)

print(X_train.shape, np.amin(X_train), np.amax(X_train))
print(Y_train.shape, np.amin(Y_train), np.amax(Y_train))
print(X_test.shape, np.amin(X_test), np.amax(X_test))
print(Y_test.shape, np.amin(Y_test), np.amax(Y_test))

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


## Set parameters
Img_Width = 256
Img_Height = 256
Img_Channels = 8
Num_Classes = 2

## Define Inputs and Targets Dim
inputs = Input((Img_Height, Img_Width, Img_Channels))
print(inputs.shape)
targets = Input((Img_Height, Img_Width, 1))
print(targets.shape)

#Unet
c1 = Conv2D(16, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (inputs)
c1 = Conv2D(16, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (p1)
c2 = Conv2D(32, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(64, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (p2)
c3 = Conv2D(64, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(128, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (p3)
c4 = Conv2D(128, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c4)
p4 = MaxPooling2D((2, 2))(c4)

c5 = Conv2D(256, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (p4)
c5 = Conv2D(256, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides = (2, 2), padding = 'same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (u6)
c6 = Conv2D(128, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides = (2, 2), padding = 'same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (u7)
c7 = Conv2D(64, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides = (2, 2), padding = 'same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (u8)
c8 = Conv2D(32, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides = (2, 2), padding = 'same') (c8)
u9 = concatenate([u9, c1], axis = 3)
c9 = Conv2D(16, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (u9)
c9 = Conv2D(16, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c9)

outputs = Conv2D(1, (1, 1), activation = 'sigmoid') (c9)
print(outputs.shape)

model = Model(inputs = [inputs], outputs = [outputs])
model.compile(optimizer='Adam', loss='binary_crossentropy',  metrics=[mean_iou])
model.summary()
model_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Final_Isfahan\Training\Model_256.0.h5', 'w')
model_hf.close()
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint(r'E:\Education\MSc\thesis\UrbanExpansion\Final_Isfahan\Training\Model_256.0.h5', verbose=1, save_best_only=True)
history = model.fit(X_train,Y_train, validation_data=(X_test,Y_test),batch_size=16,epochs=300, callbacks=[earlystopper, checkpointer])

tf.keras.backend.clear_session()
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


## Set parameters
Img_Width = 256
Img_Height = 256
Img_Channels = 8
Num_Classes = 2

## Define Inputs and Targets Dim
inputs = Input((Img_Height, Img_Width, Img_Channels))
print(inputs.shape)
targets = Input((Img_Height, Img_Width, 1))
print(targets.shape)

from keras.layers import SpatialDropout2D
#Unet
c1 = Conv2D(16, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (inputs)
c1 = SpatialDropout2D(0.3)(c1)
c1 = Conv2D(16, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (p1)
c2 = SpatialDropout2D(0.5)(c2)
c2 = Conv2D(32, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(64, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (p2)
c3 = SpatialDropout2D(0.5)(c3)
c3 = Conv2D(64, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(128, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (p3)
c4 = SpatialDropout2D(0.5)(c4)
c4 = Conv2D(128, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c4)
p4 = MaxPooling2D((2, 2))(c4)

c5 = Conv2D(256, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (p4)
c5 = SpatialDropout2D(0.5)(c5)
c5 = Conv2D(256, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides = (2, 2), padding = 'same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (u6)
c6 = SpatialDropout2D(0.5) (c6)
c6 = Conv2D(128, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides = (2, 2), padding = 'same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (u7)
c7 = SpatialDropout2D(0.5) (c7)
c7 = Conv2D(64, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides = (2, 2), padding = 'same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (u8)
c8 = SpatialDropout2D(0.5) (c8)
c8 = Conv2D(32, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides = (2, 2), padding = 'same') (c8)
u9 = concatenate([u9, c1], axis = 3)
c9 = Conv2D(16, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (u9)
c9 = SpatialDropout2D(0.5) (c9)
c9 = Conv2D(16, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c9)

outputs = Conv2D(1, (1, 1), activation = 'sigmoid') (c9)
print(outputs.shape)

model = Model(inputs = [inputs], outputs = [outputs])
model.compile(optimizer='Adam', loss='binary_crossentropy',  metrics=[mean_iou])
model.summary()
model_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Final_Isfahan\Training\Model_256.1.h5', 'w')
model_hf.close()
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint(r'E:\Education\MSc\thesis\UrbanExpansion\Final_Isfahan\Training\Model_256.1.h5', verbose=1, save_best_only=True)
history = model.fit(X_train,Y_train, validation_data=(X_test,Y_test),batch_size=16,epochs=300, callbacks=[earlystopper, checkpointer])

tf.keras.backend.clear_session()
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

## Set parameters
Img_Width = 256
Img_Height = 256
Img_Channels = 8
Num_Classes = 2

## Define Inputs and Targets Dim
inputs = Input((Img_Height, Img_Width, Img_Channels))
print(inputs.shape)
targets = Input((Img_Height, Img_Width, 1))
print(targets.shape)

from keras.layers import SpatialDropout2D
#Unet
c1 = Conv2D(32, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (inputs)
c1 = SpatialDropout2D(0.3)(c1)
c1 = Conv2D(32, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(64, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (p1)
c2 = SpatialDropout2D(0.5)(c2)
c2 = Conv2D(64, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(128, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (p2)
c3 = SpatialDropout2D(0.5)(c3)
c3 = Conv2D(128, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(256, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (p3)
c4 = SpatialDropout2D(0.5)(c4)
c4 = Conv2D(256, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c4)
p4 = MaxPooling2D((2, 2))(c4)

c5 = Conv2D(512, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (p4)
c5 = SpatialDropout2D(0.5)(c5)
c5 = Conv2D(512, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c5)

u6 = Conv2DTranspose(256, (2, 2), strides = (2, 2), padding = 'same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(256, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (u6)
c6 = SpatialDropout2D(0.5) (c6)
c6 = Conv2D(256, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c6)

u7 = Conv2DTranspose(128, (2, 2), strides = (2, 2), padding = 'same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(128, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (u7)
c7 = SpatialDropout2D(0.5) (c7)
c7 = Conv2D(128, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c7)

u8 = Conv2DTranspose(64, (2, 2), strides = (2, 2), padding = 'same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(64, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (u8)
c8 = SpatialDropout2D(0.5) (c8)
c8 = Conv2D(64, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c8)

u9 = Conv2DTranspose(32, (2, 2), strides = (2, 2), padding = 'same') (c8)
u9 = concatenate([u9, c1], axis = 3)
c9 = Conv2D(32, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (u9)
c9 = SpatialDropout2D(0.5) (c9)
c9 = Conv2D(32, (3, 3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (c9)

outputs = Conv2D(1, (1, 1), activation = 'sigmoid') (c9)
print(outputs.shape)

model = Model(inputs = [inputs], outputs = [outputs])
model.compile(optimizer='Adam', loss='binary_crossentropy',  metrics=[mean_iou])
model.summary()
model_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Final_Isfahan\Training\Model_256.2.h5', 'w')
model_hf.close()
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint(r'E:\Education\MSc\thesis\UrbanExpansion\Final_Isfahan\Training\Model_256.2.h5', verbose=1, save_best_only=True)
history = model.fit(X_train,Y_train, validation_data=(X_test,Y_test),batch_size=16,epochs=300, callbacks=[earlystopper, checkpointer])
