# ## Labels (Change_1994_2004) 2 : NAN + Exclusions
## Labels_1994_2004 (notebook --> 9)
import numpy as np
import gdal
import h5py
from PIL import Image
from skimage.io import imread, imshow
from keras.optimizers import Adam
from keras.layers import Activation
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

LC1994_dataset = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Data_Esfahanf\re1994.asc')
LC1994 = LC1994_dataset.ReadAsArray().astype(np.float32)
print(LC1994.shape, np.amin(LC1994), np.amax(LC1994))


LC2004_dataset = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Data_Esfahanf\re2004.asc')
LC2004 = LC2004_dataset.ReadAsArray().astype(np.float32)
print(LC2004.shape, np.amin(LC2004), np.amax(LC2004))

unique, counts = np.unique(LC1994, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

unique, counts = np.unique(LC2004, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

for i in range(LC1994.shape[0]):
    for j in range(LC1994.shape[1]):
        if LC1994[i][j] == -9999:
            LC1994[i][j] = 2
        elif LC1994[i][j] == 1:
            LC1994[i][j] = 2
        elif LC1994[i][j] == 2:
            LC1994[i][j] = 4
        elif LC1994[i][j] == 3:
            LC1994[i][j] = 4
        elif LC1994[i][j] == 4:
            LC1994[i][j] = 2
        elif LC1994[i][j] == 5:
            LC1994[i][j] = 2

for i in range(LC2004.shape[0]):
    for j in range(LC2004.shape[1]):
        if LC2004[i][j] == -9999:
            LC2004[i][j] = 10
        elif LC2004[i][j] == 1:
            LC2004[i][j] = 30
        elif LC2004[i][j] == 2:
            LC2004[i][j] = 20
        elif LC2004[i][j] == 3:
            LC2004[i][j] = 20
        elif LC2004[i][j] == 4:
            LC2004[i][j] = 10
        elif LC2004[i][j] == 5:
            LC2004[i][j] = 10

unique, counts = np.unique(LC1994, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

unique, counts = np.unique(LC2004, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

Change_1994_2004 = LC2004 + LC1994
unique, counts = np.unique(Change_1994_2004, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

for i in range(Change_1994_2004.shape[0]):
    for j in range(Change_1994_2004.shape[1]):
        if Change_1994_2004[i][j] == 12:
            Change_1994_2004[i][j] = 2
        elif Change_1994_2004[i][j] == 22:
            Change_1994_2004[i][j] = 2
        elif Change_1994_2004[i][j] == 32:
            Change_1994_2004[i][j] = 2
        elif Change_1994_2004[i][j] == 14:
            Change_1994_2004[i][j] = 2

        elif Change_1994_2004[i][j] == 24:
            Change_1994_2004[i][j] = 0
        elif Change_1994_2004[i][j] == 34:
            Change_1994_2004[i][j] = 1


unique, counts = np.unique(Change_1994_2004, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

def save_image(image_data, path, geo_transform, projection):
    driver = gdal.GetDriverByName('GTiff')
    # Set Info of Image
    height, width = image_data.shape
    dataset = driver.Create(path, width, height, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(image_data[:, :])
    dataset.GetRasterBand(1).SetNoDataValue(2)
    dataset.FlushCache()

Change_1994_2004_path = r'E:\Education\MSc\thesis\UrbanExpansion\Final_Isfahan\Labels_1994_2004(0,1,2).tif'
save_image(Change_1994_2004, Change_1994_2004_path, LC1994_dataset.GetGeoTransform(), LC1994_dataset.GetProjection())

# ## Labels (Change_1994_2004) NAN : 2
## Labels_1994_2004 (notebook --> 9)
import numpy as np
import gdal
import h5py
from PIL import Image
from skimage.io import imread, imshow
from keras.optimizers import Adam
from keras.layers import Activation
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

LC1994_dataset = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Data_Esfahanf\re1994.asc')
LC1994 = LC1994_dataset.ReadAsArray().astype(np.float32)
print(LC1994.shape, np.amin(LC1994), np.amax(LC1994))


LC2004_dataset = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Data_Esfahanf\re2004.asc')
LC2004 = LC2004_dataset.ReadAsArray().astype(np.float32)
print(LC2004.shape, np.amin(LC2004), np.amax(LC2004))

unique, counts = np.unique(LC1994, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

unique, counts = np.unique(LC2004, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

for i in range(LC1994.shape[0]):
    for j in range(LC1994.shape[1]):
        if LC1994[i][j] == -9999:
            LC1994[i][j] = 2
        elif LC1994[i][j] == 1:
            LC1994[i][j] = 4
        elif LC1994[i][j] == 2:
            LC1994[i][j] = 6
        elif LC1994[i][j] == 3:
            LC1994[i][j] = 6
        elif LC1994[i][j] == 4:
            LC1994[i][j] = 6
        elif LC1994[i][j] == 5:
            LC1994[i][j] = 6

for i in range(LC2004.shape[0]):
    for j in range(LC2004.shape[1]):
        if LC2004[i][j] == -9999:
            LC2004[i][j] = 10
        elif LC2004[i][j] == 1:
            LC2004[i][j] = 20
        elif LC2004[i][j] == 2:
            LC2004[i][j] = 30
        elif LC2004[i][j] == 3:
            LC2004[i][j] = 30
        elif LC2004[i][j] == 4:
            LC2004[i][j] = 30
        elif LC2004[i][j] == 5:
            LC2004[i][j] = 30

unique, counts = np.unique(LC1994, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

unique, counts = np.unique(LC2004, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

Change_1994_2004 = LC2004 + LC1994
unique, counts = np.unique(Change_1994_2004, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

for i in range(Change_1994_2004.shape[0]):
    for j in range(Change_1994_2004.shape[1]):
        if Change_1994_2004[i][j] == 12:
            Change_1994_2004[i][j] = 2
        elif Change_1994_2004[i][j] == 22:
            Change_1994_2004[i][j] = 2
        elif Change_1994_2004[i][j] == 32:
            Change_1994_2004[i][j] = 2
        elif Change_1994_2004[i][j] == 14:
            Change_1994_2004[i][j] = 2

        elif Change_1994_2004[i][j] == 24:
            Change_1994_2004[i][j] = 0
        elif Change_1994_2004[i][j] == 34:
            Change_1994_2004[i][j] = 0

        elif Change_1994_2004[i][j] == 16:
            Change_1994_2004[i][j] = 2

        elif Change_1994_2004[i][j] == 26:
            Change_1994_2004[i][j] = 1

        elif Change_1994_2004[i][j] == 36:
            Change_1994_2004[i][j] = 0

unique, counts = np.unique(Change_1994_2004, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

def save_image(image_data, path, geo_transform, projection):
    driver = gdal.GetDriverByName('GTiff')
    # Set Info of Image
    height, width = image_data.shape
    dataset = driver.Create(path, width, height, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(image_data[:, :])
    dataset.GetRasterBand(1).SetNoDataValue(2)
    dataset.FlushCache()

Change_1994_2004_path = r'E:\Education\MSc\thesis\UrbanExpansion\Final_Isfahan\Labels_1994_2004(0,1,2)_ExclusionsAsNochange.tif'
save_image(Change_1994_2004, Change_1994_2004_path, LC1994_dataset.GetGeoTransform(), LC1994_dataset.GetProjection())

## Make Change and Mask dataset for 2004 To 2014
import numpy as np
import gdal
import h5py
from PIL import Image
from skimage.io import imread, imshow
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

LC2004_dataset = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Data_Esfahanf\re2004.asc')
LC2004 = LC2004_dataset.ReadAsArray().astype(np.float32)
print(LC2004.shape, np.amin(LC2004), np.amax(LC2004))

LC2014_dataset = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Data_Esfahanf\re2014.asc')
LC2014 = LC2014_dataset.ReadAsArray().astype(np.float32)
print(LC2014.shape, np.amin(LC2014), np.amax(LC2014))

unique, counts = np.unique(LC2004, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

unique, counts = np.unique(LC2014, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

for i in range(LC2004.shape[0]):
    for j in range(LC2004.shape[1]):
        if LC2004[i][j] == -9999:
            LC2004[i][j] = 2
        elif LC2004[i][j] == 1:
            LC2004[i][j] = 2
        elif LC2004[i][j] == 2:
            LC2004[i][j] = 4
        elif LC2004[i][j] == 3:
            LC2004[i][j] = 4
        elif LC2004[i][j] == 4:
            LC2004[i][j] = 2
        elif LC2004[i][j] == 5:
            LC2004[i][j] = 2

for i in range(LC2014.shape[0]):
    for j in range(LC2014.shape[1]):
        if LC2014[i][j] == -9999:
            LC2014[i][j] = 10
        elif LC2014[i][j] == 1:
            LC2014[i][j] = 30
        elif LC2014[i][j] == 2:
            LC2014[i][j] = 20
        elif LC2014[i][j] == 3:
            LC2014[i][j] = 20
        elif LC2014[i][j] == 4:
            LC2014[i][j] = 10
        elif LC2014[i][j] == 5:
            LC2014[i][j] = 10

unique, counts = np.unique(LC2004, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

unique, counts = np.unique(LC2014, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

Change_2004_2014 = LC2014 + LC2004
unique, counts = np.unique(Change_2004_2014, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

for i in range(Change_2004_2014.shape[0]):
    for j in range(Change_2004_2014.shape[1]):
        if Change_2004_2014[i][j] == 12:
            Change_2004_2014[i][j] = 2
        elif Change_2004_2014[i][j] == 22:
            Change_2004_2014[i][j] = 2
        elif Change_2004_2014[i][j] == 32:
            Change_2004_2014[i][j] = 2
        elif Change_2004_2014[i][j] == 14:
            Change_2004_2014[i][j] = 2

        elif Change_2004_2014[i][j] == 24:
            Change_2004_2014[i][j] = 0
        elif Change_2004_2014[i][j] == 34:
            Change_2004_2014[i][j] = 1

unique, counts = np.unique(Change_2004_2014, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

Mask_2004_2014 = np.copy(Change_2004_2014)
for i in range(Mask_2004_2014.shape[0]):
    for j in range(Mask_2004_2014.shape[1]):
        if Mask_2004_2014[i][j] == 0:
            Mask_2004_2014[i][j] = 1
        elif Mask_2004_2014[i][j] == 1:
            Mask_2004_2014[i][j] = 1
        elif Mask_2004_2014[i][j] == 2:
            Mask_2004_2014[i][j] = 0

unique, counts = np.unique(Mask_2004_2014, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

Change_2004_2014 = Change_2004_2014[:, 0:1024]
Mask_2004_2014 = Mask_2004_2014[:, 0:1024]
print(Change_2004_2014.shape, np.amin(Change_2004_2014), np.amax(Change_2004_2014))
print(Mask_2004_2014.shape, np.amin(Mask_2004_2014), np.amax(Mask_2004_2014))

unique, counts = np.unique(Change_2004_2014, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

unique, counts = np.unique(Mask_2004_2014, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

def save_image(image_data, path, geo_transform, projection):
    driver = gdal.GetDriverByName('GTiff')
    # Set Info of Image
    height, width = image_data.shape
    dataset = driver.Create(path, width, height, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(image_data[:, :])
    dataset.GetRasterBand(1).SetNoDataValue(2)
    dataset.FlushCache()

Change_2004_2014_path = r'E:\Education\MSc\thesis\UrbanExpansion\Final_Isfahan\Change_2004_2014.tif'
save_image(Change_2004_2014, Change_2004_2014_path, LC2004_dataset.GetGeoTransform(), LC2004_dataset.GetProjection())

def save_image(image_data, path, geo_transform, projection):
    driver = gdal.GetDriverByName('GTiff')
    # Set Info of Image
    height, width = image_data.shape
    dataset = driver.Create(path, width, height, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(image_data[:, :])
    dataset.GetRasterBand(1).SetNoDataValue(0)
    dataset.FlushCache()

Mask_2004_2014_path = r'E:\Education\MSc\thesis\UrbanExpansion\Final_Isfahan\Mask_2004_2014.tif'
save_image(Mask_2004_2014, Mask_2004_2014_path, LC2004_dataset.GetGeoTransform(), LC2004_dataset.GetProjection())

## Check data
Change_2004_2014_dataset = gdal.Open(Change_2004_2014_path)
Change_2004_2014 = Change_2004_2014_dataset.ReadAsArray().astype(np.float32)
print(Change_2004_2014.shape, np.amin(Change_2004_2014), np.amax(Change_2004_2014))
unique, counts = np.unique(Change_2004_2014, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

Mask_2004_2014_dataset = gdal.Open(Mask_2004_2014_path)
Mask_2004_2014 = Mask_2004_2014_dataset.ReadAsArray().astype(np.float32)
print(Mask_2004_2014.shape, np.amin(Mask_2004_2014), np.amax(Mask_2004_2014))
unique, counts = np.unique(Mask_2004_2014, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

## Prediction:
import os
import gdal
import numpy as np
from tqdm import tqdm
import h5py
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
from matplotlib.pyplot import imshow
from PIL import Image
from sklearn.preprocessing import minmax_scale
from keras.metrics import AUC


## Define Metrics and Losses

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

class_weights = np.array([1, 1])
weights = K.variable(class_weights)
def weighted_categorical_crossentropy(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calculate loss and weight loss
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def focal_loss(target, output, gamma=2):
    output /= K.sum(output, axis=-1, keepdims=True)
    eps = K.epsilon()
    output = K.clip(output, eps, 1. - eps)
    return -K.sum(K.pow(1. - output, gamma) * target * K.log(output), axis=-1)

def softmax_cross_entropy_with_logits_loss(y_true, y_pred):
  loss=tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
  return loss

def weighted_binary_crossentropy( y_true, y_pred, weight=6.14473547435114 ) :
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * weight + (1 - y_true) * K.log(1 - y_pred))
    return K.mean( logloss, axis=-1)

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


## Save GeoTIFF Image
def save_image(image_data, path, geo_transform, projection):
    driver = gdal.GetDriverByName('GTiff')

    # Set Info of Image
    height, width = image_data.shape
    dataset = driver.Create(path, width, height, 1, gdal.GDT_Float64 )
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(image_data[:,:])

    dataset.GetRasterBand(1).SetNoDataValue(2)

    dataset.FlushCache()

## input_Image : 2004 Drivers
DEM_dataset = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Ascii2\dem.asc')
DEM = DEM_dataset.GetRasterBand(1).ReadAsArray().astype(np.float32)
print(DEM.shape)
print(np.amin(DEM))
print(np.amax(DEM))
print('---------------------------------')
Distance_From_Built = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Ascii2\dist_built.asc')
Distance_From_Built = Distance_From_Built.GetRasterBand(1).ReadAsArray().astype(np.float32)
print(Distance_From_Built.shape)
print(np.amin(Distance_From_Built))
print(np.amax(Distance_From_Built))
print('---------------------------------')
Distance_From_Crop = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Ascii2\dist_crop.asc')
Distance_From_Crop = Distance_From_Crop.GetRasterBand(1).ReadAsArray().astype(np.float32)
print(Distance_From_Crop.shape)
print(np.amin(Distance_From_Crop))
print(np.amax(Distance_From_Crop))
print('---------------------------------')
Distance_From_Marsh = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Ascii2\dist_marsh.asc')
Distance_From_Marsh = Distance_From_Marsh.GetRasterBand(1).ReadAsArray().astype(np.float32)
print(Distance_From_Marsh.shape)
print(np.amin(Distance_From_Marsh))
print(np.amax(Distance_From_Marsh))
print('---------------------------------')
Distance_From_Open = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Ascii2\dist_open.asc')
Distance_From_Open = Distance_From_Open.GetRasterBand(1).ReadAsArray().astype(np.float32)
print(Distance_From_Open.shape)
print(np.amin(Distance_From_Open))
print(np.amax(Distance_From_Open))
print('---------------------------------')
Distance_From_Roads = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Ascii2\dist_road.asc')
Distance_From_Roads = Distance_From_Roads.GetRasterBand(1).ReadAsArray().astype(np.float32)
print(Distance_From_Roads.shape)
print(np.amin(Distance_From_Roads))
print(np.amax(Distance_From_Roads))
print('---------------------------------')
Easting = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Ascii2\easting.asc')
Easting = Easting.GetRasterBand(1).ReadAsArray().astype(np.float32)
print(Easting.shape)
print(np.amin(Easting))
print(np.amax(Easting))
print('---------------------------------')
Northing = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Isfahan\Ascii2\northing.asc')
Northing = Northing.GetRasterBand(1).ReadAsArray().astype(np.float32)
print(Northing.shape)
print(np.amin(Northing))
print(np.amax(Northing))
print('---------------------------------')
##
input_Image = np.stack([DEM[:, 0:1024], Distance_From_Built[:, 0:1024],Distance_From_Crop[:, 0:1024], Distance_From_Marsh[:, 0:1024],
                            Distance_From_Open[:, 0:1024], Distance_From_Roads[:, 0:1024], Easting[:, 0:1024], Northing[:, 0:1024]], axis=2)
print(input_Image.shape, np.amax(input_Image), np.amin(input_Image))
print('---------------------------------')

## Local MinMax([0,1]) --> (each image)
for i in range(input_Image.shape[-1]):
    input_Image_min = np.amin(input_Image[:, :, i])
    input_Image_max = np.amax(input_Image[:, :, i])
    input_Image[:, :, i] = (input_Image[:, :, i] - input_Image_min) / (input_Image_max - input_Image_min)
    input_Image[:, :, i] = np.round_(input_Image[:, :, i], decimals=4)
print(input_Image.shape, np.amax(input_Image), np.amin(input_Image))
print('---------------------------------')
#
## Binary
def Prediction (model_path, Predicted_image_path) :
    model = load_model(model_path, custom_objects={'weighted_categorical_crossentropy': weighted_categorical_crossentropy, 'mean_iou': mean_iou})

    model_input_height, model_input_width, model_input_channels = model.layers[0].input_shape[1:4]
    print(model_input_height, model_input_width, model_input_channels)
    model_output_height, model_output_width, model_output_channels = model.layers[len(model.layers) - 1].output_shape[
                                                                     1:4]
    print(model_output_height, model_output_width, model_output_channels)

    h, w, n = input_Image.shape
    print(h, w, n)

    n_rows = int(h / model_output_height)
    print(n_rows)
    n_cols = int(w / model_output_width)
    print(n_cols)
    batch_size = (n_rows * n_cols)
    print(batch_size)
    mb_array = np.zeros((batch_size, model_input_width, model_input_height, model_input_channels))
    print(mb_array.shape)

    number_of_classes = model_output_channels
    pred_lc_image = np.zeros((h, w, number_of_classes))
    print(pred_lc_image.shape)

    irows, icols = [], []
    ibatch = 0
    i = 0
    for row_idx in (range(n_rows)):
        for col_idx in (range(n_cols)):
            i += 1
            subimage = input_Image[row_idx * model_output_height:row_idx * model_output_height + model_input_height,
                       col_idx * model_output_width:col_idx * model_output_width + model_input_width, :]
            print(subimage.shape)
            print(i)
            print(np.amin(subimage), np.amax(subimage))

            if (subimage.shape == model.layers[0].input_shape[1:4]):

                mb_array[ibatch] = subimage
                ibatch += 1
                irows.append((row_idx * model_output_height, row_idx * model_output_height + model_input_height))
                icols.append((col_idx * model_output_width, col_idx * model_output_width + model_input_width))
                print(irows, icols)

                if (ibatch) == batch_size:

                    outputs = model.predict(mb_array)
                    for i in range(batch_size):
                        r0, r1 = irows[i]
                        c0, c1 = icols[i]

                        pred_lc_image[r0:r1, c0:c1, :] = outputs[
                            i]  # pred_lc_image.shape = (1024, 3584, 1) while outputs.shape = (56, 256, 256, 1)

                    ibatch = 0
                    irows, icols = [], []

    Predicted_image = np.array(pred_lc_image[:, :, -1])
    print('Predicted_image.shape:', Predicted_image.shape, ', Predicted_image.amin:', np.amin(Predicted_image),
          ', Predicted_image.amax:', np.amax(Predicted_image), ', Predicted_image.mean:', np.mean(Predicted_image))
    imshow(Predicted_image)
    save_image(Predicted_image, Predicted_image_path, DEM_dataset.GetGeoTransform(),
               DEM_dataset.GetProjection())
    print('Saved Predicted image in : ', Predicted_image_path)

# ## 3 Classes
# ## Define Custom Loss Function
# class_weights = np.array([1, 1, 1])
# weights = K.variable(class_weights)
# def weighted_categorical_crossentropy(y_true, y_pred):
#     # scale predictions so that the class probas of each sample sum to 1
#     y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#     # clip to prevent NaN's and Inf's
#     y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
#     # calculate loss and weight loss
#     loss = y_true * K.log(y_pred) * weights
#     loss = -K.sum(loss, -1)
#     return loss
#
# def Prediction (model_path, Predicted_image_path) :  ## For 0, 1, 2
#     model = load_model(model_path, custom_objects={'weighted_categorical_crossentropy': weighted_categorical_crossentropy, 'mean_iou': mean_iou})
#
#     model_input_height, model_input_width, model_input_channels = model.layers[0].input_shape[1:4]
#     print(model_input_height, model_input_width, model_input_channels)
#     model_output_height, model_output_width, model_output_channels = model.layers[len(model.layers) - 1].output_shape[
#                                                                      1:4]
#     print(model_output_height, model_output_width, model_output_channels)
#
#     h, w, n = input_Image.shape
#     print(h, w, n)
#
#     n_rows = int(h / model_output_height)
#     print(n_rows)
#     n_cols = int(w / model_output_width)
#     print(n_cols)
#     batch_size = (n_rows * n_cols)
#     print(batch_size)
#     mb_array = np.zeros((batch_size, model_input_width, model_input_height, model_input_channels))
#     print(mb_array.shape)
#
#     number_of_classes = model_output_channels
#     pred_lc_image = np.zeros((h, w, number_of_classes))
#     print(pred_lc_image.shape)
#
#     irows, icols = [], []
#     ibatch = 0
#     i = 0
#     for row_idx in (range(n_rows)):
#         for col_idx in (range(n_cols)):
#             i += 1
#             subimage = input_Image[row_idx * model_output_height:row_idx * model_output_height + model_input_height,
#                        col_idx * model_output_width:col_idx * model_output_width + model_input_width, :]
#             print(subimage.shape)
#             print(i)
#             print(np.amin(subimage), np.amax(subimage))
#
#             if (subimage.shape == model.layers[0].input_shape[1:4]):
#
#                 mb_array[ibatch] = subimage
#                 ibatch += 1
#                 irows.append((row_idx * model_output_height, row_idx * model_output_height + model_input_height))
#                 icols.append((col_idx * model_output_width, col_idx * model_output_width + model_input_width))
#                 print(irows, icols)
#
#                 if (ibatch) == batch_size:
#
#                     outputs = model.predict(mb_array)
#                     for i in range(batch_size):
#                         r0, r1 = irows[i]
#                         c0, c1 = icols[i]
#
#                         pred_lc_image[r0:r1, c0:c1, :] = outputs[
#                             i]  # pred_lc_image.shape = (1024, 3584, 1) while outputs.shape = (56, 256, 256, 1)
#
#                     ibatch = 0
#                     irows, icols = [], []
#
#     Predicted_image = np.array(pred_lc_image[:, :, 1])
#     print('Predicted_image.shape:', Predicted_image.shape, ', Predicted_image.amin:', np.amin(Predicted_image),
#           ', Predicted_image.amax:', np.amax(Predicted_image), ', Predicted_image.mean:', np.mean(Predicted_image))
#     imshow(Predicted_image)
#     save_image(Predicted_image, Predicted_image_path, Altitude_dataset.GetGeoTransform(),
#                Altitude_dataset.GetProjection())
#     print('Saved Predicted image in : ', Predicted_image_path)


## Read model and make prediction
Predicted_image_path = r'E:\Education\MSc\thesis\UrbanExpansion\Final_Isfahan\Training\Predicted_Model_256_1.tif'
model_path = r'E:\Education\MSc\thesis\UrbanExpansion\Final_Isfahan\Training\Model_256.1.h5'

Prediction (model_path, Predicted_image_path)

## TopDown allocation
import numpy as np
import gdal
import h5py
from skimage.io import imread, imshow
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

## Change and Mask 2004_2014
Probability_map_dataset = gdal.Open(Predicted_image_path)
Probability_map = Probability_map_dataset.ReadAsArray().astype(np.float32)
print(Probability_map.shape, np.amin(Probability_map), np.amax(Probability_map))

Change_2004_2014_dataset = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Final_Isfahan\Change_2004_2014.tif')
Change_2004_2014 = Change_2004_2014_dataset.ReadAsArray().astype(np.float32)
print(Change_2004_2014.shape, np.amin(Change_2004_2014), np.amax(Change_2004_2014))
unique, counts = np.unique(Change_2004_2014, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

Mask_2004_2014_dataset = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Final_Isfahan\Mask_2004_2014.tif')
Mask_2004_2014 = Mask_2004_2014_dataset.ReadAsArray().astype(np.float32)
print(Mask_2004_2014.shape, np.amin(Mask_2004_2014), np.amax(Mask_2004_2014))
unique, counts = np.unique(Mask_2004_2014, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

# Exclude exclusions from probability map
for i in range(Mask_2004_2014.shape[0]):
    for j in range(Mask_2004_2014.shape[1]):
        if Mask_2004_2014[i][j] == 0: ## 0 : NAN
            Probability_map[i][j] = -9999

unique, counts = np.unique(Probability_map, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

# Calculate the number of pixels for allocate (Change_2004_2014 == 1)
unique, counts = np.unique(Change_2004_2014, return_counts=True)
freq_sum_Image = np.asarray((unique, counts)).T
print(freq_sum_Image)

n_allocate = freq_sum_Image [1][1]
# print(n_allocate)

# Sort Probability_map
# print(Probability_map.shape)
reshaped_Probability_map = np.reshape(Probability_map, (Probability_map.shape[0] * Probability_map.shape[1]))
# print(reshaped_Probability_map.shape)

Sorted_reshaped_Probability_map = (-np.sort(-reshaped_Probability_map))
# print(Sorted_reshaped_Probability_map)
# print(Sorted_reshaped_Probability_map.shape)

Change_pixels = Sorted_reshaped_Probability_map [0:int(n_allocate)]
# print(Change_pixels.shape)

Change_map = np.copy(Probability_map)
# print(Change_map.shape)
for i in range(Change_map.shape[0]):
    for j in range(Change_map.shape[1]):
        if Change_map[i][j] in Change_pixels :
            Change_map[i][j] = 1
        elif Change_map[i][j] != -9999:
            Change_map[i][j] = 0
        elif Change_map[i][j] == -9999:
            Change_map[i][j] = 2

print(Change_map.shape)
unique, counts = np.unique(Change_map, return_counts=True)
freq_Change_map = np.asarray((unique, counts)).T
print(freq_Change_map)
#
print(freq_Change_map [1][1] == n_allocate)

# #Subplot
# plt.figure(figsize = (50,50))
# plt.imshow(np.r_[np.c_[Change_2004_2014], np.c_[Change_map]])


## Accuracy Assesment
from sklearn.metrics import confusion_matrix

Y_actual = np.reshape (Change_2004_2014, Change_2004_2014.shape[0] * Change_2004_2014.shape[1])
Y_Predicted = np.reshape (Change_map, Change_map.shape[0] * Change_map.shape[1])

# unique, counts = np.unique(Y_actual, return_counts=True)
# freq_Y_actual = np.asarray((unique, counts)).T
# print(freq_Y_actual)
# unique, counts = np.unique(Y_Predicted, return_counts=True)
# freq_Y_Predicted = np.asarray((unique, counts)).T
# print(freq_Y_Predicted)

# print(Y_actual.shape, Y_Predicted.shape)

Y_actual = np.delete (Y_actual, np.where(Y_actual==2))
Y_Predicted = np.delete (Y_Predicted, np.where(Y_Predicted==2))

# unique, counts = np.unique(Y_actual, return_counts=True)
# freq_Y_actual = np.asarray((unique, counts)).T
# print(freq_Y_actual)
# unique, counts = np.unique(Y_Predicted, return_counts=True)
# freq_Y_Predicted = np.asarray((unique, counts)).T
# print(freq_Y_Predicted)

print(Y_actual.shape, Y_Predicted.shape)

Conf_Matrix = confusion_matrix(Y_actual, Y_Predicted)
# print(Conf_Matrix)

TN = Conf_Matrix [0][0]
FN = Conf_Matrix [1][0]
FP = Conf_Matrix [0][1]
TP = Conf_Matrix [1][1]

print( 'TN : ', TN, ', FN : ', FN, ', FP : ', FP, ', TP : ', TP)

# FOM (Figure Of Merit), PA, UA, PCM
FOM = TP / (FN + TP + FP)
PA = TP / (FN + TP)
UA = TP / (TP + FP)
PCM = (TP / (TP + FP)) * 100
print('FOM = ', FOM, ', PA = ', PA, ', UA = ', UA, ', PCM : ', PCM)
