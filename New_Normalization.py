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
import gdal
from PIL import Image
from skimage.io import imread, imshow, imread_collection, concatenate_images
# from miou import MeanIoU
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow,imread
from PIL import Image
import matplotlib.pyplot as plt
import tifffile
from sklearn.preprocessing import MinMaxScaler
import skimage.io as io
import rasterio as rio

########Test
# All_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\All_Images.h5', 'r')
# All = np.array(All_hf.get('All_Images')).astype(np.float32)
# print(All.shape)

## Chip Craeting
# Img = Image.open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\Data\slope.tif')
# out_path = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\Chips_128x128\Slope\Images'
#
# Image_width, Image_height = Img.size
# width, height = 128, 128
# stride_x, stride_y = 32, 32
# frame_num = 1
#
# for rows in range(0, Image_height, stride_y):
#     for columns in range(0, Image_width, stride_x):
#         Img_crop = Img.crop((rows, columns, rows + height, columns + width))
#         save_to = os.path.join(out_path, "_{:03}.tif")
#         Img_crop.save(save_to.format(frame_num))
#         frame_num += 1

# Visualize sample driver image (chip)
# gtif = gdal.Open(
#     r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Dr.Shafizadeh\Ascii\Tiff\Drivers\Chips\Dist_Marsh\Dist_Marsh011.tif')
# print(type(
#     r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Dr.Shafizadeh\Ascii\Tiff\Drivers\Chips\Dist_Marsh\Dist_Marsh011.tif'))
# print(gtif.GetMetadata())
# print("[ RASTER BAND COUNT ]: ", gtif.RasterCount)
# mask = gtif.GetRasterBand(1)
# print(type(mask))
# print(mask.GetStatistics(True, True))
# print(mask.GetStatistics(True, True)[1])
# mask = np.array(gtif.GetRasterBand(1).ReadAsArray())
# print(type(mask))
# print(mask.shape)
# mask = np.expand_dims(mask, axis=0)
# print(mask.shape)
#
# f = plt.figure()
# plt.imshow(mask[0, :, :])
# plt.show()
#
# ##  Visualize sample label image
# gtif = gdal.Open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\Dr.Shafizadeh\Dr.Shafizadeh\Ascii\Tiff\ChangeDetection\Labels_Chips\Labels.tif')
# print(gtif.GetMetadata())
# print("[ RASTER BAND COUNT ]: ", gtif.RasterCount)
# mask = gtif.GetRasterBand(1)
# print(type(mask))
# print(mask.GetStatistics(True, True))
# print(mask.GetStatistics(True, True)[1])
# mask = np.array(gtif.GetRasterBand(1).ReadAsArray())
# print(type(mask))
# print(mask.shape)
# mask = np.expand_dims(mask, axis=0)
# print(mask.shape)
#
# f = plt.figure()
# plt.imshow(mask[0, :, :])
# plt.show()

## Read Images and Add to List
#
dark = 0
less_50_cover = 0
more_50_cover = 0
rgbn = 0
count_lc_class = []
folder_path = []
iterate = 0
#
input_path = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\Chips_128x128'
for files in os.listdir(input_path):
    input_path = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\Chips_128x128'
    globals()[str(files)] = []
    print(files)
    input_path = os.path.join(input_path, files)
    print(input_path)
    # print(len(folder_path))
    for items in os.listdir(os.path.join(input_path, 'Labels')):
        if items.endswith('.tif'):
            input_label_path = os.path.join(os.path.join(input_path, 'Labels'), items)
            input_label = gdal.Open(input_label_path)
            # print(type(output_image))
            if (input_label.GetRasterBand(1).GetStatistics(True, True)[1] == 0):
                dark = dark + 1
            else:
                input_label = np.array(input_label.GetRasterBand(1).ReadAsArray())
                unique, counts = np.unique(input_label, return_counts=True)
                freq = np.asarray((unique, counts)).T

                # Check Image is dark or not
                if (freq[np.where(unique == 0)[0][0], 1] > 10000):
                    less_50_cover = less_50_cover + 1
                else:
                    more_50_cover = more_50_cover + 1

                    # Read TIFF Image
                    input_image_path = input_label_path.replace('Labels', 'Images')
                    input_image = gdal.Open(input_image_path)
                    input_image = np.array(input_image.ReadAsArray())
                    # print(input_image.shape)

                    # Add New Dim to Label
                    input_label = np.expand_dims(input_label, axis=0)
                    input_image = np.expand_dims(input_image, axis=0)

                    # Concatenate Image and Mask
                    merge = np.concatenate((input_image, input_label), axis=0)
                    globals()[str(files)].append(merge)


## Convert to numpy.array
DEM = np.array(DEM)
print('DEM : ', DEM.shape)
print('---------------------------------')
Dist_Built = np.array(Dist_Built)
print('Dist_Built : ', Dist_Built.shape)
print('---------------------------------')
Dist_Crop = np.array(Dist_Crop)
print('Dist_Crop : ', Dist_Crop.shape)
print('---------------------------------')
Dist_Marsh = np.array(Dist_Marsh)
print('Dist_Marsh : ', Dist_Marsh.shape)
print('---------------------------------')
Dist_Open = np.array(Dist_Open)
print('Dist_Open : ', Dist_Open.shape)
print('---------------------------------')
Dist_Road = np.array(Dist_Road)
print('Dist_Road : ', Dist_Road.shape)
print('---------------------------------')
Easting = np.array(Easting)
print('Easting : ', Easting.shape)
print('---------------------------------')
Northing = np.array(Northing)
print('Northing : ', Northing.shape)
print('---------------------------------')
Slope = np.array(Slope)
print('Slope : ', Slope.shape)
print('---------------------------------')

## Roll Over Axis
print(DEM.shape)
DEM = np.rollaxis(DEM, 1, 4)
print(DEM.shape)
print('---------------------------------')
print(Dist_Built.shape)
Dist_Built = np.rollaxis(Dist_Built, 1, 4)
print(Dist_Built.shape)
print('---------------------------------')
print(Dist_Crop.shape)
Dist_Crop = np.rollaxis(Dist_Crop, 1, 4)
print(Dist_Crop.shape)
print('---------------------------------')
print(Dist_Marsh.shape)
Dist_Marsh = np.rollaxis(Dist_Marsh, 1, 4)
print(Dist_Marsh.shape)
print('---------------------------------')
print(Dist_Open.shape)
Dist_Open = np.rollaxis(Dist_Open, 1, 4)
print(Dist_Open.shape)
print('---------------------------------')
print(Dist_Road.shape)
Dist_Road = np.rollaxis(Dist_Road, 1, 4)
print(Dist_Road.shape)
print('---------------------------------')
print(Easting.shape)
Easting = np.rollaxis(Easting, 1, 4)
print(Easting.shape)
print('---------------------------------')
print(Northing.shape)
Northing = np.rollaxis(Northing, 1, 4)
print(Northing.shape)
print('---------------------------------')
print(Slope.shape)
Slope = np.rollaxis(Slope, 1, 4)
print(Slope.shape)
print('---------------------------------')

# Seperate Image from Label
print(DEM.shape)
DEM_Image = DEM[:, :, :, 0]
DEM_Label = DEM[:, :, :, 1]
DEM_Image = np.expand_dims(DEM_Image, axis=-1)
DEM_Label = np.expand_dims(DEM_Label, axis=-1)
print(DEM_Image.shape)
print(DEM_Label.shape)
print('---------------------------------')
print(Dist_Built.shape)
Dist_Built_Image = Dist_Built[:, :, :, 0]
Dist_Built_Label = Dist_Built[:, :, :, 1]
Dist_Built_Image = np.expand_dims(Dist_Built_Image, axis=-1)
Dist_Built_Label = np.expand_dims(Dist_Built_Label, axis=-1)
print(Dist_Built_Image.shape)
print(Dist_Built_Label.shape)
print('---------------------------------')
print(Dist_Crop.shape)
Dist_Crop_Image = Dist_Crop[:, :, :, 0]
Dist_Crop_Label = Dist_Crop[:, :, :, 1]
Dist_Crop_Image = np.expand_dims(Dist_Crop_Image, axis=-1)
Dist_Crop_Label = np.expand_dims(Dist_Crop_Label, axis=-1)
print(Dist_Crop_Image.shape)
print(Dist_Crop_Label.shape)
print('---------------------------------')
print(Dist_Marsh.shape)
Dist_Marsh_Image = Dist_Marsh[:, :, :, 0]
Dist_Marsh_Label = Dist_Marsh[:, :, :, 1]
Dist_Marsh_Image = np.expand_dims(Dist_Marsh_Image, axis=-1)
Dist_Marsh_Label = np.expand_dims(Dist_Marsh_Label, axis=-1)
print(Dist_Marsh_Image.shape)
print(Dist_Marsh_Label.shape)
print('---------------------------------')
print(Dist_Open.shape)
Dist_Open_Image = Dist_Open[:, :, :, 0]
Dist_Open_Label = Dist_Open[:, :, :, 1]
Dist_Open_Image = np.expand_dims(Dist_Open_Image, axis=-1)
Dist_Open_Label = np.expand_dims(Dist_Open_Label, axis=-1)
print(Dist_Open_Image.shape)
print(Dist_Open_Label.shape)
print('---------------------------------')
print(Dist_Road.shape)
Dist_Road_Image = Dist_Road[:, :, :, 0]
Dist_Road_Label = Dist_Road[:, :, :, 1]
Dist_Road_Image = np.expand_dims(Dist_Road_Image, axis=-1)
Dist_Road_Label = np.expand_dims(Dist_Road_Label, axis=-1)
print(Dist_Road_Image.shape)
print(Dist_Road_Label.shape)
print('---------------------------------')
print(Easting.shape)
Easting_Image = Easting[:, :, :, 0]
Easting_Label = Easting[:, :, :, 1]
Easting_Image = np.expand_dims(Easting_Image, axis=-1)
Easting_Label = np.expand_dims(Easting_Label, axis=-1)
print(Easting_Image.shape)
print(Easting_Label.shape)
print('---------------------------------')
print(Northing.shape)
Northing_Image = Northing[:, :, :, 0]
Northing_Label = Northing[:, :, :, 1]
Northing_Image = np.expand_dims(Northing_Image, axis=-1)
Northing_Label = np.expand_dims(Northing_Label, axis=-1)
print(Northing_Image.shape)
print(Northing_Label.shape)
print('---------------------------------')
print(Slope.shape)
Slope_Image = Slope[:, :, :, 0]
Slope_Label = Slope[:, :, :, 1]
Slope_Image = np.expand_dims(Slope_Image, axis=-1)
Slope_Label = np.expand_dims(Slope_Label, axis=-1)
print(Slope_Image.shape)
print(Slope_Label.shape)
print('---------------------------------')

## Normalize Data(new method)
for images in range(0,DEM_Image.shape[0]):
    mean = np.mean(DEM_Image[images])
    std = np.std(DEM_Image[images])
    DEM_Image[images] = (DEM_Image[images] - mean) / std
    print(np.mean(DEM_Image[images]))
    print(np.std(DEM_Image[images]))
    print('---------------------------------')
DB_Image = Dist_Built_Image
for images in range(0, DEM_Image.shape[0]):
    mean = np.mean(Dist_Built_Image[images])
    std = np.std(Dist_Built_Image[images])
    DB_Image[images] = (Dist_Built_Image[images] - mean) / std
    print(np.mean(Dist_Built_Image[images]))
    print(np.std(Dist_Built_Image[images]))
    print('---------------------------------')
    mean = np.mean(Dist_Crop_Image[images])
    std = np.std(Dist_Crop_Image[images])
    Dist_Crop_Image[images] = (Dist_Crop_Image[images] - mean) / std
    print(np.mean(Dist_Crop_Image[images]))
    print(np.std(Dist_Crop_Image[images]))
    print('---------------------------------')
    mean = np.mean(Dist_Marsh_Image[images])
    std = np.std(Dist_Marsh_Image[images])
    Dist_Marsh_Image[images] = (Dist_Marsh_Image[images] - mean) / std
    print(np.mean(Dist_Marsh_Image[images]))
    print(np.std(Dist_Marsh_Image[images]))
    print('---------------------------------')
    mean = np.mean(Dist_Open_Image[images])
    std = np.std(Dist_Open_Image[images])
    Dist_Open_Image[images] = (Dist_Open_Image[images] - mean) / std
    print(np.mean(Dist_Open_Image[images]))
    print(np.std(Dist_Open_Image[images]))
    print('---------------------------------')
    mean = np.mean(Dist_Road_Image[images])
    std = np.std(Dist_Road_Image[images])
    Dist_Road_Image[images] = (Dist_Road_Image[images] - mean) / std
    print(np.mean(Dist_Road_Image[images]))
    print(np.std(Dist_Road_Image[images]))
    print('---------------------------------')
    mean = np.mean(Easting_Image[images])
    std = np.std(Easting_Image[images])
    Easting_Image[images] = (Easting_Image[images] - mean) / std
    print(np.mean(Easting_Image[images]))
    print(np.std(Easting_Image[images]))
    print('---------------------------------')
    mean = np.mean(Northing_Image[images])
    std = np.std(Northing_Image[images])
    Northing_Image[images] = (Northing_Image[images] - mean) / std
    print(np.mean(Northing_Image[images]))
    print(np.std(Northing_Image[images]))
    print('---------------------------------')
    mean = np.mean(Slope_Image[images])
    std = np.std(Slope_Image[images])
    Slope_Image[images] = (Slope_Image[images] - mean) / std
    print(np.mean(Slope_Image[images]))
    print(np.std(Slope_Image[images]))



print('---------------------------------')
Dist_Built_Image_min = Dist_Built_Image.min(axis=(1, 2), keepdims=True)
Dist_Built_Image_max = Dist_Built_Image.max(axis=(1, 2), keepdims=True)
Dist_Built_Image = (Dist_Built_Image - Dist_Built_Image_min) / (Dist_Built_Image_max - Dist_Built_Image_min)
print(Dist_Built_Image.shape)
print(np.amax(Dist_Built_Image))
print(np.amin(Dist_Built_Image))
print('---------------------------------')
# Dist_Crop_Image_min = Dist_Crop_Image.min(axis=(1, 2), keepdims=True)
# Dist_Crop_Image_max = Dist_Crop_Image.max(axis=(1, 2), keepdims=True)
# Dist_Crop_Image = (Dist_Crop_Image - Dist_Crop_Image_min) / (Dist_Crop_Image_max - Dist_Crop_Image_min)
# print(Dist_Crop_Image.shape)
# print(np.amax(Dist_Crop_Image))
# print(np.amin(Dist_Crop_Image))
# print('---------------------------------')
# Dist_Marsh_Image_min = Dist_Marsh_Image.min(axis=(1, 2), keepdims=True)
# Dist_Marsh_Image_max = Dist_Marsh_Image.max(axis=(1, 2), keepdims=True)
# Dist_Marsh_Image = (Dist_Marsh_Image - Dist_Marsh_Image_min) / (Dist_Marsh_Image_max - Dist_Marsh_Image_min)
# print(Dist_Marsh_Image.shape)
# print(np.amax(Dist_Marsh_Image))
# print(np.amin(Dist_Marsh_Image))
# print('---------------------------------')
# Dist_Open_Image_min = Dist_Open_Image.min(axis=(1, 2), keepdims=True)
# Dist_Open_Image_max = Dist_Open_Image.max(axis=(1, 2), keepdims=True)
# Dist_Open_Image = (Dist_Open_Image - Dist_Open_Image_min) / (Dist_Open_Image_max - Dist_Open_Image_min)
# print(Dist_Open_Image.shape)
# print(np.amax(Dist_Open_Image))
# print(np.amin(Dist_Open_Image))
# print('---------------------------------')
# Dist_Road_Image_min = Dist_Road_Image.min(axis=(1, 2), keepdims=True)
# Dist_Road_Image_max = Dist_Road_Image.max(axis=(1, 2), keepdims=True)
# Dist_Road_Image = (Dist_Road_Image - Dist_Road_Image_min) / (Dist_Road_Image_max - Dist_Road_Image_min)
# print(Dist_Road_Image.shape)
# print(np.amax(Dist_Road_Image))
# print(np.amin(Dist_Road_Image))
# print('---------------------------------')
# Easting_Image_min = Easting_Image.min(axis=(1, 2), keepdims=True)
# Easting_Image_max = Easting_Image.max(axis=(1, 2), keepdims=True)
# Easting_Image = (Easting_Image - Easting_Image_min) / (Easting_Image_max - Easting_Image_min)
# print(Easting_Image.shape)
# print(np.amax(Easting_Image))
# print(np.amin(Easting_Image))
# print('---------------------------------')
# Northing_Image_min = Northing_Image.min(axis=(1, 2), keepdims=True)
# Northing_Image_max = Northing_Image.max(axis=(1, 2), keepdims=True)
# Northing_Image = (Northing_Image - Northing_Image_min) / (Northing_Image_max - Northing_Image_min)
# print(Northing_Image.shape)
# print(np.amax(Northing_Image))
# print(np.amin(Northing_Image))
# print('---------------------------------')
# Slope_Image_min = Slope_Image.min(axis=(1, 2), keepdims=True)
# Slope_Image_max = Slope_Image.max(axis=(1, 2), keepdims=True)
# Slope_Image = (Slope_Image - Slope_Image_min) / (Slope_Image_max - Slope_Image_min)
# print(Slope_Image.shape)
# print(np.amax(Slope_Image))
# print(np.amin(Slope_Image))
# print('---------------------------------')
# ##
# print(DEM_Label.shape)
# print(np.amax(DEM_Label))
# print(np.amin(DEM_Label))
# print('---------------------------------')
# print(Dist_Built_Label.shape)
# print(np.amax(Dist_Built_Label))
# print(np.amin(Dist_Built_Label))
# print('---------------------------------')
# print(Dist_Crop_Label.shape)
# print(np.amax(Dist_Crop_Label))
# print(np.amin(Dist_Crop_Label))
# print('---------------------------------')
# print(Dist_Marsh_Label.shape)
# print(np.amax(Dist_Marsh_Label))
# print(np.amin(Dist_Marsh_Label))
# print('---------------------------------')
# print(Dist_Open_Label.shape)
# print(np.amax(Dist_Open_Label))
# print(np.amin(Dist_Open_Label))
# print('---------------------------------')
# print(Dist_Road_Label.shape)
# print(np.amax(Dist_Road_Label))
# print(np.amin(Dist_Road_Label))
# print('---------------------------------')
# print(Easting_Label.shape)
# print(np.amax(Easting_Label))
# print(np.amin(Easting_Label))
# print('---------------------------------')
# print(Northing_Label.shape)
# print(np.amax(Northing_Label))
# print(np.amin(Northing_Label))
# print('---------------------------------')
# print(Slope_Label.shape)
# print(np.amax(Slope_Label))
# print(np.amin(Slope_Label))
# print('---------------------------------')
#
# ## Stack back label to image
# print(DEM_Image.shape)
# print(DEM_Label.shape)
# DEM = np.concatenate((DEM_Image, DEM_Label), axis=3)
# print(DEM.shape)
# print('---------------------------------')
# print(Dist_Built_Image.shape)
# print(Dist_Built_Label.shape)
# Dist_Built = np.concatenate((Dist_Built_Image, Dist_Built_Label), axis=3)
# print(Dist_Built.shape)
# print('---------------------------------')
# print(Dist_Crop_Image.shape)
# print(Dist_Crop_Label.shape)
# Dist_Crop = np.concatenate((Dist_Crop_Image, Dist_Crop_Label), axis=3)
# print(Dist_Crop.shape)
# print('---------------------------------')
# print(Dist_Marsh_Image.shape)
# print(Dist_Marsh_Label.shape)
# Dist_Marsh = np.concatenate((Dist_Marsh_Image, Dist_Marsh_Label), axis=3)
# print(Dist_Marsh.shape)
# print('---------------------------------')
# print(Dist_Open_Image.shape)
# print(Dist_Open_Label.shape)
# Dist_Open = np.concatenate((Dist_Open_Image, Dist_Open_Label), axis=3)
# print(Dist_Open.shape)
# print('---------------------------------')
# print(Dist_Road_Image.shape)
# print(Dist_Road_Label.shape)
# Dist_Road = np.concatenate((Dist_Road_Image, Dist_Road_Label), axis=3)
# print(Dist_Road.shape)
# print('---------------------------------')
# print(Easting_Image.shape)
# print(Easting_Label.shape)
# Easting = np.concatenate((Easting_Image, Easting_Label), axis=3)
# print(Easting.shape)
# print('---------------------------------')
# print(Northing_Image.shape)
# print(Northing_Label.shape)
# Northing = np.concatenate((Northing_Image, Northing_Label), axis=3)
# print(Northing.shape)
# print('---------------------------------')
# print(Slope_Image.shape)
# print(Slope_Label.shape)
# Slope = np.concatenate((Slope_Image, Slope_Label), axis=3)
# print(Slope.shape)
# ##
# DEM_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\DEM.h5', 'w')
# Dist_Built_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\Dist_Built.h5', 'w')
# Dist_Crop_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\Dist_Crop.h5', 'w')
# Dist_Marsh_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\Dist_Marsh.h5', 'w')
# Dist_Open_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\Dist_Open.h5', 'w')
# Dist_Road_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\Dist_Road.h5', 'w')
# Easting_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\Easting.h5', 'w')
# Northing_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\Northing.h5', 'w')
# Slope_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\Slope.h5', 'w')
#
#
# DEM_hf.create_dataset('DEM', data=DEM)
# Dist_Built_hf.create_dataset('Dist_Built', data=Dist_Built)
# Dist_Crop_hf.create_dataset('Dist_Crop', data=Dist_Crop)
# Dist_Marsh_hf.create_dataset('Dist_Marsh', data=Dist_Marsh)
# Dist_Open_hf.create_dataset('Dist_Open', data=Dist_Open)
# Dist_Road_hf.create_dataset('Dist_Road', data=Dist_Road)
# Easting_hf.create_dataset('Easting', data=Easting)
# Northing_hf.create_dataset('Northing', data=Northing)
# Slope_hf.create_dataset('Slope', data=Slope)
#
# DEM = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\DEM.h5', 'r')
# Dist_Built = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\Dist_Built.h5', 'r')
# Dist_Crop = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\Dist_Crop.h5','r')
# Dist_Marsh = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\Dist_Marsh.h5','r')
# Dist_Open = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\Dist_Open.h5', 'r')
# Dist_Road = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\Dist_Road.h5', 'r')
# Easting = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\Easting.h5', 'r')
# Northing = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\Northing.h5', 'r')
# Slope = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\Slope.h5', 'r')
#
# DEM = np.array(DEM.get('DEM')).astype(np.float32)
# Dist_Built = np.array(Dist_Built.get('Dist_Built')).astype(np.float32)
# Dist_Crop = np.array(Dist_Crop.get('Dist_Crop')).astype(np.float32)
# Dist_Marsh = np.array(Dist_Marsh.get('Dist_Marsh')).astype(np.float32)
# Dist_Open = np.array(Dist_Open.get('Dist_Open')).astype(np.float32)
# Dist_Road = np.array(Dist_Road.get('Dist_Road')).astype(np.float32)
# Easting = np.array(Easting.get('Easting')).astype(np.float32)
# Northing = np.array(Northing.get('Northing')).astype(np.float32)
# Slope = np.array(Slope.get('Slope')).astype(np.float32)
#
# print(DEM.shape)
# print(np.amax(DEM))
# print(np.amin(DEM))
# print('---------------------------------')
# print(Dist_Built.shape)
# print(np.amax(Dist_Built))
# print(np.amin(Dist_Built))
# print('---------------------------------')
# print(Dist_Crop.shape)
# print(np.amax(Dist_Crop))
# print(np.amin(Dist_Crop))
# print('---------------------------------')
# print(Dist_Marsh.shape)
# print(np.amax(Dist_Marsh))
# print(np.amin(Dist_Marsh))
# print('---------------------------------')
# print(Dist_Open.shape)
# print(np.amax(Dist_Open))
# print(np.amin(Dist_Open))
# print('---------------------------------')
# print(Dist_Road.shape)
# print(np.amax(Dist_Road))
# print(np.amin(Dist_Road))
# print('---------------------------------')
# print(Easting.shape)
# print(np.amax(Easting))
# print(np.amin(Easting))
# print('---------------------------------')
# print(Northing.shape)
# print(np.amax(Northing))
# print(np.amin(Northing))
# print('---------------------------------')
# print(Slope.shape)
# print(np.amax(Slope))
# print(np.amin(Slope))
#
#
# ## Data Augmentation
# ##
# def augment_data(dataset, augementation_factor=1, use_random_rotation=True,
#                  use_random_shear=True, use_random_shift=True, use_random_zoom=True):
#     augmented_image = []
#
#     for num in range(0, dataset.shape[0]):
#         for i in range(0, augementation_factor):
#
#             # original image:
#             augmented_image.append(dataset[num])
#
#             rotation = [-180, -90, 90, 180]
#             if use_random_rotation:
#                 augmented_image.append(
#                     tf.contrib.keras.preprocessing.image.random_rotation(dataset[num], random.choice(rotation),
#                                                                          row_axis=0, col_axis=1, channel_axis=2))
#
#             # shear = [0.1, 0.2, 0.3, 0.4, 0.5]
#             # if use_random_shear:
#             #     augmented_image.append(tf.contrib.keras.preprocessing.image.random_shear(dataset[num], random.choice(rotation), row_axis=0, col_axis=1, channel_axis=2))
#
#             # if use_random_shift:
#             #     augmented_image.append(tf.contrib.keras.preprocessing.image.random_shift(dataset[num], 5, 5, row_axis=0, col_axis=1, channel_axis=2))
#
#             # if use_random_zoom:
#             #     augmented_image.append(tf.contrib.keras.preprocessing.image.random_zoom(dataset[num], (0.05, 0.45), row_axis=0, col_axis=1, channel_axis=2))
#
#     return np.array(augmented_image)
# print(DEM.shape)
# augmented_DEM = augment_data(DEM, augementation_factor = 4)
# print(augmented_DEM.shape)
# print('---------------------------------')
# print(Dist_Built.shape)
# augmented_Dist_Built = augment_data(Dist_Built, augementation_factor = 4)
# print(augmented_Dist_Built.shape)
# print('---------------------------------')
# print(Dist_Crop.shape)
# augmented_Dist_Crop = augment_data(Dist_Crop, augementation_factor = 4)
# print(augmented_Dist_Crop.shape)
# print('---------------------------------')
# print(Dist_Marsh.shape)
# augmented_Dist_Marsh = augment_data(Dist_Marsh, augementation_factor = 4)
# print(augmented_Dist_Marsh.shape)
# print('---------------------------------')
# print(Dist_Open.shape)
# augmented_Dist_Open = augment_data(Dist_Open, augementation_factor = 4)
# print(augmented_Dist_Open.shape)
# print('---------------------------------')
# print(Dist_Road.shape)
# augmented_Dist_Road = augment_data(Dist_Road, augementation_factor = 4)
# print(augmented_Dist_Road.shape)
# print('---------------------------------')
# print(Easting.shape)
# augmented_Easting = augment_data(Easting, augementation_factor = 4)
# print(augmented_Easting.shape)
# print('---------------------------------')
# print(Northing.shape)
# augmented_Northing = augment_data(Northing, augementation_factor = 4)
# print(augmented_Northing.shape)
# print('---------------------------------')
# print(Slope.shape)
# augmented_Slope = augment_data(Slope, augementation_factor = 4)
# print(augmented_Slope.shape)
# print('---------------------------------')
# # globals()['Img%s' % num_of_images]  = np.expand_dims(['Img%s' % num_of_images] , axis=0)
# # print(['Img%s' % num_of_images].shape)
# # globals()['Augmented_Img%s' % num_of_images] = augment_data(['Img%s' % num_of_images], augementation_factor=4)
# # print(['Augmented_Img%s' % num_of_images].shape)
# # print('---------------------------------')
#
# ######
# # Im = np.zeros(shape=(256,256,9))
# #
# # for num_of_images in range(0, DEM.shape[0]):
# #     globals()['Img%s' % num_of_images]  = np.dstack([DEM_Image[num_of_images,:,:], Dist_Built_Image[num_of_images,:,:], Dist_Crop_Image[num_of_images,:,:],
# #                                     Dist_Marsh_Image[num_of_images,:,:], Dist_Open_Image[num_of_images,:,:], Dist_Road_Image[num_of_images,:,:],
# #                                     Easting_Image[num_of_images,:,:], Northing_Image[num_of_images,:,:], Slope_Image[num_of_images,:,:], DEM_Label[num_of_images,:,:]])
# All_Images  = np.stack([augmented_DEM[:,:,:,0], augmented_Dist_Built[:,:,:,0], augmented_Dist_Crop[:,:,:,0],
#                                     augmented_Dist_Marsh[:,:,:,0], augmented_Dist_Open[:,:,:,0], augmented_Dist_Road[:,:,:,0],
#                                     augmented_Easting[:,:,:,0], augmented_Northing[:,:,:,0], augmented_Slope[:,:,:,0], augmented_Slope[:,:,:,1]], axis=3)
# print(All_Images.shape)
# ## Shuffle
# np.random.shuffle(All_Images)
# print(All_Images.shape)
# ## Separate Label Dim from Rest of Array
# Images = All_Images[:, :, :, 0:9]
# print(Images.shape)
# Labels = All_Images[:, :, :, 9:10]
# print(Labels.shape)
#
# ## Convert Label Channel to Int
# print(Images.dtype)
# print(Labels.dtype)
#
# Labels = Labels.astype(int)
# print(Labels.dtype)
#
# Labels = keras.utils.to_categorical(Labels, num_classes = 2)
# print(Labels.shape)

## Read data
Images_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\Images.h5', 'r')
Images = np.array(Images_hf.get('Images')).astype(np.float32)

Labels_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\Labels.h5', 'r')
Labels = np.array(Labels_hf.get('Labels')).astype(np.float32)
# Labels = keras.utils.to_categorical(Labels, num_classes = 2)

print(Images.shape)
print(Labels.shape)