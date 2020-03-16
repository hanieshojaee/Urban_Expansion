from PIL import Image
import os
import gdal
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import h5py
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Layer Stacking(Image & Label)
# Dem_Label = np.concatenate((np.expand_dims(np.array(Img.GetRasterBand(1).ReadAsArray()), axis = 2),np.expand_dims(np.array(Dem.GetRasterBand(1).ReadAsArray()), axis = 2 )), axis=2)
# print(Dem_Label.shape)

# Chip Craeting
# Img = Image.open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\ChangeDetection\1994-2004\Final_Change_Map.tif')
# out_path = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\Chips_256x256\Stride64x64\Dist_Road\Labels'
#
# Image_width, Image_height = Img.size
# width, height = 256, 256
# stride_x, stride_y = 64, 64
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

dark = 0
dark_image = 0
less_50_cover = 0
more_50_cover = 0
less_50_cover_image = 0
more_50_cover_image = 0
rgbn = 0
count_lc_class = []
folder_path = []
iterate = 0
# frequancy =[]

# my_list = []
input_path = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\Chips_256x256\Stride64x64'
for files in os.listdir(input_path):
    # print(files)
    input_path = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\Chips_256x256\Stride64x64'
    globals()[str(files)] = []

    input_path = os.path.join(input_path, files)
    print(input_path)
    # print(len(folder_path))
    for items in os.listdir(os.path.join(input_path, 'Labels')):
        if items.endswith('.tif'):
            input_label_path = os.path.join(os.path.join(input_path, 'Labels'), items)
            # print(input_label_path)
            input_label = gdal.Open(input_label_path)
            if (input_label.GetRasterBand(1).GetStatistics(True, True)[1] == 0):
                # print(input_label_path)
                dark = dark + 1
            else:
                input_label = np.array(input_label.GetRasterBand(1).ReadAsArray())
                unique, counts = np.unique(input_label, return_counts=True)
                freq = np.asarray((unique, counts)).T
                # frequancy.append(freq[1, 1])

                # Check Image is dark or not
                if ((freq[0][1]) > 55000 ):
                    less_50_cover = less_50_cover + 1
                else:
                    more_50_cover = more_50_cover + 1

                    # Read TIFF Image
                    input_image_path = input_label_path.replace('Labels', 'Images')
                    input_image = gdal.Open(input_image_path)
                    # input_image = np.array(input_image.ReadAsArray())

                    if (input_image.GetRasterBand(1).GetStatistics(True, True)[1] == 0) :
                        dark_image += 1
                    else:
                        input_image = np.array(input_image.GetRasterBand(1).ReadAsArray())
                        unique, counts = np.unique(input_image, return_counts=True)
                        freq_image = np.asarray((unique, counts)).T
                        # frequancy.append(freq[1, 1])

                        if ((freq_image[0][1]) > 256*256/2):
                            less_50_cover_image = less_50_cover_image + 1
                        else:
                            more_50_cover_image = more_50_cover_image + 1



                    # Add New Dim to Label
                            input_label = np.expand_dims(input_label, axis=0)
                            input_image = np.expand_dims(input_image, axis=0)

                    # Concatenate Image and Mask
                            merge = np.concatenate((input_image, input_label), axis=0)
                    # new_list = []
                    # new_list.append(merge)
                    # globals()[str(files)] = []
                    # files.append(merge)
                    # iterate += 1
                    # globals()[str(files)].append(merge)

                    # Drivers[keys].append(merge)
                    # [str(files)] .append(merge)
                    # for keys in Drivers.keys():
                    #     if keys == files:
                        globals()[str(files)].append(merge)

    # [str(files)].append(merge)
    # my_list.append(new_list)

    # rgbn +=  1

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
DEM_Image_min = DEM_Image.min(axis=(1, 2), keepdims=True)
DEM_Image_max = DEM_Image.max(axis=(1, 2), keepdims=True)
print(np.amax(DEM_Image))
print(np.amin(DEM_Image))
DEM_Image = (DEM_Image - DEM_Image_min) / (DEM_Image_max - DEM_Image_min)
print(DEM_Image.shape)
print(np.amax(DEM_Image))
print(np.amin(DEM_Image))
print('---------------------------------')
Dist_Built_Image_min = Dist_Built_Image.min(axis=(1, 2), keepdims=True)
Dist_Built_Image_max = Dist_Built_Image.max(axis=(1, 2), keepdims=True)
print(np.amax(Dist_Built_Image))
print(np.amin(Dist_Built_Image))
Dist_Built_Image = (Dist_Built_Image - Dist_Built_Image_min) / (Dist_Built_Image_max - Dist_Built_Image_min)
print(Dist_Built_Image.shape)
print(np.amax(Dist_Built_Image))
print(np.amin(Dist_Built_Image))
print('---------------------------------')
Dist_Crop_Image_min = Dist_Crop_Image.min(axis=(1, 2), keepdims=True)
Dist_Crop_Image_max = Dist_Crop_Image.max(axis=(1, 2), keepdims=True)
print(np.amax(Dist_Crop_Image))
print(np.amin(Dist_Crop_Image))
Dist_Crop_Image = (Dist_Crop_Image - Dist_Crop_Image_min) / (Dist_Crop_Image_max - Dist_Crop_Image_min)
print(Dist_Crop_Image.shape)
print(np.amax(Dist_Crop_Image))
print(np.amin(Dist_Crop_Image))
print('---------------------------------')
Dist_Marsh_Image_min = Dist_Marsh_Image.min(axis=(1, 2), keepdims=True)
Dist_Marsh_Image_max = Dist_Marsh_Image.max(axis=(1, 2), keepdims=True)
print(np.amax(Dist_Marsh_Image))
print(np.amin(Dist_Marsh_Image))
Dist_Marsh_Image = (Dist_Marsh_Image - Dist_Marsh_Image_min) / (Dist_Marsh_Image_max - Dist_Marsh_Image_min)
print(Dist_Marsh_Image.shape)
print(np.amax(Dist_Marsh_Image))
print(np.amin(Dist_Marsh_Image))
print('---------------------------------')
Dist_Open_Image_min = Dist_Open_Image.min(axis=(1, 2), keepdims=True)
Dist_Open_Image_max = Dist_Open_Image.max(axis=(1, 2), keepdims=True)
print(np.amax(Dist_Open_Image))
print(np.amin(Dist_Open_Image))
Dist_Open_Image = (Dist_Open_Image - Dist_Open_Image_min) / (Dist_Open_Image_max - Dist_Open_Image_min)
print(Dist_Open_Image.shape)
print(np.amax(Dist_Open_Image))
print(np.amin(Dist_Open_Image))
print('---------------------------------')
Dist_Road_Image_min = Dist_Road_Image.min(axis=(1, 2), keepdims=True)
Dist_Road_Image_max = Dist_Road_Image.max(axis=(1, 2), keepdims=True)
print(np.amax(Dist_Road_Image))
print(np.amin(Dist_Road_Image))
Dist_Road_Image = (Dist_Road_Image - Dist_Road_Image_min) / (Dist_Road_Image_max - Dist_Road_Image_min)
print(Dist_Road_Image.shape)
print(np.amax(Dist_Road_Image))
print(np.amin(Dist_Road_Image))
print('---------------------------------')
Easting_Image_min = Easting_Image.min(axis=(1, 2), keepdims=True)
Easting_Image_max = Easting_Image.max(axis=(1, 2), keepdims=True)
print(np.amax(Easting_Image))
print(np.amin(Easting_Image))
Easting_Image = (Easting_Image - Easting_Image_min) / (Easting_Image_max - Easting_Image_min)
print(Easting_Image.shape)
print(np.amax(Easting_Image))
print(np.amin(Easting_Image))
print('---------------------------------')
Northing_Image_min = Northing_Image.min(axis=(1, 2), keepdims=True)
Northing_Image_max = Northing_Image.max(axis=(1, 2), keepdims=True)
print(np.amax(Northing_Image))
print(np.amin(Northing_Image))
Northing_Image = (Northing_Image - Northing_Image_min) / (Northing_Image_max - Northing_Image_min)
print(Northing_Image.shape)
print(np.amax(Northing_Image))
print(np.amin(Northing_Image))
print('---------------------------------')
Slope_Image_min = Slope_Image.min(axis=(1, 2), keepdims=True)
Slope_Image_max = Slope_Image.max(axis=(1, 2), keepdims=True)
print(np.amax(Slope_Image))
print(np.amin(Slope_Image))
Slope_Image = (Slope_Image - Slope_Image_min) / (Slope_Image_max - Slope_Image_min)
print(Slope_Image.shape)
print(np.amax(Slope_Image))
print(np.amin(Slope_Image))
print('---------------------------------')
##
print(DEM_Label.shape)
print(np.amax(DEM_Label))
print(np.amin(DEM_Label))
print('---------------------------------')
print(Dist_Built_Label.shape)
print(np.amax(Dist_Built_Label))
print(np.amin(Dist_Built_Label))
print('---------------------------------')
print(Dist_Crop_Label.shape)
print(np.amax(Dist_Crop_Label))
print(np.amin(Dist_Crop_Label))
print('---------------------------------')
print(Dist_Marsh_Label.shape)
print(np.amax(Dist_Marsh_Label))
print(np.amin(Dist_Marsh_Label))
print('---------------------------------')
print(Dist_Open_Label.shape)
print(np.amax(Dist_Open_Label))
print(np.amin(Dist_Open_Label))
print('---------------------------------')
print(Dist_Road_Label.shape)
print(np.amax(Dist_Road_Label))
print(np.amin(Dist_Road_Label))
print('---------------------------------')
print(Easting_Label.shape)
print(np.amax(Easting_Label))
print(np.amin(Easting_Label))
print('---------------------------------')
print(Northing_Label.shape)
print(np.amax(Northing_Label))
print(np.amin(Northing_Label))
print('---------------------------------')
print(Slope_Label.shape)
print(np.amax(Slope_Label))
print(np.amin(Slope_Label))
print('---------------------------------')
##
## Stack back label to image
print(DEM_Image.shape)
print(DEM_Label.shape)
DEM = np.concatenate((DEM_Image, DEM_Label), axis=3)
print(DEM.shape)
print('---------------------------------')
print(Dist_Built_Image.shape)
print(Dist_Built_Label.shape)
Dist_Built = np.concatenate((Dist_Built_Image, Dist_Built_Label), axis=3)
print(Dist_Built.shape)
print('---------------------------------')
print(Dist_Crop_Image.shape)
print(Dist_Crop_Label.shape)
Dist_Crop = np.concatenate((Dist_Crop_Image, Dist_Crop_Label), axis=3)
print(Dist_Crop.shape)
print('---------------------------------')
print(Dist_Marsh_Image.shape)
print(Dist_Marsh_Label.shape)
Dist_Marsh = np.concatenate((Dist_Marsh_Image, Dist_Marsh_Label), axis=3)
print(Dist_Marsh.shape)
print('---------------------------------')
print(Dist_Open_Image.shape)
print(Dist_Open_Label.shape)
Dist_Open = np.concatenate((Dist_Open_Image, Dist_Open_Label), axis=3)
print(Dist_Open.shape)
print('---------------------------------')
print(Dist_Road_Image.shape)
print(Dist_Road_Label.shape)
Dist_Road = np.concatenate((Dist_Road_Image, Dist_Road_Label), axis=3)
print(Dist_Road.shape)
print('---------------------------------')
print(Easting_Image.shape)
print(Easting_Label.shape)
Easting = np.concatenate((Easting_Image, Easting_Label), axis=3)
print(Easting.shape)
print('---------------------------------')
print(Northing_Image.shape)
print(Northing_Label.shape)
Northing = np.concatenate((Northing_Image, Northing_Label), axis=3)
print(Northing.shape)
print('---------------------------------')
print(Slope_Image.shape)
print(Slope_Label.shape)
Slope = np.concatenate((Slope_Image, Slope_Label), axis=3)
print(Slope.shape)
print('---------------------------------')
##
All_Images  = np.stack([DEM[:,:,:,0], Dist_Built[:,:,:,0], Dist_Crop[:,:,:,0], Dist_Marsh[:,:,:,0], Dist_Open[:,:,:,0],
                        Dist_Road[:,:,:,0], Easting[:,:,:,0], Northing[:,:,:,0], Slope[:,:,:,0], Slope[:,:,:,1]], axis=3)
print(All_Images.shape)

plt.figure()
plt.imshow(All_Images[0, :, :, 1])
plt.show()
## Shuffle
# np.random.shuffle(All_Images)
# print(All_Images.shape)
## Separate Label Dim from Rest of Array
Images = All_Images[:, :, :, 0:9]
print(Images.shape)
Labels = All_Images[:, :, :, 9:10]
print(Labels.shape)

## Convert Label Channel to Int
print(Images.dtype)
print(Labels.dtype)

Labels = Labels.astype(int)
print(Labels.dtype)

## Save in HDF file
DEM_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\DEM.h5', 'w')
Dist_Built_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Dist_Built.h5', 'w')
Dist_Crop_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Dist_Crop.h5', 'w')
Dist_Marsh_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Dist_Marsh.h5', 'w')
Dist_Open_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Dist_Open.h5', 'w')
Dist_Road_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Dist_Road.h5', 'w')
Easting_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Easting.h5', 'w')
Northing_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Northing.h5', 'w')
Slope_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Slope.h5', 'w')


DEM_hf.create_dataset('DEM', data=DEM)
Dist_Built_hf.create_dataset('Dist_Built', data=Dist_Built)
Dist_Crop_hf.create_dataset('Dist_Crop', data=Dist_Crop)
Dist_Marsh_hf.create_dataset('Dist_Marsh', data=Dist_Marsh)
Dist_Open_hf.create_dataset('Dist_Open', data=Dist_Open)
Dist_Road_hf.create_dataset('Dist_Road', data=Dist_Road)
Easting_hf.create_dataset('Easting', data=Easting)
Northing_hf.create_dataset('Northing', data=Northing)
Slope_hf.create_dataset('Slope', data=Slope)

DEM_hf.close()
Dist_Built_hf.close()
Dist_Crop_hf.close()
Dist_Marsh_hf.close()
Dist_Open_hf.close()
Dist_Road_hf.close()
Easting_hf.close()
Northing_hf.close()
Slope_hf.close()

All_Images_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\All_Images.h5', 'w')
Images_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Images.h5', 'w')
Labels_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Labels.h5', 'w')

All_Images_hf.create_dataset('All_Images', data=All_Images)
Images_hf.create_dataset('Images', data=Images)
Labels_hf.create_dataset('Labels', data=Labels)

All_Images_hf.close()
Images_hf.close()
Labels_hf.close()

## Read and check data
DEM = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\DEM.h5', 'r')
Dist_Built = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Dist_Built.h5', 'r')
Dist_Crop = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Dist_Crop.h5','r')
Dist_Marsh = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Dist_Marsh.h5','r')
Dist_Open = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Dist_Open.h5', 'r')
Dist_Road = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Dist_Road.h5', 'r')
Easting = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Easting.h5', 'r')
Northing = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Northing.h5', 'r')
Slope = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Slope.h5', 'r')

DEM = np.array(DEM.get('DEM')).astype(np.float32)
Dist_Built = np.array(Dist_Built.get('Dist_Built')).astype(np.float32)
Dist_Crop = np.array(Dist_Crop.get('Dist_Crop')).astype(np.float32)
Dist_Marsh = np.array(Dist_Marsh.get('Dist_Marsh')).astype(np.float32)
Dist_Open = np.array(Dist_Open.get('Dist_Open')).astype(np.float32)
Dist_Road = np.array(Dist_Road.get('Dist_Road')).astype(np.float32)
Easting = np.array(Easting.get('Easting')).astype(np.float32)
Northing = np.array(Northing.get('Northing')).astype(np.float32)
Slope = np.array(Slope.get('Slope')).astype(np.float32)

All_Images = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\All_Images.h5', 'r')
Images = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Images.h5', 'r')
Labels = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Labels.h5', 'r')

All_Images = np.array(All_Images.get('All_Images')).astype(np.float32)
Images = np.array(Images.get('Images')).astype(np.float32)
Labels = np.array(Labels.get('Labels')).astype(np.float32)


print(DEM.shape)
print(np.amax(DEM))
print(np.amin(DEM))
print('---------------------------------')
print(Dist_Built.shape)
print(np.amax(Dist_Built))
print(np.amin(Dist_Built))
print('---------------------------------')
print(Dist_Crop.shape)
print(np.amax(Dist_Crop))
print(np.amin(Dist_Crop))
print('---------------------------------')
print(Dist_Marsh.shape)
print(np.amax(Dist_Marsh))
print(np.amin(Dist_Marsh))
print('---------------------------------')
print(Dist_Open.shape)
print(np.amax(Dist_Open))
print(np.amin(Dist_Open))
print('---------------------------------')
print(Dist_Road.shape)
print(np.amax(Dist_Road))
print(np.amin(Dist_Road))
print('---------------------------------')
print(Easting.shape)
print(np.amax(Easting))
print(np.amin(Easting))
print('---------------------------------')
print(Northing.shape)
print(np.amax(Northing))
print(np.amin(Northing))
print('---------------------------------')
print(Slope.shape)
print(np.amax(Slope))
print(np.amin(Slope))
print('---------------------------------')
print(All_Images.shape)
print(np.amax(All_Images))
print(np.amin(All_Images))
print('---------------------------------')
print(Images.shape)
print(np.amax(Images))
print(np.amin(Images))
print('---------------------------------')
print(Labels.shape)
print(np.amax(Labels))
print(np.amin(Labels))
print('---------------------------------')
##
