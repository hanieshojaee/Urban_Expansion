import os
import numpy as np
import h5py
import gdal
from sklearn.preprocessing import MinMaxScaler

########Test
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

#
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
#
# ## Normalize Data(new method : MinMaxScaler)
scaler = MinMaxScaler(feature_range=(0, 1))
##
DEM_Image = scaler.fit_transform(DEM)
print(DEM_Image.shape)
print(np.amax(DEM_Image))
print(np.amin(DEM_Image))
print('---------------------------------')
Dist_Built_Image = scaler.fit_transform(Dist_Built_Image)
print(Dist_Built_Image.shape)
print(np.amax(Dist_Built_Image))
print(np.amin(Dist_Built_Image))
print('---------------------------------')
Dist_Crop_Image = scaler.fit_transform(Dist_Crop_Image)
print(Dist_Crop_Image.shape)
print(np.amax(Dist_Crop_Image))
print(np.amin(Dist_Crop_Image))
print('---------------------------------')
Dist_Marsh_Image = scaler.fit_transform(Dist_Marsh_Image)
print(Dist_Marsh_Image.shape)
print(np.amax(Dist_Marsh_Image))
print(np.amin(Dist_Marsh_Image))
print('---------------------------------')
Dist_Open_Image = scaler.fit_transform(Dist_Open_Image)
print(Dist_Open_Image.shape)
print(np.amax(Dist_Open_Image))
print(np.amin(Dist_Open_Image))
print('---------------------------------')
Dist_Road_Image = scaler.fit_transform(Dist_Road_Image)
print(Dist_Road_Image.shape)
print(np.amax(Dist_Road_Image))
print(np.amin(Dist_Road_Image))
print('---------------------------------')
Easting_Image = scaler.fit_transform(Easting_Image)
print(Easting_Image.shape)
print(np.amax(Easting_Image))
print(np.amin(Easting_Image))
print('---------------------------------')
Northing_Image = scaler.fit_transform(Northing_Image)
print(Northing_Image.shape)
print(np.amax(Northing_Image))
print(np.amin(Northing_Image))
print('---------------------------------')
Slope_Image = scaler.fit_transform(Slope_Image)
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