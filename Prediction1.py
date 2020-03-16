import os
import gdal
import numpy as np
from tqdm import tqdm
import h5py
import keras
from keras import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model, load_model
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from IPython.display import HTML, display
import tabulate
from PIL import Image
import tifffile
from skimage import img_as_uint
import scipy.io as io

## Chip Size:256*256

## Land Cover Classes
table = [["Non_Urban", 0], ["Urban", 1]]
display(HTML(tabulate.tabulate(table, tablefmt='html')))

## Number of Land Cover Classes
number_of_classes = 2
color_map = [[250, 0, 0], [0, 0, 250]]


## Define IoU Metric
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


# ## Define Custom Loss Function
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


## Save GeoTIFF Image
def save_image(image_data, path):
    driver = gdal.GetDriverByName('GTiff')

    # Set Info of Image
    height, width = image_data.shape
    dataset = driver.Create(path, width, height, 1, gdal.GDT_Byte)
    # dataset.SetGeoTransform(geo_transform)
    # dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(image_data)

    # Create Color Table
    color_table = gdal.ColorTable()

    for i in range(number_of_classes):
        color_table.SetColorEntry(i, tuple(color_map[i]) + (255,))
    dataset.GetRasterBand(1).SetRasterColorTable(color_table)
    dataset.GetRasterBand(1).SetNoDataValue(255)

    dataset.FlushCache()


def eval_image(input_image_path, model, output_image_path):
    input_dataset = h5py.File(input_image_path, 'r')
    input_Image = np.array(input_dataset.get('Normalized_2004_Image_With_All_Drivers')).astype(np.float32)
    print(input_Image.shape)
    print(np.amin(input_Image), np.amax(input_Image))

    # input_dataset = gdal.Open(input_image_path)
    # input_image = input_dataset.ReadAsArray().astype(np.float32)
    # input_image = np.expand_dims(input_image, axis=2)

    # print(input_image.shape)
    # input_image = np.rollaxis(input_image, 0, 3)
    h, w, n = input_Image.shape
    # print(input_image.shape)

    model_input_height, model_input_width, model_input_channels = model.layers[0].input_shape[1:4]
    print(model_input_height, model_input_width, model_input_channels)
    model_output_height, model_output_width, model_output_channels = model.layers[len(model.layers) - 1].output_shape[
                                                                     1:4]
    print(model_output_height, model_output_width, model_output_channels)

    padding_y = int((model_input_height - model_output_height) / 2)
    padding_x = int((model_input_width - model_output_width) / 2)
    print(padding_y, padding_x)
    assert model_output_channels == number_of_classes

    pred_lc_image = np.zeros((h, w, number_of_classes))
    print(pred_lc_image.shape)
    mask = np.ones((h, w))
    print(mask.shape)

    irows, icols = [], []
    batch_size = 19
    minibatch = []
    ibatch = 0

    n_rows = int(h / model_output_height)
    print(n_rows)
    n_cols = int(w / model_output_width)
    print(n_cols)
    batch_size = (n_rows * n_cols) - 1

    mb_array = np.zeros((batch_size, model_input_width, model_input_height, model_input_channels))
    print(mb_array.shape)

    for row_idx in tqdm(range(n_rows)):
        for col_idx in range(n_cols):

            subimage = input_Image[row_idx * model_output_height:row_idx * model_output_height + model_input_height,
                       col_idx * model_output_width:col_idx * model_output_width + model_input_width, :]
            ## /256 for normalized
            print(subimage.shape)
            print(np.amin(subimage), np.amax(subimage))
            # imshow(np.array(subimage[:,:,1]))
            # plt.show()

            if (subimage.shape == model.layers[0].input_shape[1:4]):

                mb_array[ibatch] = subimage
                ibatch += 1
                irows.append((row_idx * model_output_height + padding_y,
                              row_idx * model_output_height + model_input_height - padding_y))
                icols.append((col_idx * model_output_width + padding_x,
                              col_idx * model_output_width + model_input_width - padding_x))
                print(irows, icols)

                if (ibatch) == batch_size:

                    outputs = model.predict(mb_array)
                    for i in range(batch_size):
                        r0, r1 = irows[i]
                        c0, c1 = icols[i]

                        pred_lc_image[r0:r1, c0:c1, :] = outputs[i]
                        mask[r0:r1, c0:c1] = 0

                    ibatch = 0
                    irows, icols = [], []

    # if ibatch > 0:
    #     outputs = model.predict(mb_array)
    #     for i in range(ibatch):
    #         r0, r1 = irows[i]
    #         c0, c1 = icols[i]
    #
    #         pred_lc_image[r0:r1, c0:c1, :] = outputs[i]
    #         mask[r0:r1, c0:c1] = 0

    ## for softmax with 2 neurons:
    label_image = np.array(pred_lc_image[:,:,1])
    


    print('pred_lc_image.shape is :', pred_lc_image.shape)
    print('label_image.shape is :', label_image.shape)
    imshow(np.array(label_image))
    plt.show()
    res3 = list(set(i for j in np.array(label_image) for i in j))
    print('res3: ', res3)
    save_image(label_image, output_image_path)


## Evaluate Images in Folder¶
def evaluate(input_dir, model_path, output_dir):
    model = load_model(model_path, custom_objects={'mean_iou': mean_iou,
                                                   'weighted_categorical_crossentropy': weighted_categorical_crossentropy})

    for root, dirs, files in os.walk(input_dir):
        # print(root, dirs, files)
        if not files: continue
        for f in files:
            pth = os.path.join(root, f)
            # print(pth)
            out_pth = os.path.join(output_dir, 'Predicted_2004.tif')
            # print(out_pth)
            eval_image(pth, model, out_pth)
            print('saved result to ' + out_pth)


input_image_dir = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Drivers\input_Image\input_9depth'
model = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\UnetModel\ModelWithBatch_epochs_05.h5'
output_image_dir = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Drivers\input_Image\results_input_9depth'
evaluate(input_image_dir, model, output_image_dir)

############################################################################################################################################################################
import os
import gdal
import numpy as np
from tqdm import tqdm
import h5py
import keras
from keras import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model, load_model
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from IPython.display import HTML, display
import tabulate
from PIL import Image
import tifffile
from skimage import img_as_uint
import scipy.io as io

## Chip Size:128*128

## Land Cover Classes
table = [["Non_Urban", 0], ["Urban", 1]]
display(HTML(tabulate.tabulate(table, tablefmt='html')))

## Number of Land Cover Classes
number_of_classes = 2
color_map = [[250, 0, 0], [0, 0, 250]]


## Define IoU Metric
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


# ## Define Custom Loss Function
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


## Save GeoTIFF Image
def save_image(image_data, path):
    driver = gdal.GetDriverByName('GTiff')

    # Set Info of Image
    height, width = image_data.shape
    dataset = driver.Create(path, width, height, 1, gdal.GDT_Byte)
    # dataset.SetGeoTransform(geo_transform)
    # dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(image_data)

    # Create Color Table
    color_table = gdal.ColorTable()

    for i in range(number_of_classes):
        color_table.SetColorEntry(i, tuple(color_map[i]) + (255,))
    dataset.GetRasterBand(1).SetRasterColorTable(color_table)
    dataset.GetRasterBand(1).SetNoDataValue(255)

    dataset.FlushCache()


def eval_image(input_image_path, model, output_image_path):
    input_dataset = h5py.File(input_image_path, 'r')
    input_Image = np.array(input_dataset.get('Normalized_2004_Image_With_All_Drivers')).astype(np.float32)
    print(input_Image.shape)
    print(np.amin(input_Image), np.amax(input_Image))

    # input_dataset = gdal.Open(input_image_path)
    # input_image = input_dataset.ReadAsArray().astype(np.float32)
    # input_image = np.expand_dims(input_image, axis=2)

    # print(input_image.shape)
    # input_image = np.rollaxis(input_image, 0, 3)
    h, w, n = input_Image.shape
    # print(input_image.shape)

    model_input_height, model_input_width, model_input_channels = model.layers[0].input_shape[1:4]
    print(model_input_height, model_input_width, model_input_channels)
    model_output_height, model_output_width, model_output_channels = model.layers[len(model.layers) - 1].output_shape[
                                                                     1:4]
    print(model_output_height, model_output_width, model_output_channels)

    padding_y = int((model_input_height - model_output_height) / 2)
    padding_x = int((model_input_width - model_output_width) / 2)
    print(padding_y, padding_x)
    assert model_output_channels == number_of_classes

    pred_lc_image = np.zeros((h, w, number_of_classes))
    print(pred_lc_image.shape)
    mask = np.ones((h, w))
    print(mask.shape)

    irows, icols = [], []
    # batch_size = 80
    minibatch = []
    ibatch = 0

    n_rows = int(h / model_output_height)
    print(n_rows)
    n_cols = int(w / model_output_width)
    print(n_cols)
    batch_size = (n_rows * n_cols) - 1

    mb_array = np.zeros((batch_size, model_input_width, model_input_height, model_input_channels))
    print(mb_array.shape)

    for row_idx in tqdm(range(n_rows)):
        for col_idx in range(n_cols):

            subimage = input_Image[row_idx * model_output_height:row_idx * model_output_height + model_input_height,
                       col_idx * model_output_width:col_idx * model_output_width + model_input_width, :]
            ## /256 for normalized
            print(subimage.shape)
            print(np.amin(subimage), np.amax(subimage))
            # imshow(np.array(subimage[:,:,1]))
            # plt.show()

            if (subimage.shape == model.layers[0].input_shape[1:4]):

                mb_array[ibatch] = subimage
                ibatch += 1
                irows.append((row_idx * model_output_height + padding_y,
                              row_idx * model_output_height + model_input_height - padding_y))
                icols.append((col_idx * model_output_width + padding_x,
                              col_idx * model_output_width + model_input_width - padding_x))
                print(irows, icols)

                if (ibatch) == batch_size:

                    outputs = model.predict(mb_array)
                    for i in range(batch_size):
                        r0, r1 = irows[i]
                        c0, c1 = icols[i]

                        pred_lc_image[r0:r1, c0:c1, :] = outputs[i]
                        mask[r0:r1, c0:c1] = 0

                    ibatch = 0
                    irows, icols = [], []

    # if ibatch > 0:
    #     outputs = model.predict(mb_array)
    #     for i in range(ibatch):
    #         r0, r1 = irows[i]
    #         c0, c1 = icols[i]
    #
    #         pred_lc_image[r0:r1, c0:c1, :] = outputs[i]
    #         mask[r0:r1, c0:c1] = 0

    ## for softmax with 2 neurons:
    label_image = np.ma.array(pred_lc_image.argmax(axis=-1), mask=mask)
    label_image = np.ma.array(pred_lc_image.argmax(axis=-1), mask=mask)

    print('pred_lc_image.shape is :', pred_lc_image.shape)
    print('label_image.shape is :', label_image.shape)
    imshow(np.array(label_image))
    plt.show()
    res3 = list(set(i for j in np.array(label_image) for i in j))
    print('res3: ', res3)
    save_image(label_image, output_image_path)


## Evaluate Images in Folder¶
def evaluate(input_dir, model_path, output_dir):
    model = load_model(model_path, custom_objects={'mean_iou': mean_iou,
                                                   'weighted_categorical_crossentropy': weighted_categorical_crossentropy})

    for root, dirs, files in os.walk(input_dir):
        # print(root, dirs, files)
        if not files: continue
        for f in files:
            pth = os.path.join(root, f)
            # print(pth)
            out_pth = os.path.join(output_dir, 'Predict_with_?Model_For2004.tif')
            # print(out_pth)
            eval_image(pth, model, out_pth)
            print('saved result to ' + out_pth)


input_image_dir = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Drivers\input_Image\input_9depth'
model = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\UnetModel\ModelWithBatch_epochs_05.h5'
output_image_dir = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Drivers\input_Image\results_input_9depth'
evaluate(input_image_dir, model, output_image_dir)



