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
def save_image(image_data, path, geo_transform, projection):
    driver = gdal.GetDriverByName('GTiff')

    # Set Info of Image
    height, width = image_data.shape
    dataset = driver.Create(path, width, height, 1, gdal.GDT_Float64 )
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(image_data[:,:])

    # # Create Color Table
    # color_table = gdal.ColorTable()
    #
    # for i in range(number_of_classes):
    #     color_table.SetColorEntry(i, tuple(color_map[i]) + (255,))
    # dataset.GetRasterBand(1).SetRasterColorTable(color_table)
    # dataset.GetRasterBand(1).SetNoDataValue(255)

    dataset.FlushCache()

def eval_image(input_image_path, model, output_image_path):
    # input_dataset = h5py.File(input_image_path, 'r')
    # input_Image = np.array(input_dataset.get('Normalized_2004_Image_With_All_Drivers')).astype(np.float32)
    # print(input_Image.shape)
    # print(np.amin(input_Image), np.amax(input_Image))
    # input_Image = np.pad(input_Image, ((0, 0), (0, 6), (0, 0)), 'minimum')
    # print(input_Image.shape)
    # print(np.amin(input_Image), np.amax(input_Image))

    input_dataset = gdal.Open(input_image_path)
    input_Image = input_dataset.ReadAsArray().astype(np.float32)
    input_Image = np.rollaxis(input_Image, 0, 3)
    print(np.amin(input_Image), np.amax(input_Image))
    print(input_Image.shape)

    ## Normalization
    print(np.amin(input_Image), np.amax(input_Image))
    for i in range(input_Image.shape[2]):
        Min = np.amin(input_Image[:, :, i])
        Max = np.amax(input_Image[:, :, i])
        input_Image[:, :, i] = (input_Image[:, :, i] - Min) / (Max - Min)
    print(np.amin(input_Image), np.amax(input_Image))
    input_Image = np.pad(input_Image, ((0, 0), (0, 6), (0, 0)), 'minimum')
    print(input_Image.shape)
    h, w, n = input_Image.shape
    model_input_height, model_input_width, model_input_channels = model.layers[0].input_shape[1:4]
    print(model_input_height, model_input_width, model_input_channels)
    model_output_height, model_output_width, model_output_channels = model.layers[len(model.layers) - 1].output_shape[1:4]
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
    ibatch = 0

    n_rows = int(h / model_output_height)
    print(n_rows)
    n_cols = int(w / model_output_width)
    print(n_cols)
    batch_size = (n_rows * n_cols)

    mb_array = np.zeros((batch_size, model_input_width, model_input_height, model_input_channels))
    print(mb_array.shape)

    for row_idx in tqdm(range(n_rows)):
        for col_idx in range(n_cols):

            subimage = input_Image[row_idx * model_output_height:row_idx * model_output_height + model_input_height,
                       col_idx * model_output_width:col_idx * model_output_width + model_input_width, :]
            ## /256 for normalized
            print(subimage.shape)
            print(np.amin(subimage), np.amax(subimage))

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
    # label_image_0 = np.array(pred_lc_image[:, :, 0])
    mask = mask [:, :1274]
    label_image_1 = np.array(pred_lc_image[:, :1274, 1])
    # print('label_image_0:', label_image_0,
    #       np.mean(label_image_0))
    print('label_image_1:', label_image_1,
          np.mean(label_image_1),
          np.amax(label_image_1),
          np.amin(label_image_1))

    # label_image_1[label_image_1 > np.mean(label_image_1)] = 1
    # label_image_1[label_image_1 <= np.mean(label_image_1)] = 0
    print('label_image_1.shape:', label_image_1.shape)

    # print('pred_lc_image.shape :', pred_lc_image.shape)
    # print('label_image_1.shape :', label_image_1.shape)
    # label_image_1 = label_image_1[:, :1274]
    # mask = mask[:, :1274]
    print('mask:', mask)
    print('mask.shape:', mask.shape)
    # print('label_image_1.shape finall :', label_image_1.shape)
    imshow(np.array(label_image_1))
    plt.show()
    # unique, counts = np.unique(label_image_1, return_counts=True)
    # freq = np.asarray((unique, counts)).T
    # print(freq)
    # res3 = list(set(i for j in np.array(label_image_1) for i in j))
    # print('res3_label: ', res3)
    # res2 = list(set(i for j in np.array(mask) for i in j))
    # print('res2_mask: ', res2)
    # save_image(mask, output_image_path, input_dataset.GetGeoTransform(), input_dataset.GetProjection())
    save_image(label_image_1, output_image_path, input_dataset.GetGeoTransform(), input_dataset.GetProjection())

## Evaluate Images in Folder¶
def evaluate(input_dir, model_path, output_dir):
    model = load_model(model_path, custom_objects={'mean_iou': mean_iou, 'weighted_categorical_crossentropy': weighted_categorical_crossentropy})
    # model = load_model(model_path, custom_objects={'dice_coef_loss':dice_coef_loss, 'dice_coef': dice_coef, 'iou': iou, 'iou_loss': iou_loss, 'mean_iou': mean_iou, 'weighted_categorical_crossentropy': weighted_categorical_crossentropy})

    for root, dirs, files in os.walk(input_dir):
        # print(root, dirs, files)
        if not files: continue
        for f in files:
            pth = os.path.join(root, f)
            # print(pth)
            out_pth = os.path.join(output_dir, 'Model_1.tif')
            # print(out_pth)
            eval_image(pth, model, out_pth)
            print('saved result to ' + out_pth)


input_image_dir = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\input_Image\2004_LayerStacked'
model = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\UnetModel\Model\Model_1.h5'
output_image_dir = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Predict_2004To2014\New\model_1'
evaluate(input_image_dir, model, output_image_dir)


############################################################################################################################################################################
# import os
# import gdal
# import numpy as np
# from tqdm import tqdm
# import h5py
# import keras
# from keras import Input
# from keras.layers.convolutional import Conv2D, Conv2DTranspose
# from keras.layers.core import Dropout
# from keras.layers.pooling import MaxPooling2D
# from keras.layers.merge import concatenate
# from keras.models import Model, load_model
# import keras.backend as K
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow
# from IPython.display import HTML, display
# import tabulate
# from PIL import Image
# from skimage import img_as_uint
# import scipy.io as io
#
# ## Chip Size:128*128
#
# ## Land Cover Classes
# table = [["Non_Urban", 0], ["Urban", 1]]
# display(HTML(tabulate.tabulate(table, tablefmt='html')))
#
# ## Number of Land Cover Classes
# number_of_classes = 2
# color_map = [[250, 0, 0], [0, 0, 250]]
#

# ## Define IoU Metric
# def mean_iou(y_true, y_pred):
#     prec = []
#     for t in np.arange(0.5, 1.0, 0.05):
#         y_pred_ = tf.to_int32(y_pred > t)
#         score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
#         K.get_session().run(tf.local_variables_initializer())
#         with tf.control_dependencies([up_opt]):
#             score = tf.identity(score)
#         prec.append(score)
#     return K.mean(K.stack(prec), axis=0)
#
# def iou(actual, predicted):
#     """Compute Intersection over Union statistic (i.e. Jaccard Index)
#     See https://en.wikipedia.org/wiki/Jaccard_index
#     Parameters
#     ----------
#     actual : list
#         Ground-truth labels
#     predicted : list
#         Predicted labels
#     Returns
#     -------
#     float
#         Intersection over Union value
#     """
#     actual = K.flatten(actual)
#     predicted = K.flatten(predicted)
#     intersection = K.sum(actual * predicted)
#     union = K.sum(actual) + K.sum(predicted) - intersection
#     print(intersection, type(intersection))
#     print(union, type(union))
#     return 1. * intersection / union
#
# def iou_loss(actual, predicted):
#     """Loss function based on the Intersection over Union (IoU) statistic
#     IoU is comprised between 0 and 1, as a consequence the function is set as
#     `f(.)=1-IoU(.)`: the loss has to be minimized, and is comprised between 0
#     and 1 too
#     Parameters
#     ----------
#     actual : list
#         Ground-truth labels
#     predicted : list
#         Predicted labels
#     Returns
#     -------
#     float
#         Intersection-over-Union-based loss
#     """
#     return 1. - iou(actual, predicted)
#
# def dice_coef(actual, predicted, eps=1e-3):
#     """Dice coef
#     See https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
#     Examples at:
#       -
#     https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L23
#       -
#     https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model/blob/master/zf_unet_224_model.py#L36
#     Parameters
#     ----------
#     actual : list
#         Ground-truth labels
#     predicted : list
#         Predicted labels
#     eps : float
#         Epsilon value to add numerical stability
#     Returns
#     -------
#     float
#         Dice coef value
#     """
#     y_true_f = K.flatten(actual)
#     y_pred_f = K.flatten(predicted)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + eps) / (K.sum(y_true_f) + K.sum(y_pred_f) + eps)
#
# def dice_coef_loss(actual, predicted):
#     """
#     Parameters
#     ----------
#     actual : list
#         Ground-truth labels
#     predicted : list
#         Predicted labels
#     Returns
#     -------
#     float
#         Dice-coef-based loss
#     """
#     return -dice_coef(actual, predicted)
#
# # ## Define Custom Loss Function
# class_weights = np.array([1, 1])
# weights = K.variable(class_weights)
#
#
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
#
# ## Save GeoTIFF Image
# def save_image(image_data, path):
#     driver = gdal.GetDriverByName('GTiff')
#
#     # Set Info of Image
#     height, width = image_data.shape
#     dataset = driver.Create(path, width, height, 1, gdal.GDT_Byte)
#     # dataset.SetGeoTransform(geo_transform)
#     # dataset.SetProjection(projection)
#     dataset.GetRasterBand(1).WriteArray(image_data)
#
#     # Create Color Table
#     color_table = gdal.ColorTable()
#
#     for i in range(number_of_classes):
#         color_table.SetColorEntry(i, tuple(color_map[i]) + (255,))
#     dataset.GetRasterBand(1).SetRasterColorTable(color_table)
#     dataset.GetRasterBand(1).SetNoDataValue(255)
#
#     dataset.FlushCache()
#
#
#
# def eval_image(input_image_path, model, output_image_path):
#     input_dataset = h5py.File(input_image_path, 'r')
#     input_Image = np.array(input_dataset.get('Normalized_2004_Image_With_All_Drivers')).astype(np.float32)
#     print(input_Image.shape)
#     print(np.amin(input_Image), np.amax(input_Image))
#     input_Image = np.pad(input_Image, ((0, 0), (0, 6), (0, 0)), 'minimum')
#     print(input_Image.shape)
#     print(np.amin(input_Image), np.amax(input_Image))
#     # input_dataset = gdal.Open(input_image_path)
#     # input_image = input_dataset.ReadAsArray().astype(np.float32)
#     # input_image = np.expand_dims(input_image, axis=2)
#
#     # print(input_image.shape)
#     # input_image = np.rollaxis(input_image, 0, 3)
#     h, w, n = input_Image.shape
#     print(h, w, n)
#
#     model_input_height, model_input_width, model_input_channels = model.layers[0].input_shape[1:4]
#     print(model_input_height, model_input_width, model_input_channels)
#     model_output_height, model_output_width, model_output_channels = model.layers[len(model.layers) - 1].output_shape[1:4]
#     print(model_output_height, model_output_width, model_output_channels)
#
#     padding_y = int((model_input_height - model_output_height) / 2)
#     padding_x = int((model_input_width - model_output_width) / 2)
#     print(padding_y, padding_x)
#     assert model_output_channels == number_of_classes
#
#     pred_lc_image = np.zeros((h, w, number_of_classes))
#     print(pred_lc_image.shape)
#     mask = np.ones((h, w))
#     print(mask.shape)
#
#     irows, icols = [], []
#     # batch_size = 80
#     minibatch = []
#     ibatch = 0
#
#     n_rows = int(h / model_output_height)
#     print(n_rows)
#     n_cols = int(w / model_output_width)
#     print(n_cols)
#     batch_size = (n_rows * n_cols)
#
#     mb_array = np.zeros((batch_size, model_input_width, model_input_height, model_input_channels))
#     print(mb_array.shape)
#
#     for row_idx in tqdm(range(n_rows)):
#         for col_idx in range(n_cols):
#
#             subimage = input_Image[row_idx * model_output_height:row_idx * model_output_height + model_input_height,
#                        col_idx * model_output_width:col_idx * model_output_width + model_input_width, :]
#             ## /256 for normalized
#             print(subimage.shape)
#             print(np.amin(subimage), np.amax(subimage))
#             # imshow(np.array(subimage[:,:,1]))
#             # plt.show()
#
#             if (subimage.shape == model.layers[0].input_shape[1:4]):
#
#                 mb_array[ibatch] = subimage
#                 ibatch += 1
#                 irows.append((row_idx * model_output_height + padding_y,
#                               row_idx * model_output_height + model_input_height - padding_y))
#                 icols.append((col_idx * model_output_width + padding_x,
#                               col_idx * model_output_width + model_input_width - padding_x))
#                 print(irows, icols)
#
#                 if (ibatch) == batch_size:
#
#                     outputs = model.predict(mb_array)
#                     for i in range(batch_size):
#                         r0, r1 = irows[i]
#                         c0, c1 = icols[i]
#
#                         pred_lc_image[r0:r1, c0:c1, :] = outputs[i]
#                         mask[r0:r1, c0:c1] = 0
#
#                     ibatch = 0
#                     irows, icols = [], []
#
#     # if ibatch > 0:
#     #     outputs = model.predict(mb_array)
#     #     for i in range(ibatch):
#     #         r0, r1 = irows[i]
#     #         c0, c1 = icols[i]
#     #
#     #         pred_lc_image[r0:r1, c0:c1, :] = outputs[i]
#     #         mask[r0:r1, c0:c1] = 0
#
#     ## for softmax with 2 neurons:
#
#     label_image_0 = np.array(pred_lc_image[:, :, 0])
#     label_image_1 = np.array(pred_lc_image[:, :, 1])
#     print('label_image_0:', label_image_0,
#           np.mean(label_image_0))
#     print('label_image_1:', label_image_1,
#           np.mean(label_image_1),
#           np.amax(label_image_1))
#
#     label_image_1[label_image_1 > 0.20] = 1
#     label_image_1[label_image_1 <= 0.20] = 0
#     print('label_image_1:', label_image_1)
#     print('label_image_1.shape:', label_image_1.shape)
#     label_image_1 = label_image_1[:, :1274]
#     print('label_image_1.shape after change:', label_image_1.shape)
#     imshow(np.array(label_image_1))
#     plt.show()
#     res3 = list(set(i for j in np.array(label_image_1) for i in j))
#     print('res3: ', res3)
# ##
#     y_true = np.array(Image.open(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\ChangeDetection(Label)\Prediction_Label\Prediction_Label.tif'))
#     y_pred = np.array(label_image_1)
#
#     Confusion_Matrix = np.zeros(shape=y_true.shape)
#     for i in range(0, y_true.shape[0]):
#         for j in range(0, y_true.shape[1]):
#             if y_true[i, j] == 1 and y_pred[i, j] == 1:  # Urban-->Urban
#                 Confusion_Matrix[i, j] = 0  # Hits
#             elif y_true[i, j] == 0 and y_pred[i, j] == 1:  # NonUrban-->Urban
#                 Confusion_Matrix[i, j] = 1  # False Alarms
#             elif y_true[i, j] == 1 and y_pred[i, j] == 0:  # NotUrban-->Urban
#                 Confusion_Matrix[i, j] = 2  # Misses
#             elif y_true[i, j] == 0 and y_pred[i, j] == 0:
#                 Confusion_Matrix[i, j] = 3  # Correct Rejection
#
#     unique, counts = np.unique(Confusion_Matrix, return_counts=True)
#     freq = np.asarray((unique, counts)).T
#     Hits = freq[0, 1]
#     False_Alarms = freq[1, 1]
#     Misses = freq[2, 1]
#     Correct_Rejection = freq[3, 1]
#
#     print('Hits:', Hits, 'False_Alarms:', False_Alarms, 'Misses:', Misses, 'Correct_Rejection:', Correct_Rejection)
#
#     PA = Hits / (Hits + Misses)
#     OA = (Hits + Correct_Rejection) / (y_true.shape[0] * y_true.shape[1])
#     FOM = Correct_Rejection / (Correct_Rejection + False_Alarms + Misses)
#     print('PA =', PA, 'OA =', OA, 'FOM:', FOM)
# ##
#     save_image(label_image_1, output_image_path)
#
#
#
# ## Evaluate Images in Folder¶
# def evaluate(input_dir, model_path, output_dir):
#     model = load_model(model_path, custom_objects={'dice_coef': dice_coef, 'iou': iou, 'iou_loss': iou_loss, 'mean_iou': mean_iou, 'weighted_categorical_crossentropy': weighted_categorical_crossentropy})
#
#     for root, dirs, files in os.walk(input_dir):
#         # print(root, dirs, files)
#         if not files: continue
#         for f in files:
#             pth = os.path.join(root, f)
#             # print(pth)
#             out_pth = os.path.join(output_dir, 'Predict_with_ModelWithBatch_epochs_05_WithoutTh.tif')
#             # print(out_pth)
#             eval_image(pth, model, out_pth)
#             print('saved result to ' + out_pth)
#
# input_image_dir = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Drivers\input_Image\input_9depth'
# model = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\UnetModel\ModelWithBatch_epochs_05.h5'
# output_image_dir = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Predict_2004To2014\ModelWithBatch_epochs_05'
# evaluate(input_image_dir, model, output_image_dir)
#
