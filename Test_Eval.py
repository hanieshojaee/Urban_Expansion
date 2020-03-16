import numpy as np
import h5py
from keras.models import Model, load_model
import keras.backend as K
import tensorflow as tf
from skimage.io import imread, imshow, imread_collection, concatenate_images
import matplotlib.pyplot as plt

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

## Define Custom Loss Function
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

## Prediction
model = load_model(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\UnetModel\model_epochs_05.h5',
    custom_objects={'mean_iou': mean_iou, 'weighted_categorical_crossentropy': weighted_categorical_crossentropy})
#
Predict_Images = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Drivers\HDF5files\All_Images.h5', 'r')
Predict_Images = np.array(Predict_Images.get('All_Images')).astype(np.float32)
print(Predict_Images.shape)

predict_prob_Predict = model.predict(Predict_Images[:, :, :, 0:9])
print(predict_prob_Predict.shape)
predict_class_Predict = predict_prob_Predict.argmax(axis = -1)
print(predict_class_Predict.shape)

num = 10
print(Predict_Images[num, :, :, :].shape)
imshow(np.reshape(Predict_Images[num,:,:,-1], (256, 256)))
plt.show()

print(predict_class_Predict[num, :, :].shape)
imshow(predict_class_Predict[num, :, :])
plt.show()
#
Train_Images = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers\HDF5files\All_Images.h5', 'r')
Train_Images = np.array(Train_Images.get('All_Images')).astype(np.float32)

predict_prob_Train = model.predict(Train_Images[0:20, :, :, 0:9])
print(predict_prob_Train.shape)
predict_class_Train = predict_prob_Train.argmax(axis = -1)
print(predict_class_Train.shape)

num = 10
print(Train_Images[num, :, :, :].shape)
imshow(np.reshape(Train_Images[num,:,:,-1], (256, 256)))
plt.show()

print(predict_class_Train[num, :, :].shape)
imshow(predict_class_Train[num, :, :])
plt.show()

#######
dark = 0
rgbn = 0

more_50_cover = 0
less_50_cover = 0
sample_with_zero = 0
sample_no_zero = 0

count_lc_class = []


folder_paths = r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Predict(2004to2014)\Drivers\Chips'

for folders in folder_paths:
    print(folders)

    for items in os.listdir((os.path.join(folders, 'labels'))):
        if items.endswith(".tif"):

            output_image_path = os.path.join(os.path.join(folders, 'labels'), items)
            output_image = gdal.Open(output_image_path)
            output_image_band = output_image.GetRasterBand(1)

            if (output_image_band.GetStatistics(True, True)[1] == 0):
                dark = dark + 1
            else:
                label_image = np.array(output_image.GetRasterBand(1).ReadAsArray())
                unique, counts = np.unique(label_image, return_counts=True)
                freq = np.asarray((unique, counts)).T

                # Check Image is dark or not
                if (0 in unique):
                    sample_with_zero = sample_with_zero + 1
                    if (freq[numpy.where(unique == 0)[0][0], 1] > 128 * 128 / 2):
                        less_50_cover = less_50_cover + 1
                    else:
                        more_50_cover = more_50_cover + 1

                        # Read TIFF Image
                        input_image_path = output_image_path.replace('labels', 'images')
                        input_image = gdal.Open(input_image_path)
                        input_image = np.array(input_image.ReadAsArray())
                        # print(input_image.shape)

                        # Add New Dim to Label
                        label_image = np.expand_dims(label_image, axis=0)

                        # Concatenate Image and Mask
                        merge = np.concatenate((input_image, label_image), axis=0)

                        # Add Image to List
                        if (1 in unique):
                            Building.append(merge)
                        if (2 in unique):
                            Road.append(merge)
                        if (3 in unique):
                            Planted.append(merge)
                        if (4 in unique):
                            Water.append(merge)
                        if (5 in unique):
                            Forest.append(merge)
                        if (6 in unique):
                            Harvested.append(merge)

                        for items in unique:
                            count_lc_class.append(items)
                else:
                    sample_no_zero = sample_no_zero + 1

                    input_image_path = output_image_path.replace('labels', 'images')
                    input_image = gdal.Open(input_image_path)
                    input_image = np.array(input_image.ReadAsArray())
                    # print(input_image.shape)

                    # Add New Dim to Label
                    label_image = np.expand_dims(label_image, axis=0)

                    # Concatenate Image and Mask
                    merge = np.concatenate((input_image, label_image), axis=0)

                    # Add Image to List
                    if (1 in unique):
                        Building.append(merge)
                    if (2 in unique):
                        Road.append(merge)
                    if (3 in unique):
                        Water.append(merge)
                    if (4 in unique):
                        Harvested.append(merge)
                    if (5 in unique):
                        Forest.append(merge)
                    if (6 in unique):
                        Planted.append(merge)

                    for items in unique:
                        count_lc_class.append(items)

                rgbn = rgbn + 1