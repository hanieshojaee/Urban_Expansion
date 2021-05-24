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
from keras import layers
import gdal
from PIL import Image
from skimage.io import imread, imshow, imread_collection, concatenate_images
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from keras.layers import Activation


## Load data
All_Images_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\Train(1994to2004)\Drivers(1994)\HDF_256x256\Stride64x64\New_Method(New3Classes)\Th50_forLabel3\All_Images_augment6.h5', 'r')
All_Images = np.array(All_Images_hf.get('All_Images_augment6')).astype(np.float32)
print(All_Images.shape)
np.random.shuffle(All_Images)

## Separate Label Dim from Rest of Array
Images = All_Images[:, :, :, 0:9]
print(Images.shape, np.amin(), np.amax())
Labels = All_Images[:, :, :, 9:10]
print(Labels.shape, np.amin(), np.amax())



## Model
# Define Inputs and Targets Dim
# Set parameters
Img_Width    = 16
Img_Height   = 16
Img_Channels = 9
Num_Classes  = 3

inputs  = Input((Img_Height, Img_Width, Img_Channels))
print(inputs.shape)
targets = Input((Img_Height, Img_Width, Num_Classes))
print(targets.shape)
def data():
    x_train = Images
    y_train = Labels
    return x_train, y_train

def create_model(Images,Labels):
    # Model : UNet
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
    class_weights = np.array([{{uniform(0, 1)}}, {{uniform(0, 1)}}, {{uniform(0, 1)}}])

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

    c1 = Conv2D(16, (3, 3), kernel_initializer = 'he_normal', padding = 'same') (inputs)
    c1 = Activation({{choice(['relu', 'elu'])}})(c1)
    c1 = Dropout({{uniform(0, 1)}})(c1)
    c1 = Conv2D(16, (3, 3), kernel_initializer = 'he_normal', padding = 'same') (c1)
    c1 = Activation({{choice(['relu', 'elu'])}})(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), kernel_initializer = 'he_normal', padding = 'same') (p1)
    c2 = Activation({{choice(['relu', 'elu'])}})(c2)
    c2 = Dropout({{uniform(0, 1)}})(c2)
    c2 = Conv2D(32, (3, 3), kernel_initializer = 'he_normal', padding = 'same') (c2)
    c2 = Activation({{choice(['relu', 'elu'])}})(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), kernel_initializer = 'he_normal', padding = 'same') (p2)
    c3 = Activation({{choice(['relu', 'elu'])}})(c3)
    c3 = Dropout({{uniform(0, 1)}})(c3)
    c3 = Conv2D(64, (3, 3), kernel_initializer = 'he_normal', padding = 'same') (c3)
    c3 = Activation({{choice(['relu', 'elu'])}})(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), kernel_initializer = 'he_normal', padding = 'same') (p3)
    c4 = Activation({{choice(['relu', 'elu'])}})(c4)
    c4 = Dropout({{uniform(0, 1)}})(c4)
    c4 = Conv2D(128, (3, 3), kernel_initializer = 'he_normal', padding = 'same') (c4)
    c4 = Activation({{choice(['relu', 'elu'])}})(c4)
    p4 = MaxPooling2D(pool_size = (2, 2))(c4)

    c5 = Conv2D(256, (3, 3), kernel_initializer = 'he_normal', padding = 'same') (p4)
    c5 = Activation({{choice(['relu', 'elu'])}})(c5)
    c5 = Dropout({{uniform(0, 1)}})(c5)
    c5 = Conv2D(256, (3, 3), kernel_initializer = 'he_normal', padding = 'same') (c5)
    c5 = Activation({{choice(['relu', 'elu'])}})(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides = (2, 2), padding = 'same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), kernel_initializer = 'he_normal', padding = 'same') (u6)
    c6 = Activation({{choice(['relu', 'elu'])}})(c6)
    c6 = Dropout({{uniform(0, 1)}}) (c6)
    c6 = Conv2D(128, (3, 3), kernel_initializer = 'he_normal', padding = 'same') (c6)
    c6 = Activation({{choice(['relu', 'elu'])}})(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides = (2, 2), padding = 'same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), kernel_initializer = 'he_normal', padding = 'same') (u7)
    c7 = Activation({{choice(['relu', 'elu'])}})(c7)
    c7 = Dropout({{uniform(0, 1)}}) (c7)
    c7 = Conv2D(64, (3, 3), kernel_initializer = 'he_normal', padding = 'same') (c7)
    c7 = Activation({{choice(['relu', 'elu'])}})(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides = (2, 2), padding = 'same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), kernel_initializer = 'he_normal', padding = 'same') (u8)
    c8 = Activation({{choice(['relu', 'elu'])}})(c8)
    c8 = Dropout({{uniform(0, 1)}}) (c8)
    c8 = Conv2D(32, (3, 3), kernel_initializer = 'he_normal', padding = 'same') (c8)
    c8 = Activation({{choice(['relu', 'elu'])}})(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides = (2, 2), padding = 'same') (c8)
    u9 = concatenate([u9, c1], axis = 3)
    c9 = Conv2D(16, (3, 3), kernel_initializer = 'he_normal', padding = 'same') (u9)
    c9 = Activation({{choice(['relu', 'elu'])}})(c9)
    c9 = Dropout({{uniform(0, 1)}}) (c9)
    c9 = Conv2D(16, (3, 3), kernel_initializer = 'he_normal', padding = 'same') (c9)
    c9 = Activation({{choice(['relu', 'elu'])}})(c9)

    outputs = Conv2D(3, (1, 1), activation = 'softmax') (c9)
    print(outputs.shape)


    model = Model(inputs = [inputs], outputs = [outputs])
    model.compile(optimizer=Adam(lr={{uniform(10e-6,1)}}), loss=weighted_categorical_crossentropy,   metrics=[mean_iou])
    model.summary()

    # model_hf = h5py.File(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\UnetModel\Model\New_3Classes\ChipSize16_Stride4\model_25.h5', 'w')
    # model_hf.close()
    # earlystopper = EarlyStopping(patience = 5, verbose = 1)
    # checkpointer = ModelCheckpoint(r'E:\Education\MSc\thesis\UrbanExpansion\Data\UrbanExpansion\UnetModel\Model\New_3Classes\ChipSize16_Stride4\model_25.h5', verbose = 1, save_best_only=True)
    results = model.fit(Images, Labels, validation_split = 0.10, batch_size = {{choice([8, 32, 64, 128])}}, epochs = 10, shuffle=True)
    validation_mean_iou = np.amax(results.history['val_mean_iou'])
    validation_loss = np.amin(results.history['val_loss'])
    print('Best mean_iou of epoch:', validation_mean_iou,
          'Best Loss of epoch:', validation_loss)
    return {'loss': validation_loss, 'status': STATUS_OK, 'model': model}

best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())