from model import CreateModel
from utils import loss_box, meanIoU

import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from livelossplot import PlotLossesKeras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile
import tensorflow as tf

def train_model():
    cnn = CreateModel('MobileNetV2')
    my_model =  cnn.assemble_model()

    DATASET_ZIP_PATH = 'data/idchess_zadanie.zip'
    DATASET_DIR_PATH = 'data/idchess_zadanie'
    BATCH_SIZE = 100

    # with zipfile.ZipFile(DATASET_ZIP_PATH, 'r') as zip_ref:
    #     zip_ref.extractall(DATASET_DIR_PATH)

    x_train_initial = np.load('data/idchess_zadanie/xtrain.npy')
    y_train_initial = np.load('data/idchess_zadanie/ytrain.npy')

    x_train, x_valid, y_train, y_valid = train_test_split(x_train_initial, y_train_initial, test_size=0.2, random_state = 11)

    del(x_train_initial)
    del(y_train_initial)

    imageDataGenerator = ImageDataGenerator(preprocessing_function=preprocess_input)
    iterator_train = imageDataGenerator.flow(x=x_train, y=y_train, batch_size=BATCH_SIZE)
    iterator_valid = imageDataGenerator.flow(x=x_valid, y=y_valid, batch_size=BATCH_SIZE)

    adam = tf.keras.optimizers.Adam(learning_rate=0.01)
    my_model.compile(optimizer=adam, loss=loss_box, metrics=[meanIoU])

    my_model.fit(iterator_train, epochs=25, callbacks=[PlotLossesKeras()],
                validation_data=iterator_valid)

    # my_model.save('/GDrive/MyDrive/freefleks/saved_models/model_1')
    return my_model