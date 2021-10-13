import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def load_dataset(only_x_and_y_valid = False,
                 only_x_and_y_train = False,
                 preprocess_x_valid = False):
    x_train_initial = np.load('data/idchess_zadanie/xtrain.npy')
    y_train_initial = np.load('data/idchess_zadanie/ytrain.npy')

    x_train, x_valid, y_train, y_valid = train_test_split(x_train_initial, 
                                                          y_train_initial,
                                                          test_size=0.2,
                                                          random_state = 11)
    del(x_train_initial)
    del(y_train_initial)

    if preprocess_x_valid:
        preprocess_input(x_valid) # in-place preprocessing!

    if only_x_and_y_valid:
        del(x_train)
        del(y_train)
        return x_valid, y_valid
    
    if only_x_and_y_train:
        del(x_valid)
        del(y_valid)
        return x_train, y_train

    return x_train, x_valid, y_train, y_valid