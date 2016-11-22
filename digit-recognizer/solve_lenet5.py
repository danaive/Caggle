import time
import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils


PIXEL = 28
BATCH_SIZE = 128
NB_EPOCH = 30


def load_train_data(filepath):
    print('Loading the training set ...')
    train = pd.read_csv(filepath).values
    train_x = train[:, 1:].reshape(train.shape[0], 1, PIXEL, PIXEL).astype(float)
    train_y = np_utils.to_categorical(train[:, 0])
    print('%d samples loaded' % train_x.shape[0])
    return train_x, train_y


def load_test_data(filepath):
    print('Loading the test set ...')
    test = pd.read_csv(filepath).values.reshape(-1, 1, PIXEL, PIXEL)
    return test.astype(float)


def build_model():
    print('Creating the model ...')
    model = Sequential()

    model.add(Convolution2D(6, 5, 5, activation='sigmoid', border_mode='same', input_shape=(1, PIXEL, PIXEL)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Dropout(0.6))
    model.add(Convolution2D(16, 5, 5, activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(120, 5, 5, activation='sigmoid'))
    model.add(Flatten())

    model.add(Dense(84, activation='sigmoid'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.03, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.summary()
    return model


if __name__ == '__main__':
    cnn = build_model()
    train_x, train_y = load_train_data('data/train.csv')
    test = load_test_data('data/test.csv')
    cnn.fit(train_x, train_y, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, shuffle=True, verbose=2, validation_split=0.3)
    pred = cnn.predict_classes(test)
    sub = pd.read_csv('data/sample_submission.csv')
    sub['Label'] = pred
    sub.to_csv('data/submission_lenet5.csv', index=False)
