#!/usr/bin/env python3

import glob
import os
import time
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import seaborn as sns


CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
COL = 160
ROW = 90
BATCH_SIZE = 128
NB_EPOCH = 300


def load_train_data(filepath):
    print('Loading training data...')
    x = []
    y = []
    for fd in CLASSES:
        yt = CLASSES.index(fd)
        files = glob.glob(os.path.join(filepath, fd, '*.jpg'))
        for fl in files:
            x.append(img_to_array(load_img(fl)))
            y.append(yt)
    x = np.array(x) / 255
    y = to_categorical(y)
    print('%d samples of training data loaded.' % x.shape[0])
    return shuffle(x, y, random_state=1234)


def load_test_data(filepath):
    print('Loading test data...')
    x = []
    ids = []
    for fl in glob.glob(os.path.join(filepath, '*.jpg')):
        x.append(img_to_array(load_img(fl)))
        ids.append(os.path.basename(fl))
    x = np.array(x) / 255
    print('%d samples of test data loaded.' % x.shape[0])
    return np.array(ids).reshape(-1, 1), x


def build_model():
    print('Creating the model...')

    model = Sequential()

    model.add(Convolution2D(8, 3, 3, border_mode='same', activation='relu', input_shape=(3, ROW, COL)))
    model.add(Convolution2D(8, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print('Model created.')
    return model


if __name__ == '__main__':

    x_train, y_train = load_train_data('data/train_resized')
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=0.2)
    model = build_model()

    datagen = ImageDataGenerator(horizontal_flip=True,
                                 vertical_flip=True,
                                 rotation_range=45,
                                 zoom_range=0.3)
    flow = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

    mcp = ModelCheckpoint('data/naive_cnn_best.h5',
                          monitor='val_loss', verbose=1,
                          save_best_only=True, save_weights_only=True)
    history =  model.fit_generator(flow, nb_epoch=NB_EPOCH,
                                   samples_per_epoch=x_train.shape[0],
                                   validation_data=(x_val, y_val),
                                   verbose=1, callbacks=[mcp])

    sns.set_style('whitegrid')
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(train_loss, linewidth=1, label='train loss')
    plt.plot(val_loss, linewidth=1, label='val loss')
    plt.savefig('loss.png')
    model.load_weights('data/naive_cnn_best.h5')
    ids, test = load_test_data('data/test_stg1_resized')
    print('Making prediction...')
    pred = np.round(model.predict(test), 9)
    sub = pd.DataFrame(np.hstack((ids, pred)), columns=['image'] + CLASSES)
    sub.to_csv('data/naive_cnn_res.csv', index=False)
    print('done.')
