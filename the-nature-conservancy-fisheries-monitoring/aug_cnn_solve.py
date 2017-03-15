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
VAL_RATIO = 0.2
TR_PATH = 'data/train_resized'
TEST_PATH = 'data/test_stg1_resized'


def load_train_data(filepath=TR_PATH):

    def load_from_dir(fd):
        files = glob.glob(os.path.join(filepath, fd, '*.jpg'))
        x = []
        for fl in files:
            x.append(img_to_array(load_img(fl)))
        print('- %d samples of %s loaded.' % (len(x), fd))
        return x

    print('Loading training data...')
    x_train, x_val, y_train, y_val = [], [], [], []
    for y, fd in enumerate(CLASSES):
        x_t, x_v = train_test_split(load_from_dir(fd), test_size=VAL_RATIO)
        x_train.extend(x_t)
        x_val.extend(x_v)
        y_train.extend([y] * len(x_t))
        y_val.extend([y] * len(x_v))

    print('%d samples of training data loaded.' % (len(x_train) + len(x_val)))
    x_train = np.array(x_train) / 255
    x_val = np.array(x_val) / 255
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    return x_train, x_val, y_train, y_val


def load_test_data(filepath=TEST_PATH):
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

    # model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    # model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    # model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print('Model created.')
    return model


if __name__ == '__main__':

    x_train, x_val, y_train, y_val = load_train_data()

    model = build_model()

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20,
        zoom_range=0.2)
    train_flow = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

    train_loss = []
    val_loss = []
    min_loss = 10

    print('Fitting...')
    for ep in range(NB_EPOCH):
        ep_loss = []
        batches = 0
        for x, y in train_flow:
            ep_loss.append(model.train_on_batch(x, y))
            batches += BATCH_SIZE
            if batches >= x_train.shape[0]:
                break
        train_loss.append(np.mean(ep_loss))

        loss = model.evaluate(x_val, y_val, verbose=0)
        val_loss.append(loss)
        if min_loss > loss:
            print('- val_loss improved from %.6f to %.6f' % (min_loss, loss))
            min_loss = loss
            model.save_weights('data/aug_cnn_best.h5')

        print('Epoch %3d | train_loss: %.6f | val_loss: %.6f' % (ep, train_loss[-1], val_loss[-1]))

    sns.set_style('whitegrid')
    plt.plot(train_loss, linewidth=1, label='train loss')
    plt.plot(val_loss, linewidth=1, label='val loss')
    plt.savefig('aug_loss.png')
    model.load_weights('data/aug_cnn_best.h5')
    ids, test = load_test_data()
    print('Making prediction...')
    pred = np.round(model.predict(test), 9)
    sub = pd.DataFrame(np.hstack((ids, pred)), columns=['image'] + CLASSES)
    sub.to_csv('data/aug_cnn_res.csv', index=False)
    print('Done.')
