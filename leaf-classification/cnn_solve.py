#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt
import seaborn as sns


BATCH_SIZE = 32
IMG_DIM = 128
NB_EPOCH = 100
SPLIT_RATIO = 0.9


def load_train_data(csvpath):
    print('loading the training data...')
    data = pd.read_csv(csvpath)
    size = data.shape[0]
    ids = data.pop('id')
    y = data.pop('species')
    x = data.values
    SS = StandardScaler().fit(x)
    LE = LabelEncoder().fit(y)
    x = SS.transform(x)
    y = to_categorical(LE.transform(y))
    print('%d samples loaded.' % size)
    return SS, LE, ids, x, y


def load_test_data(csvpath):
    print('loading the test data...')
    data = pd.read_csv(csvpath)
    size = data.shape[0]
    index = data.pop('id')
    x = data.values
    print('%d samples loaded.' % size)
    return index, x


def load_img_data(imgpath, ids):
    print('loading image data...')
    size = ids.shape[0]
    imgs = np.empty((size, 1, IMG_DIM, IMG_DIM))
    for i in range(size):
        img = load_img(os.path.join(imgpath, '%d.jpg' % ids[i]))
        imgs[i][0] = img_to_array(img.convert('L'))
    print('%d images loaded.' % size)
    return np.round(imgs / 255.0)


def build_model():
    print('creating the model...')

    cnn = Sequential()
    cnn.add(Convolution2D(8, 5, 5, border_mode='same', activation='relu', input_shape=(1, IMG_DIM, IMG_DIM)))
    cnn.add(MaxPooling2D())
    cnn.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu'))
    cnn.add(MaxPooling2D())
    cnn.add(Flatten())

    mlp = Sequential()
    mlp.add(Flatten(input_shape=(3, 8, 8)))

    model = Sequential()
    model.add(Merge([cnn, mlp], mode='concat'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(99, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    print('model created.')
    return model


if __name__ == '__main__':

    SS, LE, ids, x, y = load_train_data('data/train.csv')
    imgs = load_img_data('data/augmented_%d' % IMG_DIM, ids)
    x_tr, x_val, img_tr, img_val, y_tr, y_val = train_test_split(
        x.reshape(-1, 3, 8, 8), imgs, y, train_size=SPLIT_RATIO, random_state=1234)

    data_gen = ImageDataGenerator()
    img_gen = ImageDataGenerator(zoom_range=0.2,
                                 rotation_range=10,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 fill_mode='nearest')
    data_augmenter = data_gen.flow(x_tr, y_tr, batch_size=BATCH_SIZE, seed=1234)
    img_augmenter = img_gen.flow(img_tr, batch_size=BATCH_SIZE, seed=1234)

    model = build_model()

    train_loss = []
    val_loss = []
    min_loss = 10

    for ep in range(NB_EPOCH):
        ep_loss = []
        batches = 0
        for (data, y), img in zip(data_augmenter, img_augmenter):
            ep_loss.append(model.train_on_batch([img, data], y)[0])
            batches += BATCH_SIZE
            if batches >= x_tr.shape[0]:
                break

        train_loss.append(np.mean(ep_loss))
        loss = model.evaluate([img_val, x_val], y_val, verbose=0)[0]
        val_loss.append(loss)
        if min_loss > loss:
            model.save_weights('data/cnn.h5')
            print('Epoch %03d: val_loss improved from %.6f to %.6f' % (ep, min_loss, loss))
            min_loss = loss
        # print('Epoch %03d | train_loss: %.6f | val_loss: %.6f' % (ep, train_loss[-1], val_loss[-1]))

    # sns.set_style("whitegrid")
    # plt.plot(train_loss, linewidth=3, label='train loss')
    print('Min Loss: %.6f' % min_loss)
    model.load_weights('data/cnn.h5')
    ids, test = load_test_data('data/test.csv')
    test = SS.transform(test).reshape(-1, 3, 8, 8)
    imgs = load_img_data('data/augmented_%d' % IMG_DIM, ids)
    pred = model.predict([imgs, test])
    pd.DataFrame(pred, index=ids, columns=LE.inverse_transform(range(99))).to_csv('data/cnn_res.csv')
