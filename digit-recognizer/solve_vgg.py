import time
import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils


PIXEL = 28
BATCH_SIZE = 128
NB_EPOCH = 100


def load_train_data(filepath, rd=0):
    print('Loading the training set ...')
    train = pd.read_csv(filepath).values
    train_x = (train[:, 1:].reshape(train.shape[0], 1, PIXEL, PIXEL) / 255.0).astype(float)
    train_y = np_utils.to_categorical(train[:, 0])
    splited = train_test_split(train_x, train_y, random_state=rd)
    print('%d samples loaded' % train_x.shape[0])
    return splited


def load_test_data(filepath):
    print('Loading the test set ...')
    test = (pd.read_csv(filepath).values.reshape(-1, 1, PIXEL, PIXEL) / 255.0).astype(float)
    print('%d samples loaded' % test.shape[0])
    return test.astype(float)


def build_model():
    print('Creating the model ...')
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', input_shape=(1, PIXEL, PIXEL)))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    # sgd = SGD(lr=0.03, decay=1e-4, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #
    # model.summary()
    return model


if __name__ == '__main__':
    for runid in range(36):
        print('running the No.%d round' % runid)
        x_train, x_val, y_train, y_val = load_train_data('data/train.csv', runid)
        cnn = build_model()

        cnn.compile(loss='categorical_crossentropy',
                    optimizer='adadelta',
                    metrics=['accuracy'])
        cnn.fit(x_train, y_train,
                batch_size=BATCH_SIZE,
                nb_epoch=30,
                validation_data=(x_val, y_val),
                verbose=0)

        datagen = ImageDataGenerator(rotation_range=20,
                                     width_shift_range=0.15,
                                     height_shift_range=0.15,
                                     shear_range=0.2,
                                     zoom_range=0.3)
        cnn.compile(loss='categorical_crossentropy',
                    optimizer='adamax',
                    metrics=['accuracy'])
        save_best = ModelCheckpoint(filepath='data/modeldump/best%d.kerasmodel' % runid,
                                    monitor='val_acc',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=True)
        cnn.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                          samples_per_epoch=len(x_train),
                          nb_epoch=NB_EPOCH,
                          verbose=0,
                          callbacks=[save_best],
                          validation_data=(x_val, y_val))

        cnn.load_weights('data/modeldump/best%d.kerasmodel' % runid)
        test = load_test_data('data/test.csv')
        pred = cnn.predict_classes(test)
        sub = pd.read_csv('data/sample_submission.csv')
        sub['Label'] = pred
        sub.to_csv('data/vgg_results/submission_vgg_%d.csv' % runid, index=False)
