''' Author: Yogesh Verma '''
''' HIGAN '''



import keras.backend as K
from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, merge,
                          Dropout, BatchNormalization, Activation, Embedding)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (UpSampling2D, Conv2D, ZeroPadding2D,
                                        AveragePooling2D)
from keras.layers.local import LocallyConnected2D
import random
import numpy as np
from keras.models import Model, Sequential
from numpy.random import rand
from numpy.random import randint


def discriminator():

    in_shape=(20, 20, 1)

    model = Sequential()
    model.add(Conv2D(32, 5, 5, border_mode='same',input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))


    model.add(ZeroPadding2D((2, 2)))
    model.add(LocallyConnected2D(16, 5, 5, border_mode='valid', subsample=(2, 2)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))



    model.add(ZeroPadding2D((1, 1)))
    model.add(LocallyConnected2D(8, 3, 3, border_mode='valid', subsample=(2, 2)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(AveragePooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid', name='generation'))

    return model


def generator(latent_size,return_intermediate=False):

    model = Sequential()
    model.add(Dense(128*7*7,input_dim=latent_size))
    model.add(Reshape((7, 7, 128)))

    #block 1: (None, 7, 7, 128) => (None, 14, 14, 64)
    model.add(Conv2D(64, 5, 5, border_mode='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    #block 2: (None, 14, 14, 64) => (None, 28, 28, 6),
    model.add(ZeroPadding2D((2, 2)))
    model.add(LocallyConnected2D(6, 5, 5))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    #block 3: (None, 28, 28, 6) => (None, 20, 20, 1),
    model.add(LocallyConnected2D(6, 3, 3))
    model.add(LeakyReLU(alpha=0.2))
    model.add(LocallyConnected2D(1, 7, 7, bias=False))
    model.add(Activation('sigmoid'))


    return model



