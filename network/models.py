
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def BasicDense(n_features=500):

    model = Sequential()

    model.add(Dense(n_features, input_shape=(n_features,)))
    model.add(Activation("relu"))
    model.add((BatchNormalization()))

    model.add(Dropout(0.2))
    model.add(Dense(int(n_features/2)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Dropout(0.2))
    model.add(Dense(int(n_features/4)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Dropout(0.2))
    model.add(Dense(int(n_features/16)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Dropout(0.2))
    model.add(Dense(int(n_features/32)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Dropout(0.2))
    model.add(Dense(int(n_features/64)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Dropout(0.2))
    model.add(Dense(int(n_features/128)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation("relu"))

    adam = Adam()
    model.compile(loss='mse',
                  optimizer=adam)

    return model
