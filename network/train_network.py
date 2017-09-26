# train_model.py
import os
os.environ["THEANO_FLAGS"] = "gpuarray.preallocate=0.8"

from keras.callbacks import Callback
from keras.models import save_model, load_model
import keras
import numpy as np
from .models import *
import datetime
import math
from random import shuffle, randint
import time

DATA_DIR = 'E:/programmering/poe-scraper/training_data/'
SAVE_DIR = 'E:/programmering/poe-scraper/'

EPOCHS = 1000

MAX_VALUE_CUT_OFF = 20000

BATCH_SIZE = 64
TRAIN_FILES_PER_RUN = 9
N_VAL_DATA_FILES = 1


class Trainer():
    def __init__(self, input_size, name='default_name', loading_model=None):
        self.input_size = input_size
        self.name = name
        self.loading_model = loading_model
        self.std = 0
        self.mean = 0

    #def normalize_target(self, target):
    #    return target/MAX_VALUE_CUT_OFF

    def get_normalization_values(self, X):
        self.mean = np.mean(X, axis = 0)
        self.std = np.std(X, axis = 0)
        print('Mean: %s' % self.mean)
        print('Std: %s' % self.std)
        return X

    def normalize_X(self, X):
        X -= self.mean
        X /= self.std
        X = np.nan_to_num(X)
        return X

    def train(self):
        if not os.path.exists(SAVE_DIR + 'logs/' + self.name):
            os.makedirs(SAVE_DIR + 'logs/' + self.name)
        if not os.path.exists(SAVE_DIR + 'models/' + self.name):
            os.makedirs(SAVE_DIR + 'models/' + self.name)

        cb_tensorboard = keras.callbacks.TensorBoard(
            log_dir='E:/programmering/poe-scraper/logs/' + self.name,
            histogram_freq=0, write_graph=True, write_images=True,
            embeddings_freq=0, embeddings_layer_names=None,
            embeddings_metadata=None)

        model = BasicDense(self.input_size)
        if self.loading_model is not None:
            model = load_model(os.path.join(SCRIPT_PATH, self.loading_model))
            print("Loaded model from disk")

        model_train_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        print('Initialized trainer %s' %
              datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

        training_files = os.listdir(DATA_DIR)
        print('Localized %s files for T&T' % len(training_files))

        # Get normalization values
        norm_data = []
        for i in range(TRAIN_FILES_PER_RUN+N_VAL_DATA_FILES):
            norm_data.extend([[d[0], d[1]] for d in np.load(
                DATA_DIR + training_files[i]) if d[1] <= MAX_VALUE_CUT_OFF])
        N_X = np.array([d[0] for d in norm_data])
        self.get_normalization_values(N_X)
        del norm_data
        del N_X

        np.save('mean-std.npy', (self.mean, self.std))

        # Get some data to validate on from the training data
        val_files = []
        for i in range(N_VAL_DATA_FILES):
            val_files.append(training_files.pop(0))

        print('Loading data from %s files per run' % TRAIN_FILES_PER_RUN)
        print('Selected %s file(s) as validation files' % len(val_files))

        val_data = []
        for i in range(len(val_files)):
            val_data.extend([[d[0], d[1]] for d in np.load(
                DATA_DIR + val_files[i]) if d[1] <= MAX_VALUE_CUT_OFF])

        files_per_file = int(len(val_data)/len(val_files))

        V_X = self.normalize_X(np.array([d[0] for d in val_data]))
        print('There should not be any nans here: %s' % V_X)
        V_Y = np.array([d[1] for d in val_data])
        del val_data


        epoch = 0
        samples_trained = 0
        print('Starting run of %s epochs' % EPOCHS)

        while epoch <= EPOCHS:

            data = []
            for i in range(TRAIN_FILES_PER_RUN):
                data.extend([[d[0], d[1]] for d in np.load(DATA_DIR +
                    training_files[np.random.randint(len(training_files))])
                    if d[1] <= MAX_VALUE_CUT_OFF])

            X = self.normalize_X(np.array([d[0] for d in data]))
            Y = np.array([d[1] for d in data])

            m = 0
            n = 0
            for value in Y:
                if value > m:
                    m = value
                if value < n:
                    n = value
            print('Y - Max: %s, min: %s' % (m, n))
            model.fit(X, Y, batch_size=BATCH_SIZE, epochs=1, shuffle=True,
                      validation_data=(V_X, V_Y), callbacks=[cb_tensorboard])

            samples_trained += len(X)

            print('Samples fitted against: %s' % samples_trained)
            epoch = int(samples_trained/(len(training_files)*files_per_file))
            print('Saving model as %s-%s-EPOCH-%s' %
                  (self.name, model_train_time, epoch))
            model.save(SAVE_DIR + 'models/' + self.name + '/' +
                       '%s-%s-EPOCH-%s' % (self.name, model_train_time, epoch))

        print('Completed training %s' % datetime.datetime.now())
