# Controls the scraper's modules

# Retrieve -> Filter -> Encode -> Write
import configparser

import os
from os import listdir
from os.path import isfile, join

import time
import numpy as np

from .filter import Filter
from .retriever import Retriever
from .currency_converter import CurrencyConverter
from .encoder import Encoder

DEFAULT_LABEL_FILE = '\labels\classes'
DEFAULT_DATA_DIR = '\saved_data\\'
DEFAULT_ENCODED_DATA_DIR = 'E:\programmering\poe-scraper\\training_data\\'

PULLS_PER_SAVE = 10


class DataRetriever:
    def __init__(self):
        self.location = os.path.dirname(os.path.realpath(__file__))
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.currencyconverter = CurrencyConverter(league='Harbinger')
        self.filter = Filter(self.currencyconverter)
        self.retriever = Retriever()
        self.encoder = Encoder(self._get_classes_path())

    def collect(self, pulls, start_id):
        next_id = start_id
        item_count, filtered_item_count, ips = 0, 0, 0
        start_time = time.time()
        filtered_data = []
        for i in range(pulls):
            if next_id is None:
                print('No more data to fetch, quitting.')
            last_id = next_id
            print('Retrieving %s (%s/%s). IPS: %s' % (next_id, i, pulls, ips)
            (data, next_id) = self.retriever.retrieve(next_id)
            X_Y = self.filter.filter_items(data)
            filtered_data.extend(X_Y)
            self.encoder.fit([item_value_tuple[0] for item_value_tuple in X_Y])

            item_count += len(data)
            filtered_item_count += len(X_Y)
            ips = filtered_item_count/(time.time()-start_time)
            if i != 0 and i % PULLS_PER_SAVE == 0:
                np.save('%s\\%s\\%s.npy' % (self.location, DEFAULT_DATA_DIR, last_id), np.array(filtered_data))
                filtered_data = []
                print('Retriever saved data. Requested %s pages and collected %s items (%s eligible) at %.1f eligible items per second'
                      % (i, item_count, filtered_item_count, ips))
        print('Retriever finished. Requested %s pages and collected %s items (%s eligible) at %.1f eligible items per second'
              % (i, item_count, filtered_item_count, ips))

    def encode(self, files):
        for i in range(len(files)):
            print('Encoding file %s' % (files[i]))
            filtered_data = np.load('%s%s%s' % (self.location, DEFAULT_DATA_DIR, files[i]))
            encoded_data = self.encoder.encode(filtered_data)
            np.save(DEFAULT_ENCODED_DATA_DIR + files[i], encoded_data)

    def _get_next_id(self):
        value = self.config['retriever']['NextId']
        if value == '':
            return 0
        return value

    def _get_config_value(self, value):
        if ',' in value:
            return value.split(',')
        return value

    def _get_classes_path(self):
        value = self.config['encoder']['ClassesFile']
        if value is '':
            print('Notice: label class file not set, using default.')
            return self.location + DEFAULT_LABEL_FILE + '.npy'
        else:
            return self.location + value + '.npy'
