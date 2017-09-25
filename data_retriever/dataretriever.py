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

PULLS_PER_SAVE = 50


class DataRetriever(Controller):
    def __init__(self, thread_id):
        self.location = os.path.dirname(os.path.realpath(__file__))
        self.thread_id = thread_id
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.currencyconverter = CurrencyConverter(league='Harbinger')
        self.filter = Filter(self.currencyconverter)
        self.retriever = Retriever()
        self.encoder = Encoder(self._get_classes_path())

    def collect(self, pulls):
        item_count = 0
        filtered_item_count = 0
        ips = 0
        start_time = time.time()
        filtered_data = []
        for i in range(pulls):
            while next_id is None:
                next_id = Controller.jobpool.get_id()
                if next_id is None:
                    time.sleep(1)
            last_id = next_id
            (data, next_id) = self.retriever.retrieve(next_id)
            Controller.jobpool.put_id(next_id)
            item_count += len(data)
            X_Y = self.filter.filter_items(data)
            filtered_data.extend(X_Y)

            self.encoder.fit([item_value_tuple[0] for item_value_tuple in X_Y])

            if i % PULLS_PER_SAVE == 0:
                encoded_data = self.encoder.encode(filtered_data)
                np.save('%s\\%s\\%s.npy' % (self.location, DEFAULT_DATA_DIR, last_id),
                        np.array(filtered_data))
                np.save(DEFAULT_ENCODED_DATA_DIR + last_id + '.npy',
                        np.array(encoded_f))
                filtered_data = []
            filtered_item_count += len(filtered_item_count)
            ips = filtered_item_count/(time.time()-start_Time)
            next_id = None
        print('Thread #%s finished. Collected %s items (%s eligible) at %.1f eligible items per second'
              % (self.thread_id, item_count, filtered_item_count, ips))

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
            return self.location + DEFAULT_LABEL_FILE + str(self.thread_id) + '.npy'
        else:
            return self.location + value + str(self.thread_id) + '.npy'
