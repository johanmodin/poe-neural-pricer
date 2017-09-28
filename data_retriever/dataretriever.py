# Controls the scraper's modules
import time
from os.path import dirname, realpath
import multiprocessing as mp
import numpy as np
import configparser
import json

from .filter import Filter
from .retriever import Retriever
from .currency_converter import CurrencyConverter
from .encoder import Encoder
from .get_next_id import get_next_id

DEFAULT_LABEL_FILE = '\labels\classes'
DEFAULT_DATA_DIR = '\saved_data\\'
DEFAULT_ENCODED_DATA_DIR = 'E:\programmering\poe-scraper\\training_data\\'

PULLS_PER_SAVE = 10
SIMULTANEOUS_REQUESTERS = 6
TIME_BETWEEN_REQUESTS = 1

class DataRetriever:
    def __init__(self):
        self.location = dirname(realpath(__file__))
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.currencyconverter = CurrencyConverter(league='Harbinger')
        self.filter = Filter(self.currencyconverter)
        self.retriever = Retriever()
        self.encoder = Encoder(self._get_classes_path())

    def collect(self, pulls, start_id):
        pool = mp.Pool(SIMULTANEOUS_REQUESTERS)
        next_id = start_id
        filtered_item_count, ips = 0, 0
        start_time = time.time()
        filtered_data = []

        print('Initiating retrieving..')
        for i in range(pulls):
            if next_id is None:
                print('No more data to fetch, quitting.')

            next_ids = self._request_ids(next_id)
            next_id = next_ids[-1]

            data = self._request_data(next_ids, pool)
            if data == None:
                print('We reached the end of the stash updates. Exiting.')
                sys.exit(0)

            X_Y = self.filter.filter_items(data)
            filtered_data.extend(X_Y)
            self.encoder.fit([item_value_tuple[0] for item_value_tuple in X_Y])

            filtered_item_count += len(X_Y)
            ips = filtered_item_count/(time.time()-start_time)

            if i != 0 and i % PULLS_PER_SAVE == 0:
                np.save('%s\\%s\\%s.npy' % (self.location, DEFAULT_DATA_DIR, next_id), np.array(filtered_data))
                filtered_data = []
                print('Retriever saved data. Requested %s/%s pages per worker and collected %s eligible items at %.1f items per second'
                      % (i, pulls, filtered_item_count, ips))
                print('Last retrieved next_change_id was: %s' % next_id)

        print('Retriever finished. Requested %s pages per worker and collected %s eligible items at %.1f items per second'
              % (i, filtered_item_count, ips))

    def encode(self, files):
        for i in range(len(files)):
            print('Encoding file %s' % (files[i]))
            filtered_data = np.load('%s%s%s' % (self.location, DEFAULT_DATA_DIR, files[i]))
            encoded_data = self.encoder.encode(filtered_data)
            np.save(DEFAULT_ENCODED_DATA_DIR + files[i], encoded_data)

    def _request_ids(self, start_id):
        id_list = [start_id]
        for i in range(SIMULTANEOUS_REQUESTERS):
            id_list.append(get_next_id(id_list[i]))
        return id_list[1:]

    def _request_data(self, next_ids, pool):
        base_request_time = time.time()
        worker_args = []
        for i in range(SIMULTANEOUS_REQUESTERS):
            worker_args.append((next_ids[i], base_request_time + i*TIME_BETWEEN_REQUESTS))
        data = pool.map(RequestWorker.request_data, worker_args)
        if None in data:
            return None
        j = json.dumps(data)
        merged_data = [item for sublist in data for item in sublist]
        return merged_data

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

class RequestWorker:
    def request_data(args):
        next_id = args[0]
        request_time = args[1]
        ret = Retriever()
        return ret.retrieve(next_id, request_time)
