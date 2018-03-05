import configparser
import multiprocessing as mp
import os

from data_retriever.dataretriever import DataRetriever

DEFAULT_DATA_DIR = '\data_retriever\saved_data\\'


class Controller:
    def __init__(self):
        self.location = os.path.dirname(os.path.realpath(__file__))
        cfg = configparser.ConfigParser()
        cfg.read('config.ini')
        self.start_id = cfg['retriever']['NextId']

    def collect(self, iterations):
        d = DataRetriever()
        d.collect(iterations, self.start_id)

    def encode(self, workers=4):
        files = os.listdir(self.location + DEFAULT_DATA_DIR)
        worker_args = []
        file_counter, worker_index = 0, 0
        while file_counter < len(files):
            if len(worker_args) <= worker_index % workers:
                worker_args.append([])
            worker_args[worker_index % workers].append(files[file_counter])
            worker_index += 1
            file_counter += 1
        pool = mp.Pool(workers)
        pool.map(EncodeWorker.worker_collect, worker_args)

class EncodeWorker:
    def worker_collect(ids):
        d = DataRetriever()
        d.encode(ids)
