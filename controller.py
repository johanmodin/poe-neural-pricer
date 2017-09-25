from data_retriever.dataretriever import DataRetriever
from network.train_network import Trainer
from job_pool import JobPool
import configparser

import multiprocessing as mp
from multiprocessing import Queue

class Controller:
    def __init__(self):
        cfg = configparser.ConfigParser()
        cfg.read('config.ini')
        self.jobpool = JobPool(cfg['retriever']['NextId'])

    def _worker_collect(self, args):
        dataretriever = args[0]
        iterations = args[1]
        print('Thread started.')
        dataretriever.collect(iterations)
        print('Thread finished.')

    def collect(self, iterations, workers=4):
         pool = mp.Pool(workers)
         worker_args = []
         for i in range(workers):
             worker_args.append((DataRetriever(i), iterations))
         pool.map(_worker_collect, worker_args)
