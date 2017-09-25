from multiprocess import Lock
import time

class JobPool:
    def __init__(self, start_id):
        self.ids = [start_id]
        self.mutex = Lock()

    def get_id(self):
        if len(self.ids) > 0:
            with self.mutex:
                return self.ids.pop(0)
        else:
            return None

    def put_id(self, next_id):
        with self.mutex:
            self.ids.append(next_id)    
