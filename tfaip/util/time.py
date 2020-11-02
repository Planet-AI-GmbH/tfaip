import time


class MeasureTime:
    def __init__(self):
        self.start = 0
        self.end = 0
        self.duration = -1

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.duration = self.end - self.start


