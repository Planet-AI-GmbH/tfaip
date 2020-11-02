from tfaip.util.multiprocessing.data.worker import DataWorker


class Worker(DataWorker):
    def initialize_thread(self):
        pass

    def process(self, *args, **kwargs):
        return args
