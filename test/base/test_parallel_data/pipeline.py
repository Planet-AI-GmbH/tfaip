from typing import Callable

from test.base.test_parallel_data.worker import Worker
from tfaip.util.multiprocessing.data.pipeline import DataPipeline
from tfaip.util.multiprocessing.data.worker import DataWorker


def create():
    return Worker()


class Pipeline(DataPipeline):
    def __init__(self, data, processes, max_int):
        super(Pipeline, self).__init__(data, processes)
        self.max_int = max_int

    def create_worker_func(self) -> Callable[[], DataWorker]:
        return create

    def generate_input(self):
        return range(self.max_int)
