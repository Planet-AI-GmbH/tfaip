from functools import partial

from tfaip.util.multiprocessing.data.pipeline import DataPipeline
from typing import TYPE_CHECKING, Callable

from tfaip.util.multiprocessing.data.worker import DataWorker

if TYPE_CHECKING:
    import tensorflow as tf


class ParallelMapWorker(DataWorker):
    def __init__(self, func):
        self.func = func

    def initialize_thread(self):
        pass

    def process(self, *args, **kwargs):
        return self.func(*args)


def create_worker(func):
    return ParallelMapWorker(func)


class ParallelMapPipeline(DataPipeline):
    def __init__(self, data, dataset: 'tf.data.Dataset', worker_func: Callable[[], DataWorker], processes: int, auto_repeat_input: bool = False):
        self.dataset = dataset
        self.worker_func = worker_func
        super(ParallelMapPipeline, self).__init__(data, processes, auto_repeat_input)

    def create_worker_func(self) -> Callable[[], DataWorker]:
        return partial(create_worker, self.worker_func)

    def generate_input(self):
        return self.dataset.as_numpy_iterator()


def parallel_map_dataset(data, dataset: 'tf.data.Dataset', output_types, worker_func: Callable[[], DataWorker], processes: int) -> 'tf.data.Dataset':
    import tensorflow as tf
    pipeline = ParallelMapPipeline(data, dataset, worker_func, processes)
    return tf.data.Dataset.from_generator(lambda: pipeline.output_generator(), output_types)
