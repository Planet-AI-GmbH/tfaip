# Copyright 2021 The tfaip authors. All Rights Reserved.
#
# This file is part of tfaip.
#
# tfaip is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# tfaip is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# tfaip. If not, see http://www.gnu.org/licenses/.
# ==============================================================================
"""Helper to implement a true parallel tf.data.Dataset.map with python functions"""
from functools import partial

from tfaip.util.multiprocessing.data.pipeline import ParallelPipeline
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


class ParallelMapPipeline(ParallelPipeline):
    """Helper class to run a tf.data.Dataset map with a py-function in parallel"""

    def __init__(
        self,
        dataset: "tf.data.Dataset",
        worker_func: Callable[[], DataWorker],
        processes: int,
        auto_repeat_input: bool = False,
    ):
        self.dataset = dataset
        self.worker_func = worker_func
        super().__init__(processes, auto_repeat_input)

    def create_worker_func(self) -> Callable[[], DataWorker]:
        return partial(create_worker, self.worker_func)

    def generate_input(self):
        return self.dataset.as_numpy_iterator()


def parallel_map_dataset(
    dataset: "tf.data.Dataset", output_types, worker_func: Callable[[], DataWorker], processes: int
) -> "tf.data.Dataset":
    import tensorflow as tf  # pylint: disable=import-outside-toplevel

    pipeline = ParallelMapPipeline(dataset, worker_func, processes)
    with pipeline as output_generator:
        return tf.data.Dataset.from_generator(output_generator, output_types)
