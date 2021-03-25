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
import random
import time
import unittest
from typing import Iterable, Type, Callable

from tfaip import DataGeneratorParams
from tfaip.data.databaseparams import DataPipelineParams
from tfaip import Sample
from tfaip.data.pipeline.datagenerator import DataGenerator
from tfaip.util.multiprocessing.data.parallel_generator import ParallelGenerator
from tfaip.util.multiprocessing.data.worker import DataWorker
from tfaip.util.multiprocessing.join import JoinableHolder


class Worker(DataWorker):
    def initialize_thread(self):
        pass

    def process(self, gen):
        for v in gen:
            yield v
            time.sleep(random.randint(1, 5) / 100)
            yield v


class DGParams(DataGeneratorParams):
    @staticmethod
    def cls() -> Type['DataGenerator']:
        class DG(DataGenerator):
            def __len__(self):
                return 1000

            def generate(self) -> Iterable[Sample]:
                return [Sample(inputs={'data': [i]}, targets={'targets': [i]}) for i in range(1000)]

        return DG


class TestParallelData(unittest.TestCase):
    def test_run(self):
        from tfaip.imports import DataBase, DataBaseParams

        class TestData(DataBase):
            def _input_layer_specs(self):
                import tensorflow as tf
                return {"data": tf.TensorSpec(shape=[1], dtype=tf.int32)}

            def _target_layer_specs(self):
                import tensorflow as tf
                return {"targets": tf.TensorSpec(shape=[1], dtype=tf.int32)}

        data = TestData(DataBaseParams())
        with data.create_pipeline(DataPipelineParams(num_processes=8), DGParams()) as rd:
            for i, d in enumerate(zip(rd.input_dataset().as_numpy_iterator(), range(100))):
                print(i, d)
                pass

    def test_parallel_generator(self):
        holder = JoinableHolder()

        class Gen(ParallelGenerator):
            def create_worker_func(self) -> Callable[[], DataWorker]:
                return Worker

            def generate_input(self):
                return range(100)

        with Gen(
                holder,
                processes=4
        ) as pg:
            for o in pg.output_generator():
                print(o)
                # NOTE: The numbers are not ordered!!
