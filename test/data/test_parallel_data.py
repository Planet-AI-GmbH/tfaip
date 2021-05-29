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
from tfaip.data.pipeline.processor.params import ComposedProcessorPipelineParams
from tfaip.util.multiprocessing.data.parallel_generator import ParallelGenerator
from tfaip.util.multiprocessing.data.worker import DataWorker


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
    def cls() -> Type["DataGenerator"]:
        class DG(DataGenerator):
            def __len__(self):
                return 1000

            def generate(self) -> Iterable[Sample]:
                return [Sample(inputs={"data": [i]}, targets={"targets": [i]}) for i in range(1000)]

        return DG


class TestParallelData(unittest.TestCase):
    def test_standalone_pipeline(self):
        from tfaip.imports import DataBaseParams

        class TestDataParams(DataBaseParams):
            @staticmethod
            def cls():
                raise NotImplementedError

        data_params = TestDataParams()
        samples = [Sample()] * 100
        pipeline = data_params.pre_proc.create(DataPipelineParams(num_processes=8), data_params)
        for i, d in enumerate(pipeline.apply(samples)):
            print(i, d)

    def test_parallel_generator(self):
        class Gen(ParallelGenerator):
            def create_worker_func(self) -> Callable[[], DataWorker]:
                return Worker

            def generate_input(self):
                return range(100)

        with Gen(processes=4) as output_generator:
            for o in output_generator:
                print(o)
                # NOTE: The numbers are not ordered!!
