# Copyright 2020 The tfaip authors. All Rights Reserved.
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
import unittest
from typing import Callable, Iterable, Type

from tfaip.base.data.pipeline.datapipeline import SimpleDataPipeline, DataPipeline
from tfaip.base.data.pipeline.dataprocessor import DataProcessorFactory
from tfaip.base.data.pipeline.definitions import PipelineMode, InputTargetSample


class TestParallelData(unittest.TestCase):
    def test_run(self):
        from tfaip.base.data.data import DataBase, DataBaseParams

        class TestData(DataBase):
            @classmethod
            def data_processor_factory(cls) -> DataProcessorFactory:
                return DataProcessorFactory([])

            @classmethod
            def data_pipeline_cls(cls) -> Type[DataPipeline]:
                class PPipeline(SimpleDataPipeline):
                    def generate_samples(self) -> Iterable[InputTargetSample]:
                        return [InputTargetSample({'data': i}, {'targets': i}) for i in range(1000)]
                return PPipeline

            def _input_layer_specs(self):
                import tensorflow as tf
                return {"data": tf.TensorSpec(shape=[], dtype=tf.int32)}

            def _target_layer_specs(self):
                import tensorflow as tf
                return {"targets": tf.TensorSpec(shape=[], dtype=tf.int32)}

        params = DataBaseParams()
        params.train.num_processes = 8
        data = TestData(params)
        with data.get_train_data() as rd:
            for i, d in enumerate(zip(rd.input_dataset().as_numpy_iterator(), range(100))):
                print(i, d)
                pass

