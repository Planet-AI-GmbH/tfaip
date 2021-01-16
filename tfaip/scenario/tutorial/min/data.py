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
from dataclasses import dataclass, field
import logging
from typing import Iterable, Type

import tensorflow as tf
import tensorflow.keras as keras
from dataclasses_json import dataclass_json

from tfaip.base.imports import DataBaseParams, DataBase
from tfaip.base.data.databaseparams import DataGeneratorParams
from tfaip.base.data.pipeline.datapipeline import DataPipeline, DataGenerator, RawDataGenerator, RawDataPipeline
from tfaip.base.data.listfile.listfiledata import ListFilePipelineParams
from tfaip.base.data.pipeline.dataprocessor import DataProcessorFactory
from tfaip.base.data.pipeline.definitions import PipelineMode, Sample
from tfaip.util.argumentparser import dc_meta


logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class DataParams(DataBaseParams):
    dataset: str = field(default='mnist', metadata=dc_meta(
        help="The dataset to select (chose also fashion_mnist)."
    ))


def to_samples(samples):
    return [Sample(inputs={'img': img}, targets={'gt': gt.reshape((1,))}) for img, gt in zip(*samples)]


class TutorialPipeline(DataPipeline):
    def create_data_generator(self) -> DataGenerator:
        # Create the DataGenerator
        # Here a simple RawDataGenerator is sufficient since the data is already loaded in the data class
        # by downloading the respective dataset
        if self.mode == PipelineMode.Training:
            return RawDataGenerator(to_samples(self.data.train), self.mode, self.generator_params)
        elif self.mode == PipelineMode.Evaluation:
            return RawDataGenerator(to_samples(self.data.test), self.mode, self.generator_params)
        else:
            raise NotImplementedError


class Data(DataBase):
    @classmethod
    def data_processor_factory(cls) -> DataProcessorFactory:
        # No data preprocessing is required, we use the raw data
        # Therefore no data processor classes are available
        return DataProcessorFactory([])

    @classmethod
    def data_pipeline_cls(cls) -> Type[DataPipeline]:
        # Data pipeline that defines how to load data
        return TutorialPipeline

    @staticmethod
    def get_params_cls():
        # Return the data parameter class
        return DataParams

    def __init__(self, params: DataParams):
        super().__init__(params)
        self._params = params  # For intelli sense

        # Preload data for train and val pipeline
        dataset = getattr(keras.datasets, self._params.dataset)
        self.train, self.test = dataset.load_data()

    def _input_layer_specs(self):
        # Shape and type of the input data for the graph
        return {'img': tf.TensorSpec(shape=(28, 28), dtype='uint8')}

    def _target_layer_specs(self):
        # Shape and type of the target (ground truth) data for the graph
        return {'gt': tf.TensorSpec(shape=[1], dtype='uint8')}
