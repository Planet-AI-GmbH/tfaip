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
import glob
from dataclasses import dataclass, field
import logging
from typing import Iterable, Type

import tensorflow as tf
import tensorflow.keras as keras
from dataclasses_json import dataclass_json

from tfaip.base.data.data import DataBaseParams, DataBase
from tfaip.base.data.databaseparams import DataGeneratorParams
from tfaip.base.data.pipeline.datapipeline import DataPipeline, DataGenerator, RawDataGenerator, RawDataPipeline
from tfaip.base.data.listfile.listfiledata import ListFilePipelineParams
from tfaip.base.data.pipeline.dataprocessor import DataProcessorFactory
from tfaip.base.data.pipeline.definitions import PipelineMode, Sample
from tfaip.util.argumentparser import dc_meta
from tfaip.util.imaging.io import load_image_from_img_file

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class DataParams(DataBaseParams):
    dataset: str = field(default='mnist', metadata=dc_meta(
        help="The dataset to select (chose also fashion_mnist)."
    ))


def to_samples(samples):
    return [Sample(inputs={'img': img}, targets={'gt': gt.reshape((1,))}) for img, gt in zip(*samples)]


class Data(DataBase):
    @classmethod
    def data_processor_factory(cls) -> DataProcessorFactory:
        return DataProcessorFactory([])

    @classmethod
    def data_pipeline_cls(cls) -> Type[DataPipeline]:
        class TutorialPipeline(DataPipeline):
            def create_data_generator(self) -> DataGenerator:
                if self.mode == PipelineMode.Training:
                    return RawDataGenerator(to_samples(self.data.train), self.mode, self.generator_params)
                elif self.mode == PipelineMode.Evaluation:
                    return RawDataGenerator(to_samples(self.data.test), self.mode, self.generator_params)
                elif self.mode == PipelineMode.Prediction:
                    if isinstance(self.generator_params, ListFilePipelineParams):
                        # Instead of loading images to a raw pipeline, you should create a custom preprocessing pipeline
                        # That is used during training and prediction
                        assert self.generator_params.list, "No images provided"
                        return RawDataGenerator(
                            [Sample(inputs={'img': img}) for img in map(load_image_from_img_file, glob.glob(self.generator_params.list))],
                            self.mode, self.generator_params)
                    else:
                        return RawDataGenerator(to_samples(self.data.test), self.mode, self.generator_params)
                elif self.mode == PipelineMode.Targets:
                    return RawDataGenerator(to_samples(self.data.test), self.mode, self.generator_params)
        return TutorialPipeline

    @classmethod
    def prediction_generator_params_cls(cls) -> Type[DataGeneratorParams]:
        return ListFilePipelineParams

    @staticmethod
    def get_params_cls():
        return DataParams

    def __init__(self, params: DataParams):
        super().__init__(params)
        self._params = params

        # Preload data for train and val pipeline
        dataset = getattr(keras.datasets, self._params.dataset)
        self.train, self.test = dataset.load_data()

    def _input_layer_specs(self):
        return {'img': tf.TensorSpec(shape=(28, 28), dtype='uint8')}

    def _target_layer_specs(self):
        return {'gt': tf.TensorSpec(shape=[1], dtype='uint8')}

    def _list_lav_dataset(self) -> Iterable[DataPipeline]:
        # Create two evaluation datasets using test and train data
        test = RawDataPipeline(to_samples(self.test), PipelineMode.Evaluation, self, self._params.val)
        train = RawDataPipeline(to_samples(self.train), PipelineMode.Evaluation, self, self._params.val)
        return [test, train]