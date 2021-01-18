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
import logging
from typing import Type

import tensorflow as tf
import tensorflow.keras as keras

from tfaip.base.data.data import DataBaseParams, DataBase
from tfaip.base.data.databaseparams import DataGeneratorParams
from tfaip.base.data.pipeline.datapipeline import RawDataPipeline, SamplePipelineParams
from tfaip.base.data.listfile.listfiledata import ListFilePipelineParams
from tfaip.base.data.pipeline.dataprocessor import DataProcessorFactory
from tfaip.base.data.pipeline.definitions import PipelineMode, DataProcessorFactoryParams
from tfaip.scenario.tutorial.full.data.data_params import DataParams
from tfaip.scenario.tutorial.full.data.data_pipeline import TutorialPipeline, to_samples
from tfaip.scenario.tutorial.full.data.processors.normalize import NormalizeProcessor

logger = logging.getLogger(__name__)


class Data(DataBase):
    @classmethod
    def data_processor_factory(cls) -> DataProcessorFactory:
        # List all available processors here
        return DataProcessorFactory([NormalizeProcessor])

    @classmethod
    def data_pipeline_cls(cls) -> Type[TutorialPipeline]:
        return TutorialPipeline

    @classmethod
    def prediction_generator_params_cls(cls) -> Type[DataGeneratorParams]:
        return ListFilePipelineParams

    @classmethod
    def get_default_params(cls) -> DataBaseParams:
        params = super(Data, cls).get_default_params()
        # Define the default python input pipeline by specifying the list of processors
        # A DataProcessorFactoryParams requires the name of the class registered above in data_processor_factory
        # The second argument is the mode when to apply (Training (e.g., data augmentation), Prediction, Evaluation
        # (=validation during training), Targets (only produce GroundTruth)), the third parameter are optional args.
        params.pre_processors_ = SamplePipelineParams(
            run_parallel=True,  # Run the pipeline in parallel (by spawning subprocesses)
            sample_processors=[
                DataProcessorFactoryParams(NormalizeProcessor.__name__)
            ])

        return params

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

    def _list_lav_dataset(self):
        # Create two evaluation datasets using test and train data
        test = RawDataPipeline(to_samples(self.test), PipelineMode.Evaluation, self, self._params.val)
        train = RawDataPipeline(to_samples(self.train), PipelineMode.Evaluation, self, self._params.val)
        return [test, train]
