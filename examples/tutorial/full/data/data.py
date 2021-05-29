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
from dataclasses import dataclass

import tensorflow as tf
from paiargparse import pai_dataclass

from examples.tutorial.full.data.processors.normalize import NormalizeProcessorParams
from tfaip import DataBaseParams
from tfaip.data.data import DataBase
from tfaip.data.pipeline.processor.params import SequentialProcessorPipelineParams


@pai_dataclass
@dataclass
class TutorialDataParams(DataBaseParams):
    @staticmethod
    def cls():
        return TutorialData


class TutorialData(DataBase[TutorialDataParams]):
    @classmethod
    def default_params(cls) -> DataBaseParams:
        params = super(TutorialData, cls).default_params()
        # Define the default python input pipeline by specifying the list of processors
        # A DataProcessorFactoryParams requires the name of the class registered above in data_processor_factory
        # The second argument is the mode when to apply (Training (e.g., data augmentation), Prediction, Evaluation
        # (=validation during training), Targets (only produce GroundTruth)), the third parameter are optional args.
        params.pre_proc = SequentialProcessorPipelineParams(
            run_parallel=False,  # Set this to True to run the pipeline in parallel (by spawning subprocesses)
            processors=[NormalizeProcessorParams()],
        )

        return params

    def _input_layer_specs(self):
        # Define the input specs of the graph. Here a [28, 28] image of type uint8. The batch dimension is not stated.
        return {"img": tf.TensorSpec(shape=(28, 28), dtype="uint8")}

    def _target_layer_specs(self):
        # Define the target specs. Here a single label of type uint8 is the target (the number in the image).
        # The batch dimension is not stated and scalars must be of dimension [1]
        return {"gt": tf.TensorSpec(shape=[1], dtype="uint8")}
