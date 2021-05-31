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
import logging
from dataclasses import dataclass, field
from typing import Type, Optional, Iterable, List

import tensorflow as tf
import tensorflow.keras as keras
from paiargparse import pai_meta, pai_dataclass

from tfaip import DataGeneratorParams, DataBaseParams
from tfaip import PipelineMode, Sample
from tfaip import TrainerPipelineParamsBase
from tfaip.data.data import DataBase
from tfaip.data.pipeline.datagenerator import DataGenerator

logger = logging.getLogger(__name__)


@pai_dataclass
@dataclass
class TutorialDataGeneratorParams(DataGeneratorParams):
    dataset: str = field(default="mnist", metadata=pai_meta(help="The dataset to select (chose also fashion_mnist)."))

    @staticmethod
    def cls() -> Type["DataGenerator"]:
        return TutorialDataGenerator


class TutorialDataGenerator(DataGenerator[TutorialDataGeneratorParams]):
    def __init__(self, mode: PipelineMode, params: "TutorialDataGeneratorParams"):
        super().__init__(mode, params)
        dataset = getattr(keras.datasets, params.dataset)
        train, test = dataset.load_data()
        data = train if mode == PipelineMode.TRAINING else test
        self.data = to_samples(data)

    def __len__(self):
        return len(self.data)

    def generate(self) -> Iterable[Sample]:
        return self.data


@pai_dataclass
@dataclass
class TutorialTrainerGeneratorParams(
    TrainerPipelineParamsBase[TutorialDataGeneratorParams, TutorialDataGeneratorParams]
):
    train_val: TutorialDataGeneratorParams = field(
        default_factory=TutorialDataGeneratorParams, metadata=pai_meta(mode="flat")
    )

    def train_gen(self) -> TutorialDataGeneratorParams:
        return self.train_val

    def val_gen(self) -> Optional[TutorialDataGeneratorParams]:
        return self.train_val


def to_samples(samples):
    return [Sample(inputs={"img": img}, targets={"gt": gt.reshape((1,))}) for img, gt in zip(*samples)]


@pai_dataclass
@dataclass
class TutorialDataParams(DataBaseParams):
    input_shape: List[int] = field(default_factory=lambda: [28, 28])

    @staticmethod
    def cls() -> Type["DataBase"]:
        return TutorialData


class TutorialData(DataBase[TutorialDataParams]):
    def _input_layer_specs(self):
        # Shape and type of the input data for the graph
        return {"img": tf.TensorSpec(shape=self.params.input_shape, dtype="uint8")}

    def _target_layer_specs(self):
        # Shape and type of the target (ground truth) data for the graph
        return {"gt": tf.TensorSpec(shape=[1], dtype="uint8")}
