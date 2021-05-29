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
from copy import copy
from dataclasses import dataclass, field
from typing import Optional, Iterable

from paiargparse import pai_dataclass, pai_meta

from examples.tutorial.full.data.prediction_data_generation import to_samples
from tfaip import DataGeneratorParams, PipelineMode, TrainerPipelineParamsBase


@pai_dataclass
@dataclass
class TutorialDataGeneratorParams(DataGeneratorParams):
    """
    Parameters for the actual `TutorialDataGenerator` which is defined inline in `cls()`.
    """

    dataset: str = field(
        default="mnist",
        metadata=pai_meta(choices=["mnist", "fashion_mnist"], help="The dataset to select (chose also fashion_mnist)."),
    )

    force_train: bool = field(default=False, metadata=pai_meta(help="Force using of training data also for validation"))

    shuffle: Optional[bool] = field(
        default=None, metadata=pai_meta(help="Set to False to disable shuffle on training.")
    )

    @staticmethod
    def cls():
        from tensorflow import keras

        from tfaip.data.pipeline.datagenerator import RawDataGenerator

        class TutorialDataGen(RawDataGenerator):
            """
            Load the data which is already split into train and val (=test).
            Depending on the pipeline `mode` and `force_train` select the dataset to return.
            """

            def __init__(self, mode: PipelineMode, params: "TutorialDataGeneratorParams"):
                dataset = getattr(keras.datasets, params.dataset)
                train, test = dataset.load_data()
                data = train if mode == PipelineMode.TRAINING or params.force_train else test
                super(TutorialDataGen, self).__init__(to_samples(data), mode, params)
                if params.shuffle is not None:
                    self.shuffle = params.shuffle

        return TutorialDataGen


@pai_dataclass
@dataclass
class TutorialTrainerGeneratorParams(
    TrainerPipelineParamsBase[TutorialDataGeneratorParams, TutorialDataGeneratorParams]
):
    """
    Definition of the training data. Since the dataset is loaded from the keras.datasets, training and validation data
    is jointly loaded (parameter `train_val`) which is why `train_gen` and `val_gen` return the same generator.
    The decision whether to chose training and validation data is dependent on the `PipelineMode`.

    Furthermore, the `lav_gen` method is overwritten to perform lav on both the training and the validation set.
    For this purpose, the `force_train` variable is overwritten, to select the training data even if the PipelineMode is
    PipelineMode.EVALUATION.
    """

    train_val: TutorialDataGeneratorParams = field(
        default_factory=TutorialDataGeneratorParams, metadata=pai_meta(mode="flat")
    )

    def train_gen(self) -> TutorialDataGeneratorParams:
        return self.train_val

    def val_gen(self) -> Optional[TutorialDataGeneratorParams]:
        return self.train_val

    def lav_gen(self) -> Iterable[TutorialDataGeneratorParams]:
        train: TutorialDataGeneratorParams = copy(self.train_val)
        train.force_train = True
        return [train, self.train_val]
