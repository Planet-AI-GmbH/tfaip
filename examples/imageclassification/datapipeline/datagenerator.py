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
import os
from dataclasses import dataclass, field
from random import shuffle
from typing import Iterable, Type, List, Optional, Dict

from paiargparse import pai_dataclass, pai_meta
from tfaip import DataGeneratorParams, Sample, TrainerPipelineParamsBase, PipelineMode
from tfaip.data.pipeline.datagenerator import DataGenerator


@pai_dataclass
@dataclass
class ICDataGeneratorParams(DataGeneratorParams):
    @staticmethod
    def cls() -> Type["DataGenerator"]:
        return ICDataGenerator

    image_files: Dict[str, List[str]] = field(default_factory=list, metadata=pai_meta(mode="ignore"))

    def num_files(self):
        return sum([len(files) for files in self.image_files.values()])


class ICDataGenerator(DataGenerator[ICDataGeneratorParams]):
    def __len__(self):
        return self.params.num_files()

    def generate(self) -> Iterable[Sample]:
        # Generate the samples
        # First flatten all, since shuffling is performed during training (on each epoch anew)
        # Also shuffle in evaluation (no effect on the accuracy) but random examples will be displayed
        flat_samples = []
        for k, filenames in self.params.image_files.items():
            for fn in filenames:
                # Pass inputs and targets, meta data is optional but can be useful for debugging
                flat_samples.append(Sample(inputs=fn, targets=k, meta={"filename": fn, "classname": k}))

        if self.mode in {PipelineMode.TRAINING, PipelineMode.EVALUATION}:
            shuffle(flat_samples)

        return flat_samples


@pai_dataclass
@dataclass
class ICTrainerPipelineParams(TrainerPipelineParamsBase[ICDataGeneratorParams, ICDataGeneratorParams]):
    dataset_path: str = field(default="")
    validation_split: float = 0.2
    shuffle_files: bool = True

    # resolved files, a list of file names per class
    image_files: Dict[str, List[str]] = field(default_factory=dict, metadata=pai_meta(mode="ignore"))

    def train_gen(self) -> ICDataGeneratorParams:
        return ICDataGeneratorParams(
            image_files={k: v[int(self.validation_split * len(v)) :] for k, v in self.image_files.items()}
        )

    def val_gen(self) -> Optional[ICDataGeneratorParams]:
        val = ICDataGeneratorParams(
            image_files={k: v[: int(self.validation_split * len(v))] for k, v in self.image_files.items()}
        )
        if val.num_files() == 0:
            return None  # No validation
        return val

    def __post_init__(self):
        self.image_files = {}
        if os.path.exists(self.dataset_path):
            for class_name in os.listdir(self.dataset_path):
                class_path = os.path.join(self.dataset_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                self.image_files[class_name] = [os.path.join(class_path, file) for file in os.listdir(class_path)]
                if self.shuffle_files:
                    shuffle(self.image_files[class_name])
