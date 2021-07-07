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
from dataclasses import dataclass, field
from typing import List, Type, Iterable

from paiargparse import pai_dataclass, pai_meta

from tfaip import DataGeneratorParams, Sample
from tfaip.data.pipeline.datagenerator import DataGenerator


@pai_dataclass
@dataclass
class ICPredictionDataGeneratorParams(DataGeneratorParams):
    @staticmethod
    def cls() -> Type["DataGenerator"]:
        return ICPredictionDataGenerator

    image_files: List[str] = field(default_factory=list, metadata=pai_meta(required=True))


# Custom data generator for prediction since the classes are unknown
class ICPredictionDataGenerator(DataGenerator[ICPredictionDataGeneratorParams]):
    def __len__(self):
        return len(self.params.image_files)

    def generate(self) -> Iterable[Sample]:
        return (Sample(inputs=fn) for fn in self.params.image_files)
