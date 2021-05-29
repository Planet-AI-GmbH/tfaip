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
from typing import Type, Iterable

from paiargparse import pai_dataclass

from tfaip import DataGeneratorParams, Sample
from tfaip.data.pipeline.datagenerator import DataGenerator


@pai_dataclass
@dataclass
class TemplateDataGeneratorParams(DataGeneratorParams):
    # [Add parameters that define where the data originates from, e.g. file paths]

    @staticmethod
    def cls() -> Type["DataGenerator"]:
        return TemplateDataGenerator


class TemplateDataGenerator(DataGenerator[TemplateDataGeneratorParams]):
    def __len__(self):
        # Return the number of samples this generator will produce
        # (if unknown, return an arbitrary number, this will only lead issues with an optional progress bar)
        raise NotImplementedError

    def generate(self) -> Iterable[Sample]:
        # Return the samples that will then be processed be the data processors
        raise NotImplementedError
