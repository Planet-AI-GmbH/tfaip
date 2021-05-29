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

from paiargparse import pai_dataclass

from tfaip import Sample
from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams, MappingDataProcessor


@pai_dataclass
@dataclass
class NormalizeProcessorParams(DataProcessorParams):
    center: bool = True

    @staticmethod
    def cls():
        return NormalizeProcessor


class NormalizeProcessor(MappingDataProcessor[NormalizeProcessorParams]):
    """
    Example class to show how to use processors that are run in parallel in the samples in the input pipeline.
    This processor will normalize and center the input sample in the range of [-1, 1] (we know the input is in [0, 255]
    """

    def apply(self, sample: Sample) -> Sample:
        inputs = sample.inputs.copy()

        inputs["img"] = inputs["img"] / 255
        if self.params.center:
            inputs["img"] = (inputs["img"] - 0.5) * 2

        return sample.new_inputs(inputs)
