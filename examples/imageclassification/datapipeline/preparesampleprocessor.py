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

from examples.imageclassification.params import Keys
from tfaip import Sample, TARGETS_PROCESSOR, INPUT_PROCESSOR
from tfaip.data.pipeline.processor.dataprocessor import DataProcessorParams, MappingDataProcessor


@pai_dataclass
@dataclass
class PrepareSampleProcessorParams(DataProcessorParams):
    @staticmethod
    def cls():
        return PrepareSampleProcessor


class PrepareSampleProcessor(MappingDataProcessor[PrepareSampleProcessorParams]):
    def apply(self, sample: Sample) -> Sample:
        if self.mode in INPUT_PROCESSOR:
            sample = sample.new_inputs({Keys.Image: sample.inputs})
        if self.mode in TARGETS_PROCESSOR:
            sample = sample.new_targets({Keys.Target: [sample.targets]})
        return sample
