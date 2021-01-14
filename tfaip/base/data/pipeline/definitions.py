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
from dataclasses import field, dataclass
from typing import NamedTuple, Any, Set, Optional

from dataclasses_json import dataclass_json

from tfaip.util.enum import StrEnum


class PipelineMode(StrEnum):
    Training = 'training'           # Inputs and Targets, however during training e.g. Data-Augmentation, etc.
    Evaluation = 'evaluation'       # Inputs and Targets
    Prediction = 'prediction'       # Inputs
    Targets = 'targets'             # Targets


INPUT_PROCESSOR = {PipelineMode.Training, PipelineMode.Evaluation, PipelineMode.Prediction}
TARGETS_PROCESSOR = {PipelineMode.Training, PipelineMode.Evaluation, PipelineMode.Targets}
GENERAL_PROCESSOR = {PipelineMode.Training, PipelineMode.Evaluation, PipelineMode.Prediction, PipelineMode.Targets}


class Sample:
    def __init__(self, *, inputs: Any = None, outputs: Any = None, targets: Any = None, meta: Any = None):
        self.inputs = inputs
        self.outputs = outputs
        self.targets = targets
        self.meta = meta

    def new_invalid(self):
        return Sample(meta=self.meta)

    def copy(self):
        return Sample(inputs=self.inputs, outputs=self.outputs, targets=self.targets, meta=self.meta)

    def new_inputs(self, inputs):
        s = self.copy()
        s.inputs = inputs
        return s

    def new_outputs(self, outputs):
        s = self.copy()
        s.outputs = outputs
        return s

    def new_targets(self, targets):
        s = self.copy()
        s.targets = targets
        return s

    def new_meta(self, meta):
        s = self.copy()
        s.meta = meta
        return s


@dataclass_json
@dataclass
class DataProcessorFactoryParams:
    name: str
    modes: Set[PipelineMode] = field(default_factory=GENERAL_PROCESSOR.copy)
    args: Optional[dict] = None
