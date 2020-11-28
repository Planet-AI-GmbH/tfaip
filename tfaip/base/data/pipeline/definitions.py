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


inputs_pipeline_modes = {PipelineMode.Training, PipelineMode.Evaluation, PipelineMode.Prediction}
targets_pipeline_modes = {PipelineMode.Training, PipelineMode.Evaluation, PipelineMode.Targets}
all_pipeline_modes = {PipelineMode.Training, PipelineMode.Evaluation, PipelineMode.Prediction, PipelineMode.Targets}


class InputTargetSample(NamedTuple):
    inputs: Any
    targets: Any
    meta: Any = None  # Meta information (optional). Can e. g. be used to identify a sample (e.g. an ID)


class InputOutputSample(NamedTuple):
    inputs: Any
    outputs: Any
    meta: Any = None


@dataclass_json
@dataclass
class DataProcessorFactoryParams:
    name: str
    modes: Set[PipelineMode] = field(default_factory=all_pipeline_modes.copy)
    args: Optional[dict] = None
