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
"""General definitions for the Pipeline: PipelineMode, Sample"""
from typing import Any

from tfaip.util.enum import StrEnum


class PipelineMode(StrEnum):
    TRAINING = "training"  # Inputs and Targets, however during training e.g. Data-Augmentation, etc.
    EVALUATION = "evaluation"  # Inputs and Targets (e.g. LAV and validation)
    PREDICTION = "prediction"  # Inputs
    TARGETS = "targets"  # Targets


INPUT_PROCESSOR = {PipelineMode.TRAINING, PipelineMode.EVALUATION, PipelineMode.PREDICTION}
TARGETS_PROCESSOR = {PipelineMode.TRAINING, PipelineMode.EVALUATION, PipelineMode.TARGETS}
GENERAL_PROCESSOR = {PipelineMode.TRAINING, PipelineMode.EVALUATION, PipelineMode.PREDICTION, PipelineMode.TARGETS}


class Sample:
    """
    Basic structure of a single (unbatched) sample which is used to process data.
    The sample provides four fields (inputs, outputs, targets, and optionally meta) which must be set dependen on the
    PipelineMode.

    Meta should be a dict with arbitrary serializable data.

    Call the provided methods `new_inputs`, `new_outputs`, `new_targets`, `new_meta`, or `new_invalid` to build and
    return a new (copied) sample after transformations were applied.
    """

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
