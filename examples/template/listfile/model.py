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
from typing import Dict, List

from examples.template.listfile.params import TemplateModelParams
from tfaip import Sample
from tfaip.model.modelbase import ModelBase
from tfaip.util.tftyping import AnyTensor


class TemplateModel(ModelBase[TemplateModelParams]):
    def _best_logging_settings(self):
        # Return the settings how the best model shall be detected, e.g. the maximum accuracy (acc is a metric which
        # must be defined in _metric or _extended_metric):
        # return "max", "acc"
        raise NotImplementedError

    def _loss(self, inputs, targets, outputs) -> Dict[str, AnyTensor]:
        # Implement the loss for this scenario
        raise NotImplementedError

    def _metric(self, inputs, targets, outputs) -> List[AnyTensor]:
        # Implement the metrics of the scenario
        raise NotImplementedError

    def _print_evaluate(self, sample: Sample, data, print_fn=print):
        # (optional) Implement to print information about the "performance" of a sample during validation
        raise NotImplementedError
