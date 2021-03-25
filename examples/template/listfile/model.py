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
from typing import Dict

from examples.template.listfile.graphs import TemplateGraph
from examples.template.listfile.params import TemplateModelParams
from tfaip import Sample
from tfaip.model.graphbase import GraphBase
from tfaip.model.losses.definitions import LossDefinition
from tfaip.model.metric.definitions import MetricDefinition
from tfaip.model.modelbase import ModelBase


class TemplateModel(ModelBase[TemplateModelParams]):
    def create_graph(self, params: TemplateModelParams) -> 'GraphBase':
        # Create an instance of the graph (layers will be created but are not connected yet!)
        return TemplateGraph(params)

    def _best_logging_settings(self):
        # Return the settings how the best model shall be detected, e.g. the maximum accuracy (acc is a metric which
        # must be defined in _metric or _extended_metric):
        # return "max", "acc"
        raise NotImplementedError

    def _loss(self) -> Dict[str, LossDefinition]:
        # Implement the loss for this scenario (alternative use _extended_loss)
        raise NotImplementedError

    def _metric(self) -> Dict[str, MetricDefinition]:
        # (optional) Implement the metric for the scenario (see also _extended_metric)
        raise NotImplementedError

    def _print_evaluate(self, sample: Sample, data, print_fn=print):
        # (optional) Implement to print information about the "performance" of a sample during validation
        raise NotImplementedError
