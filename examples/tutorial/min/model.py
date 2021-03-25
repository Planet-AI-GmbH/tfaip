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
from typing import Dict

import tensorflow.keras as keras
from paiargparse import pai_meta, pai_dataclass

from tfaip import Sample
from tfaip.imports import ModelBaseParams, ModelBase, MetricDefinition, GraphBase
from tfaip.model.losses.definitions import LossDefinition


@pai_dataclass
@dataclass
class TutorialModelParams(ModelBaseParams):
    n_classes: int = field(default=10, metadata=pai_meta(
        help="The number of classes (depends on the selected dataset)"
    ))


class TutorialModel(ModelBase[TutorialModelParams]):
    def create_graph(self, params: TutorialModelParams) -> 'GraphBase':
        from examples.tutorial.min.graphs import TutorialGraph
        # Create an instance of the graph (layers will be created but are not connected yet!)
        return TutorialGraph(params)

    def _best_logging_settings(self):
        # Logging the model with the best ("max") accuracy ("acc")
        # The first argument is either "min" or "max", the second argument refers to a metric which is defined below
        return "max", "acc"

    def _loss(self) -> Dict[str, LossDefinition]:
        # Loss functions of the model that require one target and one output (use a keras metric)
        # If a loss requires more than one input use _extended_loss
        return {'cross-entropy-loss': LossDefinition('gt', 'logits', keras.losses.SparseCategoricalCrossentropy(from_logits=True))}

    def _metric(self):
        # Metric of the model
        # The accuracy (called 'acc') is computed by using the 'gt' node of the dataset and the 'class' of the graph
        return {'acc': MetricDefinition("gt", "class", keras.metrics.Accuracy())}

    def _print_evaluate(self, sample: Sample, data, print_fn=print):
        # This optional function can be used to nicely print the data at the end of a epoch on the validation data
        # Here, the prediction and ground truth is printed and whether it is correct
        outputs, targets = sample.outputs, sample.targets
        correct = outputs['class'] == targets['gt']
        print_fn(f"PRED/GT: {outputs['class']}{'==' if correct else '!='}{targets['gt']} (p = {outputs['pred'][outputs['class']]})")
