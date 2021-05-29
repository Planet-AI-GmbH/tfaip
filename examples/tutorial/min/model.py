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
from tfaip.imports import ModelBaseParams, ModelBase
from tfaip.util.tftyping import AnyTensor


@pai_dataclass
@dataclass
class TutorialModelParams(ModelBaseParams):
    n_classes: int = field(
        default=10, metadata=pai_meta(help="The number of classes (depends on the selected dataset)")
    )

    @staticmethod
    def cls():
        return TutorialModel

    def graph_cls(self):
        from examples.tutorial.min.graphs import TutorialGraph

        return TutorialGraph


class TutorialModel(ModelBase[TutorialModelParams]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # setup the metrics and losses for the later usage
        self.metric_acc = keras.metrics.Accuracy("acc")
        self.scc_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="loss/cross-entropy")

    def _best_logging_settings(self):
        # Logging the model with the best ("max") accuracy ("acc")
        # The first argument is either "min" or "max", the second argument refers to a metric which is defined below
        return "max", "acc"

    def _loss(self, inputs, targets, outputs) -> Dict[str, AnyTensor]:
        # call the loss function and return its name and the resulting value as a dict
        return {self.scc_loss.name: self.scc_loss(targets["gt"], outputs["logits"])}

    def _metric(self, inputs, targets, outputs):
        # Metric of the model
        # The accuracy (called 'acc') is computed by using the 'gt' node of the dataset and the 'class' of the graph
        return [self.metric_acc(targets["gt"], outputs["class"])]

    def _print_evaluate(self, sample: Sample, data, print_fn=print):
        # This optional function can be used to nicely print the data at the end of a epoch on the validation data
        # Here, the prediction and ground truth is printed and whether it is correct
        outputs, targets = sample.outputs, sample.targets
        correct = outputs["class"] == targets["gt"]
        print_fn(
            f"PRED/GT: {outputs['class']}{'==' if correct else '!='}{targets['gt']} (p = {outputs['pred'][outputs['class']]})"
        )
