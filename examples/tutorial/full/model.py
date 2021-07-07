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
from typing import Dict, Optional

import tensorflow as tf
import tensorflow.keras as keras
from paiargparse import pai_meta, pai_dataclass

from examples.tutorial.full.graphs.backend import TutorialBackendParams
from examples.tutorial.full.graphs.cnn import ConvGraphParams
from examples.tutorial.full.graphs.mlp import MLPGraphParams
from examples.tutorial.full.graphs.tutorialgraph import TutorialGraph
from examples.tutorial.full.metric import CustomMetric
from tfaip import Sample
from tfaip.imports import ModelBaseParams, ModelBase
from tfaip.util.tftyping import AnyTensor


@pai_dataclass
@dataclass
class TutorialModelParams(ModelBaseParams):
    @staticmethod
    def cls():
        return TutorialModel

    @classmethod
    def graph_cls(cls):
        return TutorialGraph

    n_classes: int = field(
        default=10,
        metadata=pai_meta(
            help="The number of classes (depends on the selected dataset)",
        ),
    )
    graph: TutorialBackendParams = field(
        default_factory=MLPGraphParams,
        metadata=pai_meta(
            choices=[MLPGraphParams, ConvGraphParams],  # Optionally list valid selections
            help="The network architecture to apply",
        ),
    )


class TutorialModel(ModelBase[TutorialModelParams]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.acc_metric = keras.metrics.Accuracy(name="acc")
        self.scc_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="keras_loss")
        self.acc_raw_metric = keras.metrics.Mean(name="raw_acc")
        self.custom_metric = CustomMetric(name="custom_acc")

    def _best_logging_settings(self):
        return "max", "acc"

    def _loss(self, inputs, targets, outputs) -> Dict[str, AnyTensor]:
        # Loss functions of the model that require one target and one output (use a keras metric)
        # If a loss requires more than one input use _extended_loss
        return {
            "keras_loss": self.scc_loss(targets["gt"], outputs["logits"]),  # either call a keras.Loss
            "raw_loss": tf.keras.losses.sparse_categorical_crossentropy(
                targets["gt"], outputs["logits"], from_logits=True
            ),  # or add a raw loss
        }

    def _loss_weights(self) -> Optional[Dict[str, float]]:
        # Weight the losses (if desired)
        # Here, both losses compute the same, hence weighting has no effect
        return {"keras_loss": 0.5, "raw_loss": 0.5}

    def _metric(self, inputs, targets, outputs):
        # Return a dict of metrics. The MetricDefinition defines the target and output which is passed to the
        # respective metric. If more than one target or output is required to compute a (custom) metric, use
        # _extended_metric instead
        gt = tf.squeeze(targets["gt"], axis=-1)
        return [
            self.custom_metric(gt, outputs["pred"]),
            self.acc_metric(gt, outputs["class"]),
            self.acc_raw_metric(tf.cast(tf.cast(gt, "int64") == outputs["class"], "float32")),
        ]

    def _print_evaluate(self, sample: Sample, data, print_fn=print):
        # Print informative text during validation
        outputs, targets = sample.outputs, sample.targets
        correct = outputs["class"] == targets["gt"]
        print_fn(
            f"PRED/GT: {outputs['class']}{'==' if correct else '!='}{targets['gt']} (p = {outputs['pred'][outputs['class']]})"
        )
