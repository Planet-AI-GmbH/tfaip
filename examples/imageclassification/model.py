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
from typing import Dict, List, Tuple

import tensorflow as tf
from paiargparse import pai_dataclass

from examples.imageclassification.params import Keys
from tfaip import ModelBaseParams, Sample
from tfaip.model.modelbase import ModelBase
from tfaip.util.tftyping import AnyTensor


@pai_dataclass
@dataclass
class ICModelParams(ModelBaseParams):
    @staticmethod
    def cls():
        return ICModel

    def graph_cls(self):
        from examples.imageclassification.graphs import ICGraph

        return ICGraph

    num_classes: int = -1
    conv_filters: List[int] = field(default_factory=lambda: [16, 32, 64])
    dense: List[int] = field(default_factory=lambda: [128])
    activation: str = "relu"


class ICModel(ModelBase[ICModelParams]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acc_metric = tf.keras.metrics.Accuracy(name="acc")

    def _best_logging_settings(self) -> Tuple[str, str]:
        return "min", self.acc_metric.name

    def _loss(self, inputs, targets, outputs) -> Dict[str, AnyTensor]:
        return {
            "loss": tf.losses.sparse_categorical_crossentropy(
                y_true=targets[Keys.Target], y_pred=outputs[Keys.OutputLogits], from_logits=True
            )
        }

    def _metric(self, inputs, targets, outputs) -> List[AnyTensor]:
        return [self.acc_metric(targets[Keys.Target], outputs[Keys.OutputClass])]

    def _print_evaluate(self, sample: Sample, data, print_fn):
        print_fn(f"TARGET/PREDICTION {sample.targets[Keys.Target][0]}/{sample.outputs[Keys.OutputClass]}")
