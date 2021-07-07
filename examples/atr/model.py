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
from typing import Dict, List, TYPE_CHECKING, Tuple

import Levenshtein
import tensorflow as tf
from examples.atr.params import Keys
from paiargparse import pai_dataclass
from tfaip import ModelBaseParams, Sample
from tfaip.model.modelbase import ModelBase
from tfaip.util.tftyping import AnyTensor

if TYPE_CHECKING:
    from examples.atr.data import ATRData


@pai_dataclass
@dataclass
class ATRModelParams(ModelBaseParams):
    num_classes: int = -1
    conv_filters: List[int] = field(default_factory=lambda: [20, 40])
    lstm_nodes: int = 100
    dropout: float = 0.5

    @staticmethod
    def cls():
        return ATRModel

    def graph_cls(self):
        from examples.atr.graphs import ATRGraph

        return ATRGraph


class ATRModel(ModelBase[ATRModelParams]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cer_metric = tf.keras.metrics.Mean(name="cer")

    def _best_logging_settings(self) -> Tuple[str, str]:
        return "min", self.cer_metric.name

    def _loss(self, inputs, targets, outputs) -> Dict[str, AnyTensor]:
        def to_2d_list(x):
            return tf.keras.backend.expand_dims(tf.keras.backend.flatten(x), axis=-1)

        # note: blank is last index
        return {
            "loss": tf.keras.backend.ctc_batch_cost(
                targets[Keys.Targets],
                outputs["blank_last_softmax"],
                to_2d_list(outputs["out_len"]),
                to_2d_list(targets[Keys.TargetsLength]),
            )
        }

    def _metric(self, inputs, targets, outputs) -> List[AnyTensor]:
        def compute_cer(decoded, targets, targets_length):
            # -1 is padding value which is expected to be 0, so shift all values by + 1
            greedy_decoded = tf.sparse.from_dense(decoded + 1)
            sparse_targets = tf.cast(
                tf.keras.backend.ctc_label_dense_to_sparse(
                    targets + 1, tf.cast(tf.keras.backend.flatten(targets_length), dtype="int32")
                ),
                "int32",
            )
            return tf.edit_distance(tf.cast(greedy_decoded, tf.int32), sparse_targets, normalize=True)

        return [
            self.cer_metric(
                compute_cer(outputs["decoded"], targets[Keys.Targets], targets[Keys.TargetsLength]),
                sample_weight=tf.keras.backend.flatten(targets[Keys.TargetsLength]),
            )
        ]

    def _print_evaluate(self, sample: Sample, data: "ATRData", print_fn):
        # trim the sentences, decode them , compute their CER, amd print
        targets = sample.targets
        outputs = sample.outputs

        pred_sentence = "".join([data.params.codec[i] for i in outputs["decoded"] if i >= 0])  # -1 is padding
        gt_sentence = "".join([data.params.codec[i] for i in targets[Keys.Targets][: targets[Keys.TargetsLength][0]]])
        cer = Levenshtein.distance(pred_sentence, gt_sentence) / len(gt_sentence)
        print_fn(f"\n  CER:  {cer}" + f"\n  PRED: '{pred_sentence}'" + f"\n  TRUE: '{gt_sentence}'")
