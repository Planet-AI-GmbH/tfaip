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
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import Dict, Any
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tfaip.base.model import ModelBaseParams, ModelBase
from tfaip.base.model.modelbase import SimpleMetric
from tfaip.base.model.util.graph_enum import create_graph_enum
from tfaip.util.argument_parser import dc_meta
from tfaip.util.typing import AnyNumpy


Graphs = create_graph_enum(__name__)


@dataclass_json
@dataclass
class ModelParams(ModelBaseParams):
    n_classes: int = field(default=10, metadata=dc_meta(
        help="The number of classes (depends on the selected dataset)"
    ))
    graph: Graphs = field(default=Graphs.MLPLayers, metadata=dc_meta(
        help="The network architecture to apply"
    ))


class TutorialModel(ModelBase):
    @staticmethod
    def get_params_cls():
        return ModelParams

    def create_graph(self, params) -> 'GraphBase':
        return params.graph.cls(params)

    def _best_logging_settings(self):
        return "max", "acc"

    def _loss(self, inputs, outputs) -> Dict[str, tf.Tensor]:
        return {'loss': tf.keras.layers.Lambda(
            lambda x: tf.keras.metrics.sparse_categorical_crossentropy(*x, from_logits=True), name='loss')(
            (inputs['gt'], outputs['logits']))}

    def _extended_metric(self, inputs, outputs) -> Dict[str, tf.keras.layers.Layer]:
        return {'acc': tf.keras.layers.Lambda(lambda x: tf.keras.metrics.sparse_categorical_accuracy(*x), name='acc')(
            (inputs['gt'], outputs['pred']))}

    def _metric(self):
        return {'simple_acc': SimpleMetric("gt", "class", keras.metrics.Accuracy())}

    def _print_evaluate(self, inputs, outputs: Dict[str, AnyNumpy], targets: Dict[str, AnyNumpy], data, print_fn=print):
        correct = outputs['class'] == targets['gt']
        print_fn(f"PRED/GT: {outputs['class']}{'==' if correct else '!='}{targets['gt']} (p = {outputs['pred'][outputs['class']]})")