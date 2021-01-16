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
from typing import Dict
import tensorflow as tf
import tensorflow.keras as keras

from tfaip.base.imports import ModelBaseParams, ModelBase, MetricDefinition
from tfaip.scenario.tutorial.min.graphs import ConvLayersGraph
from tfaip.util.argumentparser import dc_meta
from tfaip.util.typing import AnyNumpy, AnyTensor


@dataclass_json
@dataclass
class ModelParams(ModelBaseParams):
    n_classes: int = field(default=10, metadata=dc_meta(
        help="The number of classes (depends on the selected dataset)"
    ))


class TutorialModel(ModelBase):
    @staticmethod
    def get_params_cls():
        # Parameters of the model
        return ModelParams

    def create_graph(self, params: ModelParams) -> 'GraphBase':
        # Create an instance of the graph (layers will be created but are not connected yet!)
        return ConvLayersGraph(params)

    def _best_logging_settings(self):
        # Logging the model with the best ("max") accuracy ("acc")
        # The first argument is either "min" or "max", the second argument refers to a metric which is defined below
        return "max", "acc"

    def _loss(self, inputs, outputs) -> Dict[str, AnyTensor]:
        # Loss of the model
        # Return a keras layer which outputs the loss (multiple losses are allowed)
        # Here, a sparse categorical crossentropy is used
        return {'loss': tf.keras.layers.Lambda(
            lambda x: tf.keras.metrics.sparse_categorical_crossentropy(*x, from_logits=True), name='loss')(
            (inputs['gt'], outputs['logits']))}

    def _metric(self):
        # Metric of the model
        # The accuracy (called 'acc') is computed by using the 'gt' node of the dataset and the 'class' of the graph
        return {'acc': MetricDefinition("gt", "class", keras.metrics.Accuracy())}

    def _print_evaluate(self, inputs, outputs: Dict[str, AnyNumpy], targets: Dict[str, AnyNumpy], data, print_fn=print):
        # This optional function can be used to nicely print the data at the end of a epoch on the validation data
        # Here, the prediction and ground truth is printed and whether it is correct
        correct = outputs['class'] == targets['gt']
        print_fn(f"PRED/GT: {outputs['class']}{'==' if correct else '!='}{targets['gt']} (p = {outputs['pred'][outputs['class']]})")
