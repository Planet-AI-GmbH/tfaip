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
import numpy as np

from tfaip.base.imports import ModelBaseParams, ModelBase, MetricDefinition, MultiMetricDefinition
from tfaip.base.model.graphbase import GraphBase
from tfaip.base.model.metric.multi import MultiMetric
from tfaip.base.model.util.graph_enum import create_graph_enum
from tfaip.base.trainer.callbacks.tensor_board_data_handler import TensorBoardDataHandler
from tfaip.util.argumentparser import dc_meta
from tfaip.util.typing import AnyNumpy, AnyTensor

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
        # Return the class type of the model params for this model
        # These are the params defined above
        return ModelParams

    def create_graph(self, params) -> 'GraphBase':
        return params.graph.cls(params)

    def _best_logging_settings(self):
        return "max", "acc"

    def _loss(self, inputs, outputs) -> Dict[str, AnyTensor]:
        # Loss functions for the model (if several, you can weight them by overriding _loss_weights)
        return {'loss': tf.keras.layers.Lambda(
            lambda x: tf.keras.metrics.sparse_categorical_crossentropy(*x, from_logits=True), name='loss')(
            (inputs['gt'], outputs['logits']))}

    def _extended_metric(self, inputs, outputs) -> Dict[str, tf.keras.layers.Layer]:
        # Extended metric is an alternative to _metric if a metric requires more than one target or output
        # Override _sample_weights to provide a weighting factor for different samples in a batch
        return {'acc': tf.keras.layers.Lambda(lambda x: tf.keras.metrics.sparse_categorical_accuracy(*x), name='acc')(
            (inputs['gt'], outputs['pred']))}

    def _metric(self):
        # Return a dict of metrics. The MetricDefinition defines the target and output which is passed to the
        # respective metric. If more than one target or output is required to compute a (custom) metric, use
        # _extended_metric instead
        return {'simple_acc': MetricDefinition("gt", "class", keras.metrics.Accuracy())}

    def _multi_metric(self) -> Dict[str, MultiMetricDefinition]:
        # Example showing how to manipulate true and pred for sub metrics
        class MyMultiMetric(MultiMetric):
            def _precompute_values(self, y_true, y_pred, sample_weight):
                # Compute some intermediate values that will be used in the sub metrics
                # Here, the Identity is returned, and applied to the default keras Accuracy metrics (see below)
                return y_true, y_pred, sample_weight

        return {'multi_metric': MultiMetricDefinition('gt', 'class', MyMultiMetric([keras.metrics.Accuracy(name='macc1'), keras.metrics.Accuracy(name='macc2')]))}

    def _print_evaluate(self, inputs, outputs: Dict[str, AnyNumpy], targets: Dict[str, AnyNumpy], data, print_fn=print):
        # Print informative text during validation
        correct = outputs['class'] == targets['gt']
        print_fn(f"PRED/GT: {outputs['class']}{'==' if correct else '!='}{targets['gt']} (p = {outputs['pred'][outputs['class']]})")

    def _create_tensorboard_handler(self) -> 'TensorBoardDataHandler':
        # This tensorboard handler shows how to write image data (last batch of validation) to the Tensorboard
        # The image is the output of the conv layers
        # See TensorBoardDataHandler for further options
        class TutorialTBHandler(TensorBoardDataHandler):
            def _outputs_for_tensorboard(self, inputs, outputs) -> Dict[str, AnyTensor]:
                # List the outputs of the model that are used for the Tensorboard
                # Here, access the 'conv_out'
                return {k: v for k, v in outputs.items() if k in ['conv_out']}

            def handle(self, name, name_for_tb, value, step):
                # Override handle to state, that something other than writing a scalar must be performed
                # for a output. Value is the output of the network as numpy array
                if name == 'conv_out':
                    # Create the image data as numpt array
                    b, w, h, c = value.shape
                    ax_dims = int(np.ceil(np.sqrt(c)))
                    out_conv_v = np.zeros([b, w * ax_dims, h * ax_dims, 1])
                    for i in range(c):
                        x = i % ax_dims
                        y = i // ax_dims
                        out_conv_v[:,x*w:(x+1)*w,y*h:(y+1)*h, 0] = value[:,:,:,i]

                    # Write the image (use 'name_for_tb' and step)
                    tf.summary.image(name_for_tb, out_conv_v, step=step)
                else:
                    # The default case, write a scalar
                    super(TutorialTBHandler, self).handle(name, name_for_tb, value, step)

        return TutorialTBHandler()
