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

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from paiargparse import pai_meta, pai_dataclass

from examples.tutorial.full.graphs.cnn import ConvGraphParams
from examples.tutorial.full.graphs.mlp import MLPGraphParams
from examples.tutorial.full.graphs.tutorialgraph import TutorialGraph
from tfaip import Sample
from tfaip.imports import ModelBaseParams, ModelBase, MetricDefinition, MultiMetricDefinition, GraphBase
from examples.tutorial.full.graphs.backend import TutorialBackendParams
from tfaip.model.losses.definitions import LossDefinition
from tfaip.model.metric.multi import MultiMetric
from tfaip.trainer.callbacks.tensor_board_data_handler import TensorBoardDataHandler
from tfaip.util.tftyping import AnyTensor


@pai_dataclass
@dataclass
class TutorialModelParams(ModelBaseParams):
    n_classes: int = field(default=10, metadata=pai_meta(
        help="The number of classes (depends on the selected dataset)",
    ))
    graph: TutorialBackendParams = field(default_factory=MLPGraphParams, metadata=pai_meta(
        choices=[MLPGraphParams, ConvGraphParams],  # Optionally list valid seleictions
        help="The network architecture to apply",
    ))


class TutorialModel(ModelBase[TutorialModelParams]):
    def create_graph(self, params) -> 'GraphBase':
        return TutorialGraph(params)

    def _best_logging_settings(self):
        return "max", "acc"

    def _loss(self) -> Dict[str, LossDefinition]:
        # Loss functions of the model that require one target and one output (use a keras metric)
        # If a loss requires more than one input use _extended_loss
        return {'keras_loss': LossDefinition('gt', 'logits', keras.losses.SparseCategoricalCrossentropy(from_logits=True))}

    def _extended_loss(self, inputs_targets, outputs) -> Dict[str, AnyTensor]:
        # Extended loss functions have access to the full inputs, targets, and outputs of the model
        # Hence, they are more flexible than a keras.losses.Loss (see _loss) since an arbitrary number of tensors
        # can be used to define such a loss. The return value must be a dict of tensors
        return {'extended_loss': tf.keras.losses.sparse_categorical_crossentropy(inputs_targets['gt'], outputs['logits'], from_logits=True)}

    def _loss_weights(self) -> Optional[Dict[str, float]]:
        # Weight the losses (if desired)
        # Here, both losses compute the same, hence weighting has no effect
        return {'keras_loss': 0.5, 'extended_loss': 0.5}

    def _extended_metric(self, inputs_targets, outputs) -> Dict[str, tf.keras.layers.Layer]:
        # Extended metric is an alternative to _metric if a metric requires more than one target or output
        # Override _sample_weights to provide a weighting factor for different samples in a batch
        return {'acc': tf.keras.metrics.sparse_categorical_accuracy(inputs_targets['gt'], outputs['pred'])}

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

    def _print_evaluate(self, sample: Sample, data, print_fn=print):
        # Print informative text during validation
        outputs, targets = sample.outputs, sample.targets
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
                    # Create the image data as numpy array
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
