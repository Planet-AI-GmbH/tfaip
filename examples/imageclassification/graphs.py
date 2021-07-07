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
import tensorflow as tf
import tensorflow.keras as keras
from tfaip.model.graphbase import GraphBase

from examples.imageclassification.model import ICModelParams
from examples.imageclassification.params import Keys


class ICGraph(GraphBase[ICModelParams]):
    def __init__(self, params: ICModelParams, **kwargs):
        super(ICGraph, self).__init__(params, **kwargs)

        self.conv_layers = [
            keras.layers.Conv2D(filters, kernel_size=[2, 2], activation=params.activation)
            for filters in self._params.conv_filters
        ]
        self.pool_layers = [keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2]) for _ in self._params.conv_filters]
        self.flatten_layer = keras.layers.Flatten()
        self.dense_layers = [keras.layers.Dense(nodes, activation=params.activation) for nodes in params.dense]
        self.logits_layer = keras.layers.Dense(units=self._params.num_classes)

    def build_graph(self, inputs, training=None):
        flowing_data = tf.cast(inputs[Keys.Image], tf.float32) / 255.0
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            flowing_data = pool(conv(flowing_data))

        flowing_data = self.flatten_layer(flowing_data)

        for dense in self.dense_layers:
            flowing_data = dense(flowing_data)

        logits = self.logits_layer(flowing_data)
        softmax = tf.nn.softmax(logits)
        class_index = tf.argmax(softmax, axis=-1)

        return {Keys.OutputLogits: logits, Keys.OutputSoftmax: softmax, Keys.OutputClass: class_index}
