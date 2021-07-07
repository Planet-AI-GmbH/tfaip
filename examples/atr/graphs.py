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
from tensorflow.python.ops import ctc_ops
from tfaip.model.graphbase import GraphBase
import tensorflow.keras as keras
import tensorflow as tf

from examples.atr.model import ATRModelParams
from examples.atr.params import Keys


class ATRGraph(GraphBase[ATRModelParams]):
    def __init__(self, params: ATRModelParams, **kwargs):
        super(ATRGraph, self).__init__(params, **kwargs)

        self.conv_layers = [
            keras.layers.Conv2D(filters, kernel_size=[2, 2], padding="same") for filters in self._params.conv_filters
        ]
        self.pool_layers = [
            keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding="same") for _ in self._params.conv_filters
        ]
        self.bilstm_layer = keras.layers.Bidirectional(
            keras.layers.LSTM(self._params.lstm_nodes, return_sequences=True)
        )
        self.dropout_layer = keras.layers.Dropout(rate=self._params.dropout)
        self.logits_layer = keras.layers.Dense(units=self._params.num_classes)

    def build_graph(self, inputs, training=None):
        image = tf.expand_dims(inputs[Keys.Image], axis=-1)  # add channel axis
        data_length = inputs[Keys.ImageLength]
        batch_size = tf.shape(image)[0]

        flowing_data = 1 - tf.cast(image, tf.float32) / 255.0  # Rescale and invert, so that black is now 1, white 0
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            flowing_data = pool(conv(flowing_data))
            data_length = (data_length + 1) // 2  # 2x2 pooling

        subsampled_height, features = flowing_data.shape[2:4]
        flowing_data = tf.reshape(flowing_data, [batch_size, -1, subsampled_height * features])
        flowing_data = tf.transpose(flowing_data, [1, 0, 2])
        flowing_data = self.bilstm_layer(flowing_data)
        flowing_data = tf.transpose(flowing_data, [1, 0, 2])
        flowing_data = self.dropout_layer(flowing_data)

        blank_last_logits = self.logits_layer(flowing_data)
        blank_last_softmax = tf.nn.softmax(blank_last_logits)
        logits = tf.roll(blank_last_logits, shift=1, axis=-1)
        softmax = tf.roll(blank_last_softmax, shift=1, axis=-1)

        greedy_decoded = ctc_ops.ctc_greedy_decoder(
            inputs=tf.transpose(blank_last_logits, perm=[1, 0, 2]),
            sequence_length=tf.cast(keras.backend.flatten(data_length), "int32"),
        )[0][0]

        return {
            "blank_last_logits": blank_last_logits,
            "blank_last_softmax": blank_last_softmax,
            "logits": logits,
            "softmax": softmax,
            "decoded": tf.sparse.to_dense(greedy_decoded, default_value=-1),
            "out_len": data_length,
        }
