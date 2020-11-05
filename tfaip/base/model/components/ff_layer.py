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
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np


class FF(keras.layers.Layer):
    def __init__(self,
                 out_dimension,
                 activation='relu',
                 use_bias=True,
                 use_batch_norm=False,
                 init_opt=0,
                 bias_init=0.1,
                 dropout=0.0,
                 name="dense"):
        super(FF, self).__init__(name=name)
        self._out_dimension = out_dimension
        assert self._out_dimension > 0
        self._activation = activation
        self._use_bias = use_bias
        self._use_batch_norm = use_batch_norm
        self._init_opt = init_opt
        self._bias_init = bias_init
        self._dropout = dropout
        self._w = None
        self._b = None
        self._batch_norm = None
        self._dropout_layer = None

        if isinstance(activation, str):
            self._activation = getattr(keras.activations, activation)

    def get_config(self):
        cfg = super(FF, self).get_config().copy()
        cfg.update({
            'out_dimension': self._out_dimension,
            'activation': self._activation,
            'use_bias': self._use_bias,
            'use_batch_norm': self._use_batch_norm,
            'init_opt': self._init_opt,
            'bias_init': self._bias_init,
            'dropout': self._dropout,
            'name': self._name,
        })
        return cfg

    def build(self, input_shape):
        stddev = 5e-2
        if self._init_opt == 0:
            stddev = np.sqrt(2.0 / (input_shape[-1] + self._out_dimension))
        if self._init_opt == 1:
            stddev = 5e-2
        if self._init_opt == 2:
            stddev = min(np.sqrt(2.0 / (input_shape[-1])), 5e-2)
        initializer = keras.initializers.RandomNormal(stddev=stddev)
        if self._init_opt < 0:
            initializer = keras.initializers.TruncatedNormal(0.0, -self._init_opt)

        self._w = self.add_weight("weights", [input_shape[-1], self._out_dimension], initializer=initializer)
        if self._use_bias:
            self._b = self.add_weight("bias", self._out_dimension, initializer=keras.initializers.Constant(value=self._bias_init))
        if self._use_batch_norm:
            self._batch_norm = keras.layers.BatchNormalization(name="batchNorm")

        if self._dropout > 0:
            self._dropout_layer = keras.layers.Dropout(rate=self._dropout)

    def call(self, inputs, **kwargs):
        if len(inputs.shape) > 2:
            # Broadcasting is required for the inputs if rank is greater than 2.
            outputs = tf.tensordot(inputs, self._w, [[len(inputs.shape) - 1], [0]], name="Wx")
            # Reshape the output back to the original ndim of the input.
            # shape = tf.shape(inputs)
            shape = inputs.shape.as_list()
            output_shape = shape[:-1] + [self._out_dimension]
            outputs.set_shape(output_shape)
        else:
            outputs = tf.matmul(inputs, self._w, name='Wx')

        if self._use_bias:
            outputs = tf.nn.bias_add(outputs, self._b, name='preActivation')

        if self._use_batch_norm:
            outputs = self._batch_norm(outputs, **kwargs)

        if self._activation:
            outputs = self._activation(outputs)

        if self._dropout_layer:
            outputs = self._dropout_layer(outputs)

        return outputs


# alias
Dense = FF
