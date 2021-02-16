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
import tensorflow.keras.backend as K
import numpy as np
from typing import Tuple, Optional

from tfaip.base.model.components.activation import activation_by_str


class ConvToRnn(keras.layers.Layer):
    def __init__(self, name='conv_to_rnn', time_major=True, data_format='NHWC'):
        super(ConvToRnn, self).__init__(name=name)
        self._time_major = time_major
        self._data_format = data_format

    def call(self, input, **kwargs):
        if self._data_format == 'NCHW':
            if self._time_major:
                # (batch_size, Z, Y, X) -> (X, batch_size, Y, Z)
                rnn_in = K.permute_dimensions(input, [3, 0, 2, 1])
            else:
                # (batch_size, Z, Y, X) -> (batch_size, X, Y, Z)
                rnn_in = K.permute_dimensions(input, [0, 3, 2, 1])
        elif self._data_format == 'NHWC':
            if self._time_major:
                # (batch_size, Y, X, Z) -> (X, batch_size, Y, Z)
                rnn_in = K.permute_dimensions(input, [2, 0, 1, 3])
            else:
                # (batch_size, Y, X, Z) -> (batch_size, X, Y, Z)
                rnn_in = K.permute_dimensions(input, [0, 2, 1, 3])
        else:
            raise Exception(f"Only NCHW and NHWC are supported, but got {self._data_format}")


        shape_static = rnn_in.get_shape().as_list()
        y = shape_static[2]
        z = shape_static[3]
        shape_dynamic = K.shape(rnn_in)
        dim0 = shape_dynamic[0]
        dim1 = shape_dynamic[1]
        # (dim0, dim1, Y, Z) -> (dim0, dim1, Y*Z)
        rnn_in = K.reshape(rnn_in, [dim0, dim1, y * z])
        # (X, batch_size, Y*Z) corresponds to [max_time, batch_size, cell_size]
        return rnn_in


class Conv2D(keras.layers.Layer):
    def __init__(self,
                 kernel_size: Tuple[int, int],
                 filters: int,
                 strides: Tuple[int, int] = (1, 1),
                 activation: Optional[str] = 'relu',
                 padding: str = 'valid',
                 use_bias: bool = True,
                 init_opt: int = 0,
                 bias_init: float = 0.1,
                 drop_rate: float = 0.0,
                 batch_norm: bool = False,
                 name: str = 'conv2d',
                 ):
        super(Conv2D, self).__init__(name=name)
        assert(filters > 0)
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.activation = activation_by_str(activation)
        self.padding = padding
        self.use_bias = use_bias
        self.init_opt = init_opt
        self.bias_init = bias_init
        self.drop_rate = drop_rate
        self.batch_norm = batch_norm

        self._conv_layer: Optional[keras.layers.Layer] = None
        self._batch_norm_layer: Optional[keras.layers.Layer] = None
        self._dropout_layer: Optional[keras.layers.Layer] = None

    def build(self, input_shape):
        kernel_shape = [self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters]
        if self.init_opt == 0:
            stddev = np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2] + kernel_shape[3]))
        elif self.init_opt == 1:
            stddev = 5e-2
        elif self.init_opt == 2:
            stddev = min(np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2])), 5e-2)
        else:
            stddev = 5e-2
        if self.init_opt < 0:
            kernel_initializer = keras.initializers.TruncatedNormal(0.0, -self.init_opt)
        else:
            kernel_initializer = keras.initializers.RandomNormal(stddev=stddev)

        self._conv_layer = keras.layers.Conv2D(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=keras.initializers.Constant(self.bias_init),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name='conv2d',
        )

        if self.batch_norm:
            self._batch_norm_layer = keras.layers.BatchNormalization(name="batchNorm")

        if self.drop_rate > 0:
            self._dropout_layer = keras.layers.Dropout(rate=self.drop_rate)

    def call(self, inputs, mask=None, **kwargs):
        # TODO: Swapped activation and batch norm compared to tf1_aip
        if mask is not None:
            assert(len(inputs.get_shape()) == len(mask.get_shape()))
            inputs *= mask
        y = self._conv_layer(inputs, **kwargs)
        if self.batch_norm:
            y = self._batch_norm_layer(y, **kwargs)
        if self.drop_rate:
            y = self._dropout_layer(y, **kwargs)

        return y
