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
from tensorflow.keras.layers import AveragePooling2D as TfAvgPool2D
from tensorflow.keras.layers import AveragePooling3D as TfAvgPool3D
from tensorflow.keras.layers import MaxPool2D as TfMaxPool2D
from tensorflow.keras.layers import MaxPool3D as TfMaxPool3D


class MaxPool2D(keras.layers.MaxPool2D):
    def __init__(self, **kwargs):
        super(MaxPool2D, self).__init__(**kwargs)

    def call(self, input, mask=None, **kwargs):
        if mask is not None:
            assert (len(input.get_shape()) == len(mask.get_shape()))
            input -= (1 - mask) * 1e10

        return super(MaxPool2D, self).call(input)


class AveragePool2D(keras.layers.AveragePooling2D):
    def __init__(self, **kwargs):
        super(AveragePool2D, self).__init__(**kwargs)

    def call(self, input, mask=None, **kwargs):
        if mask is not None:
            assert (len(input.get_shape()) == len(mask.get_shape()))
            input *= mask

        return super(AveragePool2D, self).call(input)


class Pool2D(keras.layers.Layer):
    def __init__(self,
                 pool_size=(2, 2),
                 strides=None,
                 padding='valid',
                 method: str = 'max',
                 data_format=None,
                 **kwargs):
        super(Pool2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.method = method.lower()
        self.data_format = data_format
        self._kwargs = kwargs
        assert self.method in ['max', 'avg', 'maximum', 'average']
        layer_cls = TfMaxPool2D if self.method in ['max', 'maximum'] else TfAvgPool2D
        self._layer = layer_cls(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            **self._kwargs)

    def call(self, input, **kwargs):
        return self._layer(input, **kwargs)


class Pool3D(keras.layers.Layer):
    def __init__(self,
                 pool_size=(2, 2, 2),
                 strides=None,
                 padding='same',
                 method: str = 'max',
                 data_format=None,
                 **kwargs):
        super(Pool3D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.method = method.lower()
        self.data_format = data_format
        self._kwargs = kwargs
        assert self.method in ['max', 'avg', 'maximum', 'average']
        layer_cls = TfMaxPool3D if self.method in ['max', 'maximum'] else TfAvgPool3D
        self.layer = layer_cls(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            **self._kwargs)

    def call(self, input, **kwargs):
        return self.layer(input, **kwargs)


class AvgPool2D(keras.layers.AvgPool2D):
    def __init__(self, **kwargs):
        super(AvgPool2D, self).__init__(**kwargs)

    def call(self, input, mask=None, **kwargs):
        if mask is not None:
            assert(len(input.get_shape()) == len(mask.get_shape()))
            input -= (1 - mask) * 1e10

        return super(AvgPool2D, self).call(input)
