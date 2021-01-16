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
from typing import TYPE_CHECKING

from tensorflow import keras
import tensorflow.keras.backend as K

from tfaip.base.imports import GraphBase
from tfaip.base.model.components.conv import Conv2D
from tfaip.base.model.components.ff_layer import FF
from tfaip.base.model.components.pool import MaxPool2D

if TYPE_CHECKING:
    from tfaip.scenario.tutorial.full.model import ModelParams


class ConvLayersGraph(GraphBase):
    @classmethod
    def params_cls(cls):
        # Parameter class for the graph, local import to prevent recursive imports
        from tfaip.scenario.tutorial.full.model import ModelParams
        return ModelParams

    def __init__(self, params: 'ModelParams', name='conv', **kwargs):
        super(ConvLayersGraph, self).__init__(params, name=name, **kwargs)
        self._params = params

        # Create all layers
        self.conv1 = Conv2D(kernel_size=(2, 2), filters=16, strides=(1, 1), padding='same', name='conv1')
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool1')
        self.conv2 = Conv2D(kernel_size=(2, 2), filters=32, strides=(1, 1), padding='same', name='conv2')
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool2')
        self.flatten = keras.layers.Flatten()
        self.ff = FF(out_dimension=128, name='f_ff', activation='relu')
        self.logits = FF(out_dimension=self._params.n_classes, activation=None, name='classify')

    def call(self, inputs, **kwargs):
        # Connect all layers and return a dict of the outputs
        rescaled_img = K.expand_dims(K.cast(inputs['img'], dtype='float32') / 255, -1)
        conv_out = self.pool2(self.conv2(self.pool1(self.conv1(rescaled_img))))
        logits = self.logits(self.ff(self.flatten(conv_out)))
        pred = K.softmax(logits, axis=-1)
        cls = K.argmax(pred, axis=-1)
        out = {'pred': pred, 'logits': logits, 'class': cls}
        return out
