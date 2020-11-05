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

from tfaip.base.model.components.conv import Conv2D
from tfaip.base.model.components.ff_layer import FF
from tfaip.base.model.components.pool import MaxPool2D
from tfaip.scenario.tutorial.graphs.tutorialgraph import TutorialGraph

if TYPE_CHECKING:
    from tfaip.scenario.tutorial.model import ModelParams


class ConvLayers(TutorialGraph):
    def __init__(self, params: 'ModelParams'):
        super(ConvLayers, self).__init__(params, name='conv')
        self._params = params

        self.conv1 = Conv2D(kernel_size=(2, 2), filters=16, strides=(1, 1), padding='same', name='conv1')
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool1')
        self.conv2 = Conv2D(kernel_size=(2, 2), filters=32, strides=(1, 1), padding='same', name='conv2')
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool2')
        self.flatten = keras.layers.Flatten()
        self.ff = FF(out_dimension=128, name='f_ff', activation='relu')
        self.logits = FF(out_dimension=self._params.n_classes, activation=None, name='classify')

    def _call(self, images, **kwargs):
        conv_out = self.pool2(self.conv2(self.pool1(self.conv1(images, **kwargs), **kwargs), **kwargs), **kwargs)
        return self.logits(self.ff(self.flatten(conv_out), **kwargs), **kwargs)
