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
#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Gundram Leifert"
__copyright__ = "Copyright 2020, Planet AI GmbH"
__credits__ = ["Gundram Leifert"]
__email__ = "gundram.leifert@planet-ai.de"

import tensorflow as tf

from tfaip.base.model.components.crop import crop_to_same_dimension


class ParallelLayer(tf.keras.layers.Layer):

    def __init__(self, *layers, crop_to_same_size=False):
        """
        computes the given input with all layer and finally concatenate the ouputs
        @param layers:
        """
        super(ParallelLayer, self).__init__()
        assert len(layers) > 0  # at least one layers should be given
        self.the_layers = layers
        self._crop_to_same_size = crop_to_same_size

    def call(self, inputs, **kwargs):
        out_sum = self.the_layers[0](inputs, **kwargs)
        for idx in range(1, len(self.the_layers)):
            out = self.the_layers[idx](inputs, **kwargs)
            if self._crop_to_same_size:
                out, out_sum = crop_to_same_dimension(out, out_sum)
            out_sum = tf.concat(out_sum, out, axis=-1)
        return inputs

class SequentialLayer(tf.keras.layers.Layer):

    def __init__(self, *layers):
        """
        stacks all layer after each other
        @param layers:
        """
        super(SequentialLayer, self).__init__()
        self.the_layers = layers

    def call(self, inputs, **kwargs):
        for l in self.the_layers:
            inputs = l(inputs, **kwargs)
        return inputs
