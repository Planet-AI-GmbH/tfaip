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

# similar to https://towardsdatascience.com/hitchhikers-guide-to-residual-networks-resnet-in-keras-385ec01ec8ff

# similar to https://towardsdatascience.com/hitchhikers-guide-to-residual-networks-resnet-in-keras-385ec01ec8ff
from typing import Tuple, Optional, List, Union

import tensorflow as tf

from tfaip.base.model.components.conv import Conv3D


class SequentialBlock_3D(tf.keras.layers.Layer):

    def __init__(self,
                 depth: int,
                 filters: Union[None, int, List[int]],
                 kernel_sizes: Union[Tuple[int, int, int], List[Tuple[int, int, int]]] = (1, 1, 1),
                 # strides: Union[Tuple[int, int, int], List[Tuple[int, int, int]]] = (1, 1, 1),
                 activations: Union[Optional[str], List[Optional[str]]] = 'relu',
                 paddings: Union[str, List[str]] = 'same',
                 use_bias: Union[bool, List[bool]] = True,
                 init_opt: int = 0,
                 bias_init: float = 0.0,
                 drop_rates: Union[float, List[float]] = 0.0,
                 batch_norms: Union[bool, List[bool]] = False,
                 residual: bool = True,
                 name: str = 'ResId3D',
                 ):
        """
        TODO: make description
        @param depth:
        @param filters:
        @param kernel_sizes:
        @param activations:
        @param paddings:
        @param use_bias:
        @param init_opt:
        @param bias_init:
        @param drop_rates:
        @param batch_norms:
        @param residual:
        @param name:
        """
        super(SequentialBlock_3D, self).__init__(name=name)

        self._filters = [filters] * depth if isinstance(filters, (int, None)) else filters
        assert len(self._filters) == depth # size have to match number of components
        self._kernel_sizes = [kernel_sizes] * depth if isinstance(kernel_sizes[0], int) else kernel_sizes
        assert len(self._kernel_sizes) == depth # size have to match number of components
        self._depth = depth
        # self._strides = [strides] * depth if isinstance(strides[0], (int, None)) else strides
        self._activations = [activations] * depth if isinstance(activations, (str, None)) else activations
        assert len(self._activations) == depth # size have to match number of components
        self._paddings = [paddings] * depth if isinstance(paddings, str) else paddings
        assert len(self._paddings) == depth # size have to match number of components
        self._use_bias = [use_bias] * depth if isinstance(use_bias, bool) else use_bias
        assert len(self._use_bias) == depth # size have to match number of components
        self._init_opt = init_opt
        self._bias_init = bias_init
        self._drop_rates = [drop_rates] * depth if isinstance(drop_rates, (float, int)) else drop_rates
        assert len(self._drop_rates) == depth # size have to match number of components
        self._batch_norms = [batch_norms] * depth if isinstance(batch_norms, bool) else batch_norms
        assert len(self._batch_norms) == depth # size have to match number of components
        self._residual = residual
        if self._residual:
            # check if input and output dimension will be the same
            assert all([p.lower() == 'same' or k.count(1) == 3 for p, k in zip(self._paddings, self._kernel_sizes)])
        for struct in [self._filters, self._kernel_sizes, self._activations, self._paddings, self._use_bias,
                       self._drop_rates, self._batch_norms]:
            assert len(struct) == depth  # length does not fit to depth

        # if self._residual:
        #     # if residual should be applied, output dimension has to fit to input dimension
        #     assert self._filters[-1] in [None, currentfilters]
        #     self._filters[-1] = currentfilters
        self._mylayers: List[tf.keras.layers.Layer] = []
        # use input shape for last layer to define size
        for i, (f, k, a, p, b, d, n) in enumerate(
                zip(self._filters, self._kernel_sizes, self._activations, self._paddings, self._use_bias,
                    self._drop_rates, self._batch_norms)):
            currentfilters = f if f is not None else currentfilters
            self._mylayers.append(
                Conv3D(k, currentfilters, (1, 1, 1), a, p, b, self._init_opt, self._bias_init, d, n,
                       self.name + "_conv" + str(i)))

    def call(self, input_tensor, **kwargs):
        runner_tensor = input_tensor
        for layer in self._mylayers:
            runner_tensor = layer(runner_tensor, **kwargs)
        if self._residual:
            runner_tensor = runner_tensor + input_tensor
        return runner_tensor
