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
"""How to create a graph with different training and prediction graph.

This is for example required for S2S-setups where training is performed with teacher forcing
and prediction with beam search decoding (or best path).
"""
from dataclasses import dataclass

from paiargparse import pai_dataclass
import tensorflow as tf

from tfaip import ModelBaseParams
from tfaip.model.graphbase import GenericGraphBase


@pai_dataclass
@dataclass
class MyModelParams(ModelBaseParams):
    @staticmethod
    def cls():
        raise NotImplementedError  # Model is not implemented in this howto

    def graph_cls(self):
        return MyGraph


# Inherit from GenericGraphBase instead of GraphBase
# Basically the graph implements two calls, one to setup the training graph, one to setup the prediction graph
class MyGraph(GenericGraphBase[MyModelParams]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shared_layer = tf.keras.layers.Dense(100)

    def build_train_graph(self, inputs, targets=None, training=None):
        # For example use teacher forcing in a S2S-setup
        return self.shared_layer(inputs) * 2

    def build_prediction_graph(self, inputs):
        # For example use beam search decoding in a S2S-setup
        return self.shared_layer(inputs)
