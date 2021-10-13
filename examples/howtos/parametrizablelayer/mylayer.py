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
"""How to create a layer with is parametrizable and extensible.

This setup can be adapted for example if there are components of a graph that shall be replaceable.

E.g. a basic `BackboneBase`, and `ResNet`, `ImageNet`, ... as children.
"""

from dataclasses import dataclass, field
from typing import Type, TypeVar

import tensorflow as tf
from paiargparse import pai_dataclass, pai_meta

from tfaip import ModelBaseParams
from tfaip.model.graphbase import GraphBase
from tfaip.model.layerbase import LayerBaseParams, LayerBase


# ===============================================
# My Layer, the basic class, can also be abstract!
# Define base params that must be implemented by each child, create a new TypeVar and use it in `MyLayer`


@pai_dataclass
@dataclass
class MyLayerParams(LayerBaseParams):
    nodes: int = field(default=100)  # Parameter that can be set from the command line

    @staticmethod
    def cls() -> Type["LayerBase"]:
        return MyLayer


# Since it shall be possible to inherit MyLayer, the parameters must be Generic.
# Either use TLP (import from layerbase.py) or a custom type var.
TMyLayerParams = TypeVar("TMyLayerParams", bound=MyLayerParams)


class MyLayer(LayerBase[TMyLayerParams]):
    def __init__(self, output_nodes: int, **kwargs):  # Static parameter to be set by the graph itself
        super().__init__(**kwargs)
        self.l1 = tf.keras.layers.Dense(self.params.nodes)
        self.l2 = tf.keras.layers.Dense(output_nodes)

    def call(self, inputs, training=None):
        return self.l2(self.l1(inputs))


# ==============================================
# My layer with extension


@pai_dataclass
@dataclass
class MyExtendedLayerParams(MyLayerParams):
    pre_layer_nodes: int = field(default=100)  # Additional parameter

    @staticmethod
    def cls() -> Type["LayerBase"]:
        return MyExtendedLayer


class MyExtendedLayer(MyLayer[MyExtendedLayerParams]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pre_layer = tf.keras.layers.Dense(self.params.pre_layer_nodes)

    def call(self, inputs, **kwargs):
        return super().call(self.pre_layer(inputs, **kwargs))


# ==============================================
# The Model that used my model


@pai_dataclass
@dataclass
class MyModelParams(ModelBaseParams):
    @staticmethod
    def cls():
        raise NotImplementedError  # Model class not added in this howto

    def graph_cls(self):
        return MyGraph

    layer: MyLayerParams = field(
        default_factory=MyLayerParams,
        metadata=pai_meta(
            choices=[MyLayerParams, MyExtendedLayerParams],  # This is optional
        ),
    )


class MyGraph(GraphBase[MyModelParams]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # create the layer, add the static arguments
        self.my_layer = self.params.layer.create(output_nodes=20)  # Independent of the actual layer

    def build_graph(self, inputs, training=None):
        return self.my_layer(inputs)
