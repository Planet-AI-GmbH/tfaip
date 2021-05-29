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
from dataclasses import dataclass, field
from typing import List

from paiargparse import pai_dataclass, pai_meta
from tensorflow.keras.layers import Dense, Flatten

from examples.tutorial.full.graphs.backend import TutorialBackendParams, TutorialBackend


@pai_dataclass(alt="MLP")
@dataclass
class MLPGraphParams(TutorialBackendParams):
    nodes: List[int] = field(default_factory=lambda: [128], metadata=pai_meta(help="Definition of the hidden layers"))
    activation: str = field(default="relu", metadata=pai_meta(help="Activation function of the hidden layers"))

    def cls(self):
        return MLPLayers(self)


class MLPLayers(TutorialBackend[MLPGraphParams]):
    def __init__(self, graph_params: MLPGraphParams, name="MLP", **kwargs):
        super(MLPLayers, self).__init__(graph_params, name=name, **kwargs)
        self.flatten = Flatten()
        self.hidden_layers = [Dense(nodes, activation=graph_params.activation) for nodes in graph_params.nodes]

    def call(self, images, **kwargs):
        out = self.flatten(images)
        for layer in self.hidden_layers:
            out = layer(out)

        return {
            "out": out,
        }
