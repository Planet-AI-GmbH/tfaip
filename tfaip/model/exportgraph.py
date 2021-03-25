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
"""Definition of ExportGraph"""
from typing import Dict

import tensorflow as tf


class ExportGraph:
    """
    An export graph defines a Graph that will be exported by the ScenarioBase.
    This graph is defined by a label and its inputs and outputs.

    The default export graph is the prediction graph which comprises all inputs and outputs of the model.
    This class allows to define additional graphs, which might also only export a subgraph.

    See Also:
        ScenarioBase.export
        ModelBase._export_graphs
    """

    def __init__(self,
                 label: str,
                 inputs: Dict[str, tf.Tensor],
                 outputs: Dict[str, tf.Tensor],
                 ):
        self.label = label
        self.inputs = inputs
        self.outputs = outputs
        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
