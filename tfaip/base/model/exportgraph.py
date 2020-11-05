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
from dataclasses import dataclass
from typing import Dict

import tensorflow as tf


class ExportGraph:
    def __init__(self,
                 label: str,
                 inputs: Dict[str, tf.Tensor],
                 outputs: Dict[str, tf.Tensor],
                 ):
        self.label = label
        self.inputs = inputs
        self.outputs = outputs
        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
