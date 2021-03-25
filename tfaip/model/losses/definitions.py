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
"""Definition of LossDefinition"""
from typing import NamedTuple

from tensorflow import keras


class LossDefinition(NamedTuple):
    """
    A loss based on keras.losses.Loss, e.g., keras.losses.BinaryCrossentropy
    Such a loss has access to one target, one output and the sample weights.

    See Also:
        - ModelBase.loss
        - ModelBase.extended_loss

    Attributes:
        target (str): the dictionary key of the target which will be passed to metric.
        output (str): the dictionary key of the models output
        loss: The keras loss
    """
    target: str
    output: str
    loss: keras.losses.Loss
