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
"""Definition of MetricDefinition"""
from typing import NamedTuple, Any


class MetricDefinition(NamedTuple):
    """
    A simple metric, e.g. keras.metrics.Accuracy.
    Such a metric has access to one target, one output and the sample weights.

    Attributes:
        target (str): the dictionary key of the target which will be passed to metric.
        output (str): the dictionary key of the models output
        metric: The keras metric
    """
    target: str
    output: str
    metric: Any


