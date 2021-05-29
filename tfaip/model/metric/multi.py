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
"""Definition of a MultiMetric"""
from abc import ABC, abstractmethod
from typing import List, NamedTuple

from tensorflow.keras import metrics


class MultiMetric(metrics.Metric, ABC):
    """`MultiMetrics` are an _optional_ extension to the standard keras metrics.

    They enable to hierarchically compute metrics that are all based on intermediate values, e.g.,
    first compute TP, FP, FN, then compute the derived metrics precision, recall, and F1.
    To use implement a `MultiMetric` overwrite `_precomputed_values` to compute derived tensors of any shape
    (e.g. dicts).
    These tensors will then be passes to the attached child-metrics that are stated upon definition of the `MultiMetric`
    """

    def __init__(self, children: List[metrics.Metric], name="multi_metric", **kwargs):
        super().__init__(name=name, **kwargs)
        self.children = [MultiMetricWrapper(c) for c in children]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred, sample_weight = self._precompute_values(y_true, y_pred, sample_weight)
        for c in self.children:
            c.wrapped_update_state(y_true, y_pred, sample_weight)

    @abstractmethod
    def _precompute_values(self, y_true, y_pred, sample_weight):
        return y_true, y_pred, sample_weight

    def reset_states(self):
        for c in self.children:
            c.reset_states()

    def result(self):
        # Unfortunately we must return something
        return 0


class MultiMetricWrapper(metrics.Metric):
    """Utility class to wrap the actual metric

    This wrapper will not call update_state, as instead wrapped_update_state is called by the MultiMetric
    """

    def __init__(self, metric: metrics.Metric, **kwargs):
        super().__init__(name=metric.name, **kwargs)
        self.metric = metric

    def update_state(self, *args, **kwargs):
        pass

    def wrapped_update_state(self, *args, **kwargs):
        r = self.metric.update_state(*args, **kwargs)
        return r

    def result(self):
        return self.metric.result()

    def reset_states(self):
        return self.metric.reset_states()


class MultiMetricDefinition(NamedTuple):
    target: str
    output: str
    metric: MultiMetric
