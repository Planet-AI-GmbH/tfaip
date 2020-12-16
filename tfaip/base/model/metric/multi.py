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
from abc import ABC, abstractmethod
from typing import List, NamedTuple

from tensorflow.keras import metrics


class MultiMetric(metrics.Metric, ABC):
    def __init__(self, children: List[metrics.Metric], name='multi_metric', **kwargs):
        super(MultiMetric, self).__init__(name=name, **kwargs)
        self.children = [MulitMetricWrapper(c) for c in children]

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

    def reset_states(self):
        for c in self.children:
            c.reset_states()


class MulitMetricWrapper(metrics.Metric):
    def __init__(self, metric: metrics.Metric, **kwargs):
        super(MulitMetricWrapper, self).__init__(name=metric.name, **kwargs)
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

