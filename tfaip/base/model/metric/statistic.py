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

import tensorflow as tf

from tfaip.util.multiprocessing.tensor_utils import slice_from_last_dim

class PerChannelMean(tf.keras.metrics.Mean):
    def __init__(self, channel, name=None):
        super(PerChannelMean, self).__init__(name=name or f"stat/mean_{channel}")
        self._channel = channel

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(PerChannelMean, self).update_state(slice_from_last_dim(self._channel, y_pred), sample_weight)


class PerChannelPosMean(tf.keras.metrics.Mean):
    def __init__(self, channel, name=None):
        super(PerChannelPosMean, self).__init__(name=name or f"stat/meanP_{channel}")
        self._channel = channel

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.cast(y_true == self._channel, tf.float32)
        if sample_weight is not None:
            mask *= sample_weight
        super(PerChannelPosMean, self).update_state(slice_from_last_dim(self._channel, y_pred), mask)


class PerChannelNegMean(tf.keras.metrics.Mean):
    def __init__(self, channel, name=None):
        super(PerChannelNegMean, self).__init__(name=name or f"stat/meanN_{channel}")
        self._channel = channel

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.cast(y_true != self._channel, dtype=tf.float32)
        if sample_weight is not None:
            mask *= sample_weight
        super(PerChannelNegMean, self).update_state(slice_from_last_dim(self._channel, y_pred), mask)

