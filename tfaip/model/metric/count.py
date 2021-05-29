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
import tensorflow as tf


class Count(tf.keras.metrics.Metric):
    """Metric which counts the number of examples seen"""

    def __init__(self, name="count", dtype=tf.int64, **kwargs):
        super().__init__(name, dtype, **kwargs)
        self.count = self.add_weight(name, initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        first_tensor = tf.nest.flatten(y_true)[0]
        batch_size = tf.shape(first_tensor)[0]
        self.count.assign_add(tf.cast(batch_size, dtype=self.dtype))

    def result(self):
        return self.count
