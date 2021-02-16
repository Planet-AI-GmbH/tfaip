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
import tensorflow_addons.optimizers as addons_optimizer
import tensorflow as tf
K = tf.keras.backend


class WeightsMovingAverage(addons_optimizer.MovingAverage):
    def __init__(self, *args, **kwargs):
        super(WeightsMovingAverage, self).__init__(*args, **kwargs)
        self.is_avg = False

    def to_avg(self, var_list):
        if len(self._slots) == 0:
            return

        if self.is_avg:
            return

        self.is_avg = True
        self._swap(var_list)

    def to_model(self, var_list):
        if len(self._slots) == 0:
            return

        if not self.is_avg:
            return

        self.is_avg = False
        self._swap(var_list)

    def _swap(self, var_list):
        for var in var_list:
            if not var.trainable:
                continue

            try:
                avg = self.get_slot(var, "average")
            except KeyError:
                # occurs of var is not trainable...
                continue

            # swap variable but without extra memory
            K.set_value(var, var + avg)
            K.set_value(avg, var - avg)
            K.set_value(var, var - avg)
