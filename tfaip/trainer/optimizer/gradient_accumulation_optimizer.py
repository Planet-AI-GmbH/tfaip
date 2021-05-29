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
"""Setup for GradientAccumulation"""
from typing import Type

import tensorflow as tf
from typeguard import typechecked

K = tf.keras.backend


@typechecked
def create_gradient_accumulation_optimizer(
    accum_steps: int, parent_optimizer: Type[tf.keras.optimizers.Optimizer], optimizer: dict
) -> tf.keras.optimizers.Optimizer:
    if accum_steps <= 1:
        # No need to create an accumulation optimizer
        return parent_optimizer(**optimizer)

    # noinspection PyAbstractClass
    # We know that the parent_optimizer must not be abstract and implements all methods
    class GradientAccumulationOptimizer(parent_optimizer):
        """Wrapper of the actual optimizer to add gradient accumulation functionality"""

        def _create_slots(self, var_list):
            super()._create_slots(var_list)
            for var in var_list:
                self.add_slot(var, "accumulation")

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._batch = tf.Variable(1, dtype="int64", name="train_accumulation_batch_step")

        def _distributed_apply(self, distribution, grads_and_vars, name, apply_state):
            cond = tf.equal(tf.math.floormod(self._batch, accum_steps), 0)

            def update_op():
                return tf.group([self.get_slot(v, "accumulation").assign_add(g) for g, v in grads_and_vars])

            def assign_op():
                gvs = [((g + self.get_slot(v, "accumulation")) / accum_steps, v) for g, v in grads_and_vars]
                # This super call in python2 style is required here! pylint: disable=super-with-arguments
                op = super(GradientAccumulationOptimizer, self)._distributed_apply(distribution, gvs, name, apply_state)
                with tf.control_dependencies([op]):
                    clear_op = tf.group(
                        [self.get_slot(v, "accumulation").assign(tf.zeros(tf.shape(v))) for _, v in gvs]
                    )
                return tf.group([op, clear_op])

            cond_op = tf.cond(cond, assign_op, update_op)
            with tf.control_dependencies([cond_op]):
                return tf.group([cond_op, self._batch.assign_add(1)])

    return GradientAccumulationOptimizer(**optimizer)
