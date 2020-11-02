from typing import Type

import tensorflow as tf
from typeguard import typechecked

K = tf.keras.backend


@typechecked
def create_gradient_accumulation_optimizer(accum_steps: int, parent_optimizer: Type[tf.keras.optimizers.Optimizer],
                                           optimizer_params: dict) -> tf.keras.optimizers.Optimizer:
    if accum_steps <= 1:
        # No need to create an accumulation optimizer
        return parent_optimizer(**optimizer_params)

    # noinspection PyAbstractClass
    # We know that the parent_optimizer must not be abstract and implements all methods
    class GradientAccumulationOptimizer(parent_optimizer):
        def get_config(self):
            return super(GradientAccumulationOptimizer, self).get_config()

        def _create_slots(self, var_list):
            super(GradientAccumulationOptimizer, self)._create_slots(var_list)
            for var in var_list:
                self.add_slot(var, "accumulation")

        def __init__(self, **kwargs):
            super(GradientAccumulationOptimizer, self).__init__(**kwargs)
            self._batch = tf.Variable(1, dtype='int64', name='train_accumulation_batch_step')

        def _distributed_apply(self, distribution, grads_and_vars, name, apply_state):
            cond = tf.equal(tf.math.floormod(self._batch, accum_steps), 0)

            def update_op():
                return tf.group([self.get_slot(v, 'accumulation').assign_add(g) for g, v in grads_and_vars])

            def assign_op():
                gvs = [((g + self.get_slot(v, 'accumulation')) / accum_steps, v) for g, v in grads_and_vars]
                op = super(GradientAccumulationOptimizer, self)._distributed_apply(distribution, gvs, name, apply_state)
                with tf.control_dependencies([op]):
                    clear_op = tf.group(
                        [self.get_slot(v, 'accumulation').assign(tf.zeros(tf.shape(v))) for _, v in gvs])
                return tf.group([op, clear_op])

            cond_op = tf.cond(cond, assign_op, update_op)
            with tf.control_dependencies([cond_op]):
                return tf.group([cond_op, self._batch.assign_add(1)])

    return GradientAccumulationOptimizer(**optimizer_params)
