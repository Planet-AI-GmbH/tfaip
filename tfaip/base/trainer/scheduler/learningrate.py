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
from typing import TYPE_CHECKING

from tensorflow.keras.optimizers.schedules import LearningRateSchedule as LearningRateScheduleBase
import tensorflow.keras as keras

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops


if TYPE_CHECKING:
    from tfaip.base.trainer.scheduler.learningrate_params import LearningRateParams


class LearningRateSchedule(LearningRateScheduleBase, ABC):
    def __init__(self,
                 params: 'LearningRateParams',
                 name, *args, **kwargs):
        super(LearningRateSchedule, self).__init__(*args, **kwargs)
        self.params = params
        self.name = name
        assert self.params.steps_per_epoch_ > 0
        assert self.params.epochs_ > 0

    def get_config(self):
        return {'params': self.params.to_dict()}

    @classmethod
    def from_config(cls, config):
        from tfaip.base.trainer.scheduler.learningrate_params import LearningRateParams
        config['params'] = LearningRateParams.from_dict(config['params'])
        return cls(**config)

    def __call__(self, step):
        epoch = step // self.params.steps_per_epoch_ if self.params.step_function else step / self.params.steps_per_epoch_
        return self.lr(epoch)

    @abstractmethod
    def lr(self, epoch) -> float:
        raise NotImplemented


class ExpDecay(LearningRateSchedule):
    def __init__(self,
                 params: 'LearningRateParams',
                 ):
        super(ExpDecay, self).__init__(params, 'exponential_decay')
        self.exp_decay = keras.optimizers.schedules.ExponentialDecay(
            self.params.lr, self.params.learning_circle, self.params.lr_decay_rate, staircase=True, name=self.name)

    def lr(self, epoch):
        return self.exp_decay(epoch)


class FinalDecay(LearningRateSchedule):
    def __init__(self,
                 params: 'LearningRateParams',
                 ):
        super(FinalDecay, self).__init__(params, 'final_exponential_decay')

    def lr(self, epoch):
        return cosine_decay(
            self.params.lr,
            epoch,
            self.params.learning_circle,
            self.params.lr_decay_rate,
            self.params.decay_fraction,
            self.params.epochs_,
            self.params.final_epochs,
            name=self.name,
        )


class WarmupFinalDecay(LearningRateSchedule):
    def __init__(self, params):
        super(WarmupFinalDecay, self).__init__(params, 'warmup_final_decay')

    def lr(self, epoch):
        lr = warmup_cosine_decay(self.params.lr,  # learning rate
                                 epoch,  # epoch
                                 self.params.learning_circle,  # batch epoch
                                 self.params.lr_decay_rate,  # decay
                                 self.params.decay_fraction,  # alpha
                                 self.params.epochs_,
                                 self.params.final_epochs,  # finalepoch
                                 self.params.warmup_epochs,
                                 self.params.warmup_factor,
                                 name=self.name)
        return lr


class WarmupConstantFinalDecay(LearningRateSchedule):
    def __init__(self, params):
        super(WarmupConstantFinalDecay, self).__init__(params, 'warmup_constant_final_decay')

    def lr(self, epoch):
        lr = warmup_constant_cosine_decay(self.params.lr,  # learning rate
                                          epoch,  # epoch
                                          self.params.learning_circle,  # batch epoch
                                          self.params.lr_decay_rate,  # decay
                                          self.params.decay_fraction,  # alpha
                                          self.params.epochs_,
                                          self.params.final_epochs,  # finalepoch
                                          self.params.warmup_epochs,
                                          self.params.warmup_factor,
                                          self.params.constant_epochs,
                                          name=self.name)
        return lr


def cosine_decay(learn_rate,  # learning rate
                 epoch,  # epoch
                 batch,  # batch epoch
                 decay,  # decay
                 alpha,  # alpha
                 epochs,
                 final_epochs,  # finalepoch
                 delay=0,
                 name=None):
    with ops.name_scope(name, "LR_Finetune", [learn_rate, epoch]) as name:
        # learning_rate = ops.convert_to_tensor(
        #     learning_rate, name="initial_learning_rate")
        learn_rate = ops.convert_to_tensor(
            learn_rate, name="initial_learning_rate")
        dtype = tf.float32
        learn_rate = math_ops.cast(learn_rate, dtype)
        batch = math_ops.cast(batch, dtype)
        final_epochs = math_ops.cast(final_epochs, dtype)
        alpha = math_ops.cast(alpha, dtype)
        decay = math_ops.cast(decay, dtype)
        epoch = math_ops.cast(epoch, dtype)
        completed_fraction = (epoch - delay) / batch
        lam = control_flow_ops.cond(
            math_ops.less_equal(epoch, delay),
            lambda: learn_rate,
            lambda: learn_rate * (decay ** math_ops.floor(completed_fraction)))
        return control_flow_ops.cond(
            math_ops.less_equal(epoch, epochs - final_epochs),
            lambda: lam,
            lambda: lam * (alpha + (1 - alpha) * (0.5 + 0.5 * math_ops.cos(
                (epoch - epochs + final_epochs) / final_epochs * 3.14159))))


def warmup_cosine_decay(learn_rate,  # learning rate
                        epoch,  # epoch
                        batch,  # batch epoch
                        decay,  # decay
                        alpha,  # alpha
                        epochs,
                        final_epochs,  # finalepoch
                        warmup_epochs,
                        warmup_factor,
                        name=None):
    """piecewise definde function:
    from 0 to warmup_epoch: linear increas from learningrate to warmup_factor*learningrate
    from warmup_epoch to epochs - final_epochs: decay using alpha and learning circle
    from epochs - final_epochs to end: cosine cooldown like in adam final/cosine_decay"""

    if warmup_epochs > 0:
        start = learn_rate / warmup_factor
        peak = learn_rate
        learn_rate = control_flow_ops.cond(
            math_ops.less(epoch, warmup_epochs),
            lambda: start + (peak - start) / warmup_epochs * epoch,
            lambda: peak)

    return cosine_decay(learn_rate=learn_rate, epoch=epoch,
                        batch=batch,
                        decay=decay,
                        alpha=alpha,
                        epochs=epochs,
                        final_epochs=final_epochs,
                        delay=warmup_epochs,
                        name=name)


def warmup_constant_cosine_decay(learn_rate,  # learning rate
                                 epoch,  # epoch
                                 batch,  # batch epoch
                                 decay,  # decay
                                 alpha,  # alpha
                                 epochs,
                                 final_epochs,  # finalepoch
                                 warmup_epochs,
                                 warmup_factor,
                                 constant_epochs,
                                 name=None):
    """piecewise defined function:
    from 0 to warmup_epoch: linear increase from learningrate to warmup_factor*learningrate
    from warmup_epoch to epochs - final_epochs: decay using alpha and learning circle
    from epochs - final_epochs to end: cosine cooldown like in adam final/cosine_decay"""

    if warmup_epochs > 0:
        start = learn_rate / warmup_factor
        peak = learn_rate
        learn_rate = control_flow_ops.cond(
            math_ops.less(epoch, warmup_epochs),
            lambda: start + (peak - start) / warmup_epochs * epoch,
            lambda: peak)

    return cosine_decay(learn_rate=learn_rate, epoch=epoch,
                        batch=batch,
                        decay=decay,
                        alpha=alpha,
                        epochs=epochs,
                        final_epochs=final_epochs,
                        delay=warmup_epochs+constant_epochs,
                        name=name)

