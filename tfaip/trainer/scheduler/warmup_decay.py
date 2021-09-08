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
from tfaip.trainer.scheduler.learningrate import LearningRateSchedule
import tensorflow as tf


class WarmupDecaySchedule(LearningRateSchedule):
    def lr(self, epoch):
        if self.params.warmup_epochs > 0:
            if self.params.warmup_steps > 0:
                raise ValueError("Set either warmup_epochs or warmup_steps")
            warmup_steps = self.params.steps_per_epoch * self.params.warmup_epochs
        else:
            warmup_steps = self.params.warmup_steps

        assert warmup_steps >= 0, "Warmup steps may not be negative"

        step = epoch * self.params.steps_per_epoch + 1  # start at 1, not at 0
        return self.params.lr * warmup_steps ** 0.5 * tf.minimum(step ** -0.5, step * warmup_steps ** -1.5)
