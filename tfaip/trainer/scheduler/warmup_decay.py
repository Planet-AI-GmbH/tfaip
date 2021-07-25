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
