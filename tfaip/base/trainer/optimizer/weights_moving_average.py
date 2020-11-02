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

            avg = self.get_slot(var, "average")

            # swap variable but without extra memory
            K.set_value(var, var + avg)
            K.set_value(avg, var - avg)
            K.set_value(var, var - avg)
