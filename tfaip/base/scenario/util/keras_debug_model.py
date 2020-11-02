import tensorflow.keras as keras

from tfaip.base.model import ModelBase


class KerasDebugModel(keras.Model):
    def get_config(self):
        raise NotImplementedError

    def __init__(self, model: 'ModelBase'):
        super(KerasDebugModel, self).__init__()
        self.model = model

    def call(self, inputs, training=None, mask=None):
        outputs = self.model.build(inputs)
        losses = self.model.loss(inputs, outputs)
        metrics = self.model.extended_metric(inputs, outputs)
        return {**outputs, **losses, **metrics}
