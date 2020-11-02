from typing import TYPE_CHECKING

from tensorflow import keras

from tfaip.base.model import GraphBase
from tfaip.base.model.components.ff_layer import FF

if TYPE_CHECKING:
    from tfaip.scenario.tutorial.model import ModelParams


class MLPLayers(GraphBase):
    @classmethod
    def params_cls(cls):
        from tfaip.scenario.tutorial.model import ModelParams
        return ModelParams

    def __init__(self, params: 'ModelParams'):
        super(MLPLayers, self).__init__(params, name="MLP")
        self.n_classes = params.n_classes
        self.flatten = keras.layers.Flatten()
        self.ff = FF(out_dimension=128, name='f_ff', activation='relu')
        self.logits = FF(out_dimension=params.n_classes, activation=None, name='classify')

    def call(self, images, **kwargs):
        return self.logits(self.ff(self.flatten(images), **kwargs), **kwargs)
