from abc import ABC, abstractmethod

from tfaip.base.model import GraphBase
import tensorflow.keras.backend as K


class TutorialGraph(GraphBase, ABC):
    @classmethod
    def params_cls(cls):
        from tfaip.scenario.tutorial.model import ModelParams
        return ModelParams

    def __init__(self, params, **kwargs):
        super(TutorialGraph, self).__init__(params, **kwargs)

    def call(self, inputs, **kwargs):
        # call function that is shared by all other graphs
        rescaled_img = K.expand_dims(K.cast(inputs['img'], dtype='float32') / 255, -1)
        logits = self._call(rescaled_img)  # call the actual graph (MLP or CNN)
        pred = K.softmax(logits, axis=-1)
        cls = K.argmax(pred, axis=-1)
        return {'pred': pred, 'logits': logits, 'class': cls}

    @abstractmethod
    def _call(self, inputs, **kwargs):
        # reimplement by actual graph
        raise NotImplementedError
