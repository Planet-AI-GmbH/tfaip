from abc import abstractmethod

import tensorflow.keras as keras

from tfaip.base.model import ModelBaseParams


class GraphBase(keras.layers.Layer):
    """
    This Layer can be inherited to buildup the graph (you can however chose any method you want to).
    """

    @classmethod
    @abstractmethod
    def params_cls(cls):
        raise NotImplemented

    def __init__(self, params: 'ModelBaseParams', **kwargs):
        super(GraphBase, self).__init__(**kwargs)
        self._params = params

    def get_config(self):
        cfg = super(GraphBase, self).get_config()
        cfg['params'] = self._params.to_dict()
        return cfg

    @classmethod
    def from_config(cls, config):
        config['params'] = cls.params_cls().from_dict(config['params'])
        return super(GraphBase, cls).from_config(config)
