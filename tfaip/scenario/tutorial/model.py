from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import Dict, Any
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tfaip.base.model import ModelBaseParams, ModelBase
from tfaip.base.model.modelbase import SimpleMetric
from tfaip.base.model.util.graph_enum import create_graph_enum
from tfaip.util.argument_parser import dc_meta
from tfaip.util.typing import AnyNumpy


Graphs = create_graph_enum(__name__)


@dataclass_json
@dataclass
class ModelParams(ModelBaseParams):
    n_classes: int = field(default=10, metadata=dc_meta(
        help="The number of classes (depends on the selected dataset)"
    ))
    graph: Graphs = field(default=Graphs.MLPLayers, metadata=dc_meta(
        help="The network architecture to apply"
    ))


class TutorialModel(ModelBase):
    @staticmethod
    def get_params_cls():
        return ModelParams

    def __init__(self, params: ModelParams, *args, **kwargs):
        super(TutorialModel, self).__init__(params, *args, **kwargs)
        self._params: ModelParams = self._params  # For IntelliSense
        self.graph = self._params.graph.cls(self._params)
        self.predict_layer = keras.layers.Lambda(lambda x: K.softmax(x), name='pred')
        self.class_layer = keras.layers.Lambda(lambda x: K.argmax(x), name='class')

    def _best_logging_settings(self):
        return "max", "acc"

    def _loss(self, inputs, outputs) -> Dict[str, tf.Tensor]:
        return {'loss': tf.keras.layers.Lambda(
            lambda x: tf.keras.metrics.sparse_categorical_crossentropy(*x, from_logits=True), name='loss')(
            (inputs['gt'], outputs['logits']))}

    def _extended_metric(self, inputs, outputs) -> Dict[str, tf.keras.layers.Layer]:
        return {'acc': tf.keras.layers.Lambda(lambda x: tf.keras.metrics.sparse_categorical_accuracy(*x), name='acc')(
            (inputs['gt'], outputs['pred']))}

    def _metric(self):
        return {'simple_acc': SimpleMetric("gt", "class", keras.metrics.Accuracy())}

    def _build(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        logits = self.graph(K.expand_dims(K.cast(inputs['img'], dtype='float32') / 255, -1))
        pred = self.predict_layer(logits)
        cls = self.class_layer(pred)
        return {'pred': pred, 'logits': logits, 'class': cls}

    def _print_evaluate(self, inputs, outputs: Dict[str, AnyNumpy], targets: Dict[str, AnyNumpy], data, print_fn=print):
        correct = outputs['class'] == targets['gt']
        print_fn(f"PRED/GT: {outputs['class']}{'==' if correct else '!='}{targets['gt']} (p = {outputs['pred'][outputs['class']]})")