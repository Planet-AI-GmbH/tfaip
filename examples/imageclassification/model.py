from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import tensorflow as tf
from paiargparse import pai_dataclass

from examples.imageclassification.params import Keys
from tfaip import ModelBaseParams, Sample
from tfaip.model.modelbase import ModelBase
from tfaip.util.tftyping import AnyTensor


@pai_dataclass
@dataclass
class ICModelParams(ModelBaseParams):
    @staticmethod
    def cls():
        return ICModel

    def graph_cls(self):
        from examples.imageclassification.graphs import ICGraph

        return ICGraph

    num_classes: int = -1
    conv_filters: List[int] = field(default_factory=lambda: [16, 32, 64])
    dense: List[int] = field(default_factory=lambda: [128])
    activation: str = "relu"


class ICModel(ModelBase[ICModelParams]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acc_metric = tf.keras.metrics.Accuracy(name="acc")

    def _best_logging_settings(self) -> Tuple[str, str]:
        return "min", self.acc_metric.name

    def _loss(self, inputs, targets, outputs) -> Dict[str, AnyTensor]:
        return {
            "loss": tf.losses.sparse_categorical_crossentropy(
                y_true=targets[Keys.Target], y_pred=outputs[Keys.OutputLogits], from_logits=True
            )
        }

    def _metric(self, inputs, targets, outputs) -> List[AnyTensor]:
        return [self.acc_metric(targets[Keys.Target], outputs[Keys.OutputClass])]

    def _print_evaluate(self, sample: Sample, data, print_fn):
        print_fn(f"TARGET/PREDICTION {sample.targets[Keys.Target][0]}/{sample.outputs[Keys.OutputClass]}")
