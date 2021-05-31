from dataclasses import dataclass
from typing import Dict, Tuple, List

import tensorflow as tf
from paiargparse import pai_dataclass

from examples.text.finetuningbert.params import Keys
from tfaip import ModelBaseParams, Sample
from tfaip.model.modelbase import ModelBase, TMP
from tfaip.util.tftyping import AnyTensor
from tfaip.util.typing import AnyNumpy


@pai_dataclass
@dataclass
class FTBertModelParams(ModelBaseParams):
    @staticmethod
    def cls():
        return FTBertModel

    def graph_cls(self):
        from examples.text.finetuningbert.graphs import FTBertGraph

        return FTBertGraph

    model_name: str = "albert-base-v2"


class FTBertModel(ModelBase[FTBertModelParams]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acc_metric = tf.keras.metrics.Accuracy(name="acc")

    def _best_logging_settings(self) -> Tuple[str, str]:
        return "min", self.acc_metric.name

    def _loss(self, inputs, targets, outputs) -> Dict[str, AnyTensor]:
        return {
            "ce_loss": tf.losses.sparse_categorical_crossentropy(
                y_true=targets[Keys.Target], y_pred=outputs[Keys.OutputLogits]
            )
        }

    def _metric(self, inputs, targets, outputs) -> List[AnyTensor]:
        return [self.acc_metric(targets[Keys.Target], outputs[Keys.OutputClass])]

    def _print_evaluate(self, sample: Sample, data, print_fn):
        print_fn(f"TARGET/PREDICTION {sample.targets[Keys.Target][0]}/{sample.outputs[Keys.OutputClass]}")
