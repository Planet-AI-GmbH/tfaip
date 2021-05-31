from typing import Dict

import tensorflow as tf
from tfaip.data.data import DataBase

from examples.text.finetuningbert.datapipeline.tokenizerprocessor import TokenizerProcessorParams
from examples.text.finetuningbert.params import Keys, FTBertDataParams


class FTBertData(DataBase[FTBertDataParams]):
    @classmethod
    def default_params(cls) -> FTBertDataParams:
        p = super().default_params()
        p.pre_proc.processors = [TokenizerProcessorParams()]
        p.post_proc.run_parallel = False
        p.post_proc.processors = []
        return p

    def _input_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        return {
            Keys.InputWordIds: tf.TensorSpec([None], tf.int32),
            Keys.InputTypeIds: tf.TensorSpec([None], tf.int32),
            Keys.InputMask: tf.TensorSpec([None], tf.int32),
        }

    def _target_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        return {
            Keys.Target: tf.TensorSpec([1], tf.int32),
        }
