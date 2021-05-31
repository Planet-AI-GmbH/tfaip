from typing import Dict

import tensorflow as tf

from examples.imageclassification.datapipeline.indextoclassprocessor import IndexToClassProcessorParams
from examples.imageclassification.datapipeline.loadprocessor import LoadProcessorParams
from examples.imageclassification.datapipeline.preparesampleprocessor import PrepareSampleProcessorParams
from examples.imageclassification.datapipeline.rescaleprocessor import RescaleProcessorParams
from examples.imageclassification.params import Keys, ICDataParams
from tfaip.data.data import DataBase


class ICData(DataBase[ICDataParams]):
    @classmethod
    def default_params(cls) -> ICDataParams:
        p = super().default_params()
        p.pre_proc.processors = [
            LoadProcessorParams(),
            RescaleProcessorParams(),
            PrepareSampleProcessorParams(),
        ]
        p.post_proc.run_parallel = False
        p.post_proc.processors = [IndexToClassProcessorParams()]
        return p

    def _input_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        return {
            Keys.Image: tf.TensorSpec([self.params.image_height, self.params.image_width, 3], tf.uint8),
        }

    def _target_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        return {
            Keys.Target: tf.TensorSpec([1], tf.int32),
        }
