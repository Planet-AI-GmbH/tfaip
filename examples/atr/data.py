from typing import Dict

import tensorflow as tf
from tfaip.data.data import DataBase

from examples.atr.datapipeline.decoderprocessor import DecoderProcessorParams
from examples.atr.datapipeline.loadprocessor import LoadProcessorParams
from examples.atr.datapipeline.prepareprocessor import PrepareProcessorParams
from examples.atr.datapipeline.scale_to_height_processor import ScaleToHeightProcessorParams
from examples.atr.params import ATRDataParams, Keys


class ATRData(DataBase[ATRDataParams]):
    @classmethod
    def default_params(cls) -> ATRDataParams:
        p = super().default_params()
        p.pre_proc.processors = [
            LoadProcessorParams(),
            ScaleToHeightProcessorParams(),
            PrepareProcessorParams(),
        ]
        p.post_proc.run_parallel = False
        p.post_proc.processors = [
            DecoderProcessorParams(),
        ]
        return p

    def _input_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        return {
            Keys.Image: tf.TensorSpec([None, self.params.height], tf.uint8),
            Keys.ImageLength: tf.TensorSpec([1], tf.int32),
        }

    def _target_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        return {
            Keys.Targets: tf.TensorSpec([None], tf.int32),
            Keys.TargetsLength: tf.TensorSpec([1], tf.int32),
        }
