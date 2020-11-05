from dataclasses import dataclass
from typing import Dict

import tensorflow as tf


class ExportGraph:
    def __init__(self,
                 label: str,
                 inputs: Dict[str, tf.Tensor],
                 outputs: Dict[str, tf.Tensor],
                 ):
        self.label = label
        self.inputs = inputs
        self.outputs = outputs
        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
