# Copyright 2021 The tfaip authors. All Rights Reserved.
#
# This file is part of tfaip.
#
# tfaip is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# tfaip is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# tfaip. If not, see http://www.gnu.org/licenses/.
# ==============================================================================
"""Definition of the TensorBoardDataHandler"""
from typing import List, Dict

import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2

from tfaip.util.tftyping import AnyTensor


class TensorBoardDataHandler:
    """
    The TensorBoardDataHandler allows to customize writing arbitrary data to the TensorBoard.

    Use case: Writing image (see tfaip.scenario.tutorial.full)
        - Add the raw image data (e.g. weights) as output of the model by implementing _outputs_for_tensorboard
        - Overwrite handle to write the data adapted data to the Tensorboard (e.g. tf.summary.write_image())

    Use case: PR Curve
        - The PR-Curve is a Metric of bytes.
        - Add the name of the metric to _tensorboard_only_metrics (to mark this metrik to be only added to tensorboard)
          Note, this step is optional, if the type of the metric is bytes.
        - Overwrite handle to write the actual data with the raw tensorboard writer
    """
    def __init__(self):
        self.all_tensorboard_keys = self._tensorboard_only_metrics()

    def setup(self, inputs, outputs) -> Dict[str, AnyTensor]:
        outputs = self._outputs_for_tensorboard(inputs, outputs)
        self.all_tensorboard_keys = list(outputs.keys()) + self._tensorboard_only_metrics()
        return outputs

    def _outputs_for_tensorboard(self, inputs, outputs) -> Dict[str, AnyTensor]:
        # these outputs will be added to the logs by keras by treating them as metrics
        # They will be removed from the complete logs and then only presented to the tensorboard data handler
        del inputs  # not used in the default implementation
        del outputs  # not used in the default implementation
        return {}

    def _tensorboard_only_metrics(self) -> List[str]:
        # A list of metrics that will only be handled by the tensorboard handle
        return []

    def handle(self, name, name_for_tb, value, step):
        del name  # not used in the default implementation
        # Handle your output.
        if isinstance(value, bytes):
            summary_ops_v2.write_raw_pb(value, step, name_for_tb)
        else:
            summary_ops_v2.scalar(name_for_tb, value, step=step)

    def is_tensorboard_only(self, key: str, value: AnyTensor):
        return key in self.all_tensorboard_keys or isinstance(value, bytes) or (
                    isinstance(value, tf.Tensor) and value.dtype == tf.string)
