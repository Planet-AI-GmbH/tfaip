# Copyright 2020 The tfaip authors. All Rights Reserved.
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
from typing import List, Dict
from tensorflow.python.ops import summary_ops_v2
import tensorflow as tf

from tfaip.util.typing import AnyTensor


class TensorBoardDataHandler:
    def __init__(self):
        self.all_tensorboard_keys = self._tensorboard_only_metrics()

    def setup(self, inputs, outputs) -> Dict[str, AnyTensor]:
        outputs = self._outputs_for_tensorboard(inputs, outputs)
        self.all_tensorboard_keys = list(outputs.keys()) + self._tensorboard_only_metrics()
        return outputs

    def _outputs_for_tensorboard(self, inputs, outputs) -> Dict[str, AnyTensor]:
        # these outputs will be added to the logs by keras by treating them as metrics
        # They will be removed from the complete logs and then only presented to the tensorboard data handler
        return {}

    def _tensorboard_only_metrics(self) -> List[str]:
        # A list of metrics that will only be handled by the tensorboard handle
        return []

    def handle(self, name, name_for_tb, value, step):
        # Handle your output.
        summary_ops_v2.scalar(name_for_tb, value, step=step)
