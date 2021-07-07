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
