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
"""Definitions of types requiring Tensorflow

Note: typing and tf-typing are split so that tensorflow is only imported in this file
This is required so that the data pipeline (Data, DataProcessors, ...) can import typing without importing tensorflow.

"""
from typing import Union

import tensorflow as tf
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

try:
    import keras.engine.keras_tensor
except ImportError:
    AnyTensor = Union[tf.Tensor, KerasTensor]
else:
    AnyTensor = Union[tf.Tensor, KerasTensor, keras.engine.keras_tensor.KerasTensor]


try:
    # tf 2.5.x
    from tensorflow.python.keras.utils.tf_utils import sync_to_numpy_or_python_type
except ImportError:
    # tf 2.4.x
    from tensorflow.python.keras.utils.tf_utils import to_numpy_or_python_type as sync_to_numpy_or_python_type
