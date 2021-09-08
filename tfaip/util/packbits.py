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
from typing import Union

import numpy as np
import tensorflow as tf

from tfaip.util.math.shape_utils import combined_static_and_dynamic_shape as to_shape


# def is_packable(array, dtype):
#     if t is None:
#         raise Exception(f"unknown type, use one of {len_dict.keys()}")
#     values = set(np.unique(array))
#     del values[0]
#     del values[1]
#     if len(values) > 0:
#         raise Exception(f"found invalid values in array, expect only 0,1 but also found {values}")
#     return array.shape[-1] <= t


def packsize(size: int):
    return (size - 1) // 8 + 1


def packbits(array: np.ndarray):
    """
    returns an array with same number of dimensions, but last dimension has lower dimension.
    8 values are composed together such that the new last dimension has size_new =  (size_old - 1) // 8 + 1
    @param array:
    @return:
    """
    if array.shape[-1] > 8:
        stacks = np.split(array, [8 * i for i in range(1, packsize(array.shape[-1]))], axis=-1)
        res = []
        for part in stacks:
            res.append(packbits(part))
        return np.concatenate(res, axis=-1)
    return np.packbits(array, axis=-1)


def unpackbits(array: Union[np.ndarray, tf.Tensor], length: int) -> Union[np.ndarray, tf.Tensor]:
    """
    undo the packbits-method
    @param array:
    @param length:
    @return:
    """
    if isinstance(array, np.ndarray):
        if length > 8:
            stack = []
            for i in range(array.shape[-1]):
                stack.append(unpackbits(array[..., i : i + 1], min(length, 8)))
                length -= 8
            return np.concatenate(stack, axis=-1)
        return np.unpackbits(array, count=length, axis=-1)
    if length > 8:
        stack = []
        for i in range(array.shape[-1]):
            stack.append(unpackbits(tf.expand_dims(array[..., i], axis=-1), min(length, 8)))
            length -= 8
        return tf.concat(stack, axis=-1)
    b = tf.constant([2 ** (8 - s - 1) for s in range(length)], dtype=tf.uint8)
    shape = to_shape(array)
    shape[-1] = length
    casted = tf.cast(array[..., None] // b, "int32")
    floored = tf.math.floormod(casted, 2)
    return tf.reshape(floored, shape)
