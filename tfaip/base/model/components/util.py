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
import tensorflow as tf
import logging

from tensorflow.python.ops.math_ops import to_float

logger = logging.getLogger(__name__)


# taken from tensorflow/tensor2tensor
def cast_like(x, y):
    """Cast x to y's dtype, if necessary."""
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    if x.dtype.base_dtype == y.dtype.base_dtype:
        return x

    cast_x = tf.cast(x, y.dtype)
    if cast_x.device != x.device:
        x_name = "(eager Tensor)"
        try:
            x_name = x.name
        except AttributeError:
            pass
        logger.warning("Cast for %s may induce copy from '%s' to '%s'", x_name,
                       x.device, cast_x.device)
    return cast_x


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i, dim in enumerate(static):
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def top_kth_iterative(x, k):
    """Compute the k-th top element of x on the last axis iteratively.
    This assumes values in x are non-negative, rescale if needed.
    It is often faster than tf.nn.top_k for small k, especially if k < 30.
    Note: this does not support back-propagation, it stops gradients!
    Args:
      x: a Tensor of non-negative numbers of type float.
      k: a python integer.
    Returns:
      a float tensor of the same shape as x but with 1 on the last axis
      that contains the k-th largest number in x.
    """

    # The iterative computation is as follows:
    #
    # cur_x = x
    # for _ in range(k):
    #   top_x = maximum of elements of cur_x on the last axis
    #   cur_x = cur_x where cur_x < top_x and 0 everywhere else (top elements)
    #
    # We encode this computation in a TF graph using tf.foldl, so the inner
    # part of the above loop is called "next_x" and tf.foldl does the loop.
    def next_x(cur_x, _):
        top_x = tf.reduce_max(cur_x, axis=-1, keep_dims=True)
        return cur_x * to_float(cur_x < top_x)

    # We only do k-1 steps of the loop and compute the final max separately.
    fin_x = tf.foldl(next_x, tf.range(k - 1), initializer=tf.stop_gradient(x),
                     parallel_iterations=2, back_prop=False)
    return tf.stop_gradient(tf.reduce_max(fin_x, axis=-1, keep_dims=True))
