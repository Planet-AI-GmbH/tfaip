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
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utils used to manipulate tensor shapes."""

import tensorflow as tf


def get_feature_map_spatial_dims(feature_maps):
    """Return list of spatial dimensions for each feature map in a list.

    Args:
      feature_maps: a list of tensors where the ith tensor has shape
          [batch, height_i, width_i, depth_i].

    Returns:
      a list of pairs (height, width) for each feature map in feature_maps
    """
    feature_map_shapes = [
        combined_static_and_dynamic_shape(
            feature_map) for feature_map in feature_maps
    ]
    return [(shape[1], shape[2]) for shape in feature_map_shapes]


def fixed_padding(inputs, kernel_size, rate=1):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      rate: An integer, rate for atrous convolution.

    Returns:
      output: A tensor of size [batch, height_out, width_out, channels] with the
        input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.

    Args:
      tensor: A tensor of any type.

    Returns:
      A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_tensor_shape = tensor.shape.as_list()
    dynamic_tensor_shape = tf.shape(tensor)
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape


def check_min_image_dim(min_dim, image_tensor):
    """Checks that the image width/height are greater than some number.

    This function is used to check that the width and height of an image are above
    a certain value. If the image shape is static, this function will perform the
    check at graph construction time. Otherwise, if the image shape varies, an
    Assertion control dependency will be added to the graph.

    Args:
      min_dim: The minimum number of pixels along the width and height of the
               image.
      image_tensor: The image tensor to check size for.

    Returns:
      If `image_tensor` has dynamic size, return `image_tensor` with a Assert
      control dependency. Otherwise returns image_tensor.

    Raises:
      ValueError: if `image_tensor`'s' width or height is smaller than `min_dim`.
    """
    image_shape = image_tensor.get_shape()
    image_height = get_height(image_shape)
    image_width = get_width(image_shape)
    if image_height is None or image_width is None:
        shape_assert = tf.Assert(
            tf.logical_and(tf.greater_equal(tf.shape(image_tensor)[1], min_dim),
                           tf.greater_equal(tf.shape(image_tensor)[2], min_dim)),
            ['image size must be >= {} in both height and width.'.format(min_dim)])
        with tf.control_dependencies([shape_assert]):
            return tf.identity(image_tensor)

    if image_height < min_dim or image_width < min_dim:
        raise ValueError(
            'image size must be >= %d in both height and width; image dim = %d,%d' %
            (min_dim, image_height, image_width))

    return image_tensor


def get_batch_size(tensor_shape):
    """Returns batch size from the tensor shape.

    Args:
      tensor_shape: A rank 4 TensorShape.

    Returns:
      An integer representing the batch size of the tensor.
    """
    tensor_shape.assert_has_rank(rank=4)
    return tensor_shape[0].value


def get_height(tensor_shape):
    """Returns height from the tensor shape.

    Args:
      tensor_shape: A rank 4 TensorShape.

    Returns:
      An integer representing the height of the tensor.
    """
    tensor_shape.assert_has_rank(rank=4)
    return tensor_shape[1].value


def get_width(tensor_shape):
    """Returns width from the tensor shape.

    Args:
      tensor_shape: A rank 4 TensorShape.

    Returns:
      An integer representing the width of the tensor.
    """
    tensor_shape.assert_has_rank(rank=4)
    return tensor_shape[2].value


def get_depth(tensor_shape):
    """Returns depth from the tensor shape.

    Args:
      tensor_shape: A rank 4 TensorShape.

    Returns:
      An integer representing the depth of the tensor.
    """
    tensor_shape.assert_has_rank(rank=4)
    return tensor_shape[3].value


def assert_shape_equal(shape_a, shape_b):
    """Asserts that shape_a and shape_b are equal.

    If the shapes are static, raises a ValueError when the shapes
    mismatch.

    If the shapes are dynamic, raises a tf InvalidArgumentError when the shapes
    mismatch.

    Args:
      shape_a: a list containing shape of the first tensor.
      shape_b: a list containing shape of the second tensor.

    Returns:
      Either a tf.no_op() when shapes are all static and a tf.compat.v1.assert_equal() op
      when the shapes are dynamic.

    Raises:
      ValueError: When shapes are both static and unequal.
    """
    if (all(isinstance(dim, int) for dim in shape_a) and
            all(isinstance(dim, int) for dim in shape_b)):
        if shape_a != shape_b:
            raise ValueError('Unequal shapes {}, {}'.format(shape_a, shape_b))
        else:
            return tf.no_op()
    else:
        return tf.compat.v1.assert_equal(shape_a, shape_b)


def pad_or_clip_tensor(t, length):
    """Pad or clip the input tensor along the first dimension.

    Args:
      t: the input tensor, assuming the rank is at least 1.
      length: a tensor of shape [1]  or an integer, indicating the first dimension
        of the input tensor t after processing.

    Returns:
      processed_t: the processed tensor, whose first dimension is length. If the
        length is an integer, the first dimension of the processed tensor is set
        to length statically.
    """
    return pad_or_clip_nd(t, [length] + t.shape.as_list()[1:])


def pad_or_clip_nd(tensor, output_shape):
    """Pad or Clip given tensor to the output shape.

    Args:
      tensor: Input tensor to pad or clip.
      output_shape: A list of integers / scalar tensors (or None for dynamic dim)
        representing the size to pad or clip each dimension of the input tensor.

    Returns:
      Input tensor padded and clipped to the output shape.
    """
    tensor_shape = tf.shape(tensor)
    clip_size = [
        tf.where(tensor_shape[i] - shape > 0, shape, -1)
        if shape is not None else -1 for i, shape in enumerate(output_shape)
    ]
    clipped_tensor = tf.slice(
        tensor,
        begin=tf.zeros(len(clip_size), dtype=tf.int32),
        size=clip_size)

    # Pad tensor if the shape of clipped tensor is smaller than the expected
    # shape.
    clipped_tensor_shape = tf.shape(clipped_tensor)
    trailing_paddings = [
        shape - clipped_tensor_shape[i] if shape is not None else 0
        for i, shape in enumerate(output_shape)
    ]
    paddings = tf.stack(
        [
            tf.zeros(len(trailing_paddings), dtype=tf.int32),
            trailing_paddings
        ],
        axis=1)
    padded_tensor = tf.pad(clipped_tensor, paddings=paddings)
    output_static_shape = [
        dim if not isinstance(dim, tf.Tensor) else None for dim in output_shape
    ]
    padded_tensor.set_shape(output_static_shape)
    return padded_tensor
