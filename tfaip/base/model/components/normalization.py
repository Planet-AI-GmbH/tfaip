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
import tensorflow.keras as keras
import tensorflow as tf


class NormalizeImage(keras.layers.Layer):
    def __init__(self, name='normalize_image'):
        super(NormalizeImage, self).__init__(name=name)

        self._per_image_standardization = PerImageStandardization()

    def call(self, inputs, **kwargs):
        image, img_length = inputs
        img_length = keras.backend.squeeze(img_length, axis=-1)

        # dynamic shape values (calculated during runtime)
        shape_dynamic = tf.shape(image)
        # static shape values (defined up-front)
        shape_static = image.get_shape().as_list()
        # image normalization
        image_crop = tf.image.crop_to_bounding_box(image, 0, 0, shape_static[0], img_length)
        image_norm = self._per_image_standardization(image_crop)
        image_pad = tf.image.pad_to_bounding_box(image_norm, 0, 0, shape_static[0], shape_dynamic[1])
        return image_pad


class PerImageStandardization(keras.layers.Layer):
    """Linearly scales `image` to have zero mean and unit norm.

    This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
    of all values in image, and
    `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`.

    `stddev` is the standard deviation of all values in `image`. It is capped
    away from zero to protect against division by 0 when handling uniform images.

    Args:
      image: 3-D tensor of shape `[height, width, channels]`.

    Returns:
      The standardized image with same shape as `image`.

    Raises:
      ValueError: if the shape of 'image' is incompatible with this function.
    """

    def __init__(self, name='per_image_standardization'):
        super(PerImageStandardization, self).__init__(name=name)

    def call(self, inputs, **kwargs):
        image = tf.convert_to_tensor(inputs, name='image')
        # image = _Assert3DImage(image)
        # num_pixels = math_ops.reduce_prod(array_ops.shape(image))

        # image = math_ops.cast(image, dtype=dtypes.float32)
        image_mean = tf.math.reduce_mean(image)

        variance = (
                tf.math.reduce_mean(tf.math.square(image)) -
                tf.math.square(image_mean))
        variance = tf.nn.relu(variance)
        stddev = tf.math.sqrt(variance)

        # Apply a minimum normalization that protects us against uniform images.
        # min_stddev = math_ops.rsqrt(1.0 * num_pixels)
        pixel_value_scale = tf.math.maximum(stddev, 0.0001)
        pixel_value_offset = image_mean

        image = tf.math.subtract(image, pixel_value_offset)
        image = tf.math.divide(image, pixel_value_scale)
        return image

