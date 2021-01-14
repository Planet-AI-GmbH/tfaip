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
    """
    Linearly scales `image` to have zero mean and unit norm as described in PerImageStandardization.
    For the normalization it takes into account the 'real' shape of the tensor when it whas padded for batch-processing.
    In contrast to PerImageStandardization the input of the call-method is not only a tensor, it is a tuple
    (image , shape) with image of shape (height,width) and shape the corresponding shape.
    the shape of the output tensor equals the shape of the image tensor.
    """
    def __init__(self, name='normalize_image'):
        super(NormalizeImage, self).__init__(name=name)

        self._per_image_standardization = PerImageStandardization()

    def call(self, inputs, **kwargs):
        if len(inputs) == 2:
            image, img_width = inputs
            img_height = image.get_shape().as_list()[0]
            padded_height = image.get_shape().as_list()[0]
        else:
            image, img_width, img_height = inputs
            img_height = keras.backend.squeeze(img_height, axis=-1)
            padded_height = tf.shape(image)[0]

        img_width = keras.backend.squeeze(img_width, axis=-1)
        padded_width = tf.shape(image)[1]

        # image normalization
        image_crop = tf.image.crop_to_bounding_box(image, 0, 0, img_height, img_width)
        image_norm = self._per_image_standardization(image_crop)
        image_pad = tf.image.pad_to_bounding_box(image_norm, 0, 0, padded_height, padded_width)
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

