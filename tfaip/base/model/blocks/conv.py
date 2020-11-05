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
import tensorflow.keras.backend as K

from tfaip.base.model.components.attention.masking import _create_padding_mask, _create_image_padding_mask
from tfaip.base.model.components.conv import Conv2D
from tfaip.base.model.components.normalization import NormalizeImage
from tfaip.base.model.components.pool import MaxPool2D


class ConvLayers3(keras.layers.Layer):
    def __init__(self, subsampling: int, num_classes, mvn: bool = True, sobel_filter=False, drop_rate=0.0, auto_masking=True):
        super(ConvLayers3, self).__init__()
        self._subsampling = subsampling
        self._num_classes = num_classes
        self._mvn = mvn
        self._sobel_filter = sobel_filter
        self._auto_masking = auto_masking

        if subsampling == 8:
            self._subsampling = [2, 2, 2]
        elif subsampling == 12:
            self._subsampling = [3, 2, 2]
        elif subsampling == 18:
            self._subsampling = [3, 3, 2]
        else:
            raise ValueError("Subsampling {} not supported!"
                             " Choose from '8, 12, 18' instead.".format(self._flags.subsampling))

        # layers
        self._image_normalization = NormalizeImage()
        self._conv1 = Conv2D(kernel_size=(6, self._subsampling[0] + 2), filters=8, strides=(4, self._subsampling[0]),
                             activation='leaky_relu', padding='same', drop_rate=drop_rate, name='conv1')
        self._conv2 = Conv2D(kernel_size=(6, self._subsampling[1] + 2), filters=32, strides=(1, 1),
                             activation='leaky_relu', padding='same', drop_rate=drop_rate, name='conv2')
        self._pool2 = MaxPool2D(pool_size=(4, self._subsampling[1]), strides=(4, self._subsampling[1]), name='pool2', padding='same')
        self._conv3 = Conv2D(kernel_size=(3, 3), filters=64, strides=(1, 1),
                             activation='leaky_relu', padding="SAME", drop_rate=drop_rate, name='conv3')
        self._pool3 = MaxPool2D(pool_size=(1, self._subsampling[2]), strides=(1, self._subsampling[2]),
                                padding='same', name='pool3')

    def call(self, inputs, training=None, **kwargs):
        images, seq_length = inputs
        seq_length = tf.expand_dims(seq_length, axis=-1)
        shape_dynamic = tf.shape(images)
        shape_static = images.get_shape().as_list()
        images = tf.reshape(images, [shape_dynamic[0], shape_static[1], shape_dynamic[2], 1], name="imgWithChannel")

        if self._sobel_filter:
            images = concat_sobel(images)

        if self._mvn:
            images = K.map_fn(lambda e: self._image_normalization(e), elems=(images, seq_length), dtype='float32')

        def create_mask(sl):
            if self._auto_masking:
                return K.expand_dims(_create_image_padding_mask(sl))
            return None

        conv1 = self._conv1(images, training=training, mask=create_mask(seq_length))
        seq_length1 = tf.math.floordiv((seq_length + self._conv1.strides[1] - 1), self._conv1.strides[1], name='seqLength2')

        conv2 = self._conv2(conv1, training=training, mask=create_mask(seq_length1))
        pool2 = self._pool2(conv2, training=training, mask=create_mask(seq_length1))
        seq_length2 = tf.math.floordiv((seq_length1 + self._conv2.strides[1] * self._pool2.strides[1] - 1),
                                       (self._conv2.strides[1] * self._pool2.strides[1]), name='seqLength3')

        pool3 = self._pool3(self._conv3(pool2, training=training, mask=create_mask(seq_length2)), mask=create_mask(seq_length2))
        seq_length_for_cnn = tf.math.floordiv((seq_length2 + self._conv3.strides[1] * self._pool2.strides[1] - 1),
                                              (self._conv3.strides[1] * self._pool3.strides[1]), name='seqLengthForRNN')

        out_mask = create_mask(seq_length_for_cnn)
        if out_mask is not None:
            pool3 *= out_mask

        return pool3, tf.squeeze(seq_length_for_cnn, axis=-1)


def concat_sobel(images):
    f_sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
    f_sobel_x = tf.reshape(f_sobel_x, (3, 3, 1, 1))
    f_sobel_y = tf.constant([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=tf.float32)
    f_sobel_y = tf.reshape(f_sobel_y, (3, 3, 1, 1))
    sobel_x = tf.nn.conv2d(images, f_sobel_x, padding='SAME')
    sobel_y = tf.nn.conv2d(images, f_sobel_y, padding='SAME')
    return tf.concat([images, sobel_x, sobel_y], axis=-1)
