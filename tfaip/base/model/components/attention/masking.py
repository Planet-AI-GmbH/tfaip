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

def _create_image_padding_mask(seq_len):
    """
    :param seq_len: int32 [batch_size]
    :return:
    """
    padding_mask = tf.sequence_mask(seq_len, tf.reduce_max(seq_len), dtype=tf.int32)
    # (batch_size, max_len) -- > (batch_size, 1, 1, max_len)
    # add extra dimensions to add the padding
    # to the attention logits.
    return tf.cast(padding_mask, dtype=tf.float32)

def _create_padding_mask(seq_len):
    """
    :param seq_len: int32 [batch_size]
    :return:
    """
    padding_mask = tf.sequence_mask(seq_len, tf.reduce_max(seq_len), dtype=tf.int32)
    # (batch_size, max_len) -- > (batch_size, 1, 1, max_len)
    # add extra dimensions to add the padding
    # to the attention logits.
    padding_mask = tf.expand_dims(padding_mask, axis=1)
    padding_mask = tf.expand_dims(padding_mask, axis=1)
    return tf.cast(1 - padding_mask, dtype=tf.float32)


def _create_look_ahead_mask(seq_len):
    """
    Look ahead mask decoder step N is only allowed decoder outputs emitted prior to N
    :param seq_len: int32 scalar
    :return:
    """
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    return tf.cast(mask, dtype=tf.float32)  # (seq_len, seq_len)
