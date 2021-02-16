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


def create_image_padding_mask(seq_len):
    """
    :param seq_len: int32 [batch_size]
    :return:
    """
    padding_mask = tf.sequence_mask(seq_len, tf.reduce_max(seq_len), dtype=tf.int32)
    # (batch_size, max_len) -- > (batch_size, 1, 1, max_len)
    # add extra dimensions to add the padding
    # to the attention logits.
    return tf.cast(padding_mask, dtype=tf.float32)


def create_padding_mask(seq_len, max_len=0):
    """
    :param seq_len: int32 [batch_size]
    :return:
    """
    if seq_len.shape.rank == 2:
        seq_len = tf.squeeze(seq_len, axis=-1)
    padding_mask = tf.sequence_mask(seq_len, tf.maximum(tf.reduce_max(seq_len), max_len), dtype=tf.int32)
    # (batch_size, max_len) -- > (batch_size, 1, 1, max_len)
    # add extra dimensions to add the padding
    # to the attention logits.
    padding_mask = tf.expand_dims(padding_mask, axis=1)
    padding_mask = tf.expand_dims(padding_mask, axis=1)
    return tf.cast(1 - padding_mask, dtype=tf.float32)


def create_sparse_mask(seq_len, sparse_width, max_len=0):
    s_max = tf.maximum(tf.reduce_max(seq_len), max_len)
    padding_mask = create_padding_mask(seq_len, s_max)
    rng = list(range(-(sparse_width // 2), sparse_width // 2 + 1))
    seq_mask = tf.stack([tf.sequence_mask(seq_len + min(-i, 0), s_max, dtype=tf.float32) - tf.sequence_mask(
        max(-i, 0), s_max, dtype=tf.float32) for i in rng], axis=0)
    seq_mask = 1 - tf.expand_dims(seq_mask, axis=2)

    combined_mask = tf.maximum(tf.expand_dims(tf.squeeze(padding_mask, axis=2), axis=0), seq_mask)
    return combined_mask


def create_look_ahead_mask(seq_len):
    """
    Look ahead mask decoder step N is only allowed decoder outputs emitted prior to N
    :param seq_len: int32 scalar
    :return:
    """
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    return tf.cast(mask, dtype=tf.float32)  # (seq_len, seq_len)
