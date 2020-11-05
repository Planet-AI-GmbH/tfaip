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

from tfaip.base.model.components.util import shape_list
from tfaip.util.enum import StrEnum


class AttentionType(StrEnum):
    DotProduct = 'DotProduct'
    DotProductRelative = 'DotProductRelative'

    def create_layer(self, *args, **kwargs):
        return {
            AttentionType.DotProduct: ScaledDotProductAttention,
            AttentionType.DotProductRelative: ScaledDotProductRelativeAttention,
        }[self](*args, **kwargs)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, attention_type=AttentionType.DotProduct, name="mha", **kwargs):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, name='wq')
        self.wk = tf.keras.layers.Dense(d_model, name='wk')
        self.wv = tf.keras.layers.Dense(d_model, name='wv')

        self.dense = tf.keras.layers.Dense(d_model, name='dense')

        self.attention_type = attention_type
        self.attention_layer = attention_type.create_layer(**kwargs)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, single_step=False, training=None, mask=None):
        q = inputs['q']
        k = inputs['k']
        v = inputs['v']

        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.attention_layer((q, k, v), mask=mask, single_step=single_step)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class Attention(tf.keras.layers.Layer):
    def __init__(self, name,
                 **kwargs,
                 ):
        super(Attention, self).__init__(name=name)


class ScaledDotProductAttention(Attention):
    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__(name='scaled_dot_attention', **kwargs)

    def call(self, inputs, mask=None, **kwargs):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
          q: query shape == (..., seq_len_q, depth)
          k: key shape == (..., seq_len_k, depth)
          v: value shape == (..., seq_len_v, depth_v)
          mask: Float tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          output, attention_weights
        """
        q, k, v = inputs

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        scalar = tf.math.reciprocal(tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32)))
        scaled_attention_logits = tf.math.scalar_mul(scalar, matmul_qk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights


class ScaledDotProductRelativeAttention(Attention):
    def __init__(self,
                 max_relative_position=16,
                 **kwargs,
                 ):
        super(ScaledDotProductRelativeAttention, self).__init__(
            name='scaled_dot_relative_attention',
            **kwargs)

        self.max_relative_position = max_relative_position
        if not max_relative_position:
            raise ValueError("Max relative position (%s) should be > 0 when using "
                             "relative self attention." % (max_relative_position))

        self.rel_pos_lookup_k = None
        self.rel_pos_lookup_v = None

    def build(self, input_shape):
        q, k, v = input_shape
        vocab_size = self.max_relative_position * 2 + 1
        depth = k[3]
        self.rel_pos_lookup_k = tf.keras.layers.Embedding(vocab_size, depth, name="embedding_k")
        self.rel_pos_lookup_v = tf.keras.layers.Embedding(vocab_size, depth, name="embedding_v")

    def call(self, inputs, mask=None, single_step=False):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.
        Args:
          q: query shape == (..., seq_len_q, depth)
          k: key shape == (..., seq_len_k, depth)
          v: value shape == (..., seq_len_v, depth_v)
          mask: Float tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.
        Returns:
          output, attention_weights
        """
        q, k, v = inputs
        keys_length = tf.shape(k)[2]
        query_length = tf.shape(q)[2]
        relative_pos = relative_positions(query_length, keys_length, self.max_relative_position)
        relative_repr_keys = self.rel_pos_lookup_k(relative_pos)
        relative_repr_values = self.rel_pos_lookup_v(relative_pos)
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        matmul_qk += matmul_with_relative_representations(q, relative_repr_keys, transpose_b=True)

        # scale matmul_qk
        scalar = tf.math.reciprocal(tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32)))
        scaled_attention_logits = tf.math.scalar_mul(scalar, matmul_qk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
        output += matmul_with_relative_representations(attention_weights, relative_repr_values)

        return output, attention_weights


def relative_positions(length_q, length_k, maximum_position):
    """Builds the relative positions.
    Args:
      length: The maximum length of the sequence.
      maximum_position: The maximum relative position to represent.
    Returns:
      Positive relative positions with shape :math:`[T or 1, T]`.
    """
    if length_q is length_k:
        range_vec_q = range_vec_k = tf.range(length_q)
    else:
        range_vec_k = tf.range(length_k)
        range_vec_q = range_vec_k[-length_q:]
    distance = range_vec_k[None, :] - range_vec_q[:, None]
    distance = tf.clip_by_value(distance, -maximum_position, maximum_position)
    return distance + maximum_position  # Return positive indices.


def matmul_with_relative_representations(a, b, transpose_b=False):  # pylint: disable=invalid-name
    """Multiplies :obj:`a` with the relative representations :obj:`b`.
    Args:
      a: Tensor with shape :math:`[B, H, T, _]`.
      b: Tensor with shape :math:`[T, T, _]`.
    Returns:
      Tensor with shape :math:`[B, H, T, T]`.
    """
    shapes = shape_list(a)
    batch, head, time = shapes[0], shapes[1], shapes[2]
    a = tf.transpose(a, perm=[2, 0, 1, 3])
    a = tf.reshape(a, [time, batch * head, -1])
    c = tf.matmul(a, b, transpose_b=transpose_b)
    c = tf.reshape(c, [time, batch, head, -1])
    c = tf.transpose(c, perm=[1, 2, 0, 3])
    return c
