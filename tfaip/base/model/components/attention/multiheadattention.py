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
    WindowedSelfAttention = 'WindowedSelfAttention'
    WindowedSelfRelativeAttention = 'WindowedSelfRelativeAttention'

    def create_layer(self, *args, **kwargs):
        return {
            AttentionType.DotProduct: ScaledDotProductAttention,
            AttentionType.DotProductRelative: ScaledDotProductRelativeAttention,
            AttentionType.WindowedSelfAttention: WindowedSelfAttention,
            AttentionType.WindowedSelfRelativeAttention: WindowedSelfRelativeAttention,

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
    def __init__(self,
                 softmax_axis=-1,
                 **kwargs):
        super(ScaledDotProductAttention, self).__init__(name='scaled_dot_attention', **kwargs)
        self.softmax_axis = softmax_axis

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
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=self.softmax_axis)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights


class WindowedSelfAttention(Attention):
    def __init__(self,
                 look_ahead=True,
                 width=10,
                 **kwargs):
        super(WindowedSelfAttention, self).__init__(name='sparse_scaled_dot_attention', **kwargs)
        self.look_ahead = look_ahead
        self.width = width
        # if look ahead is set, we can only observe values from the past (negative values)
        self.rng = list(range(-(self.width // 2), (self.width // 2 if self.look_ahead else 0) + 1))

    def call(self, inputs, mask=None, **kwargs):
        q, k, v = inputs

        seq_len = tf.shape(q)[2]
        rng = self.rng

        # speed up in T=1 case (decoder), roll is not required, just make the normal mat mul and keep the last n entries
        windowed_qk = tf.stack([tf.reduce_sum(q * tf.roll(k, axis=2, shift=-i), axis=-1) for i in rng], axis=0)  # W x B x H x T
        windowed_qk = windowed_qk[:, :, :, -seq_len:]

        scalar = tf.math.reciprocal(tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32)))
        scaled_attention_logits = tf.math.scalar_mul(scalar, windowed_qk)

        if mask is not None:
            mask = mask[:, :, :, -seq_len:]
            scaled_attention_logits += mask * -1e9

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=0)  # W x B x H xT
        attention_weights = tf.unstack(tf.expand_dims(attention_weights, axis=-1), axis=0)

        outputs_f = tf.stack([tf.roll(v, axis=2, shift=-i) * d for i, d in zip(rng, attention_weights)])  # W X B X T x F
        outputs_f = outputs_f[:, :, :, -seq_len:]
        outputs = tf.reduce_sum(outputs_f, axis=0)

        return outputs, None


def calculate_padding(input, scaling_factor: int = 32):
    def scale(i: int, f: int) -> int:
        return (f - i % f) % f

    shape = tf.shape(input=input)
    px = scale(tf.gather(shape, 1), scaling_factor)
    py = scale(tf.gather(shape, 2), scaling_factor)
    px = 0
    return px, py


def pad(input_tensors):
    input, padding = input_tensors[0], input_tensors[1]
    px, py = padding
    shape = tf.keras.backend.shape(input)
    output = tf.image.pad_to_bounding_box(input, 0, 0, tf.keras.backend.gather(shape, 1) + px,
                                          tf.keras.backend.gather(shape, 2) + py)
    return output


def crop(input_tensors):
    input, padding = input_tensors[0], input_tensors[1]

    if input is None:
        return None

    three_dims = len(input.get_shape()) == 3
    if three_dims:
        input = tf.expand_dims(input, axis=-1)

    px, py = padding
    shape = tf.shape(input=input)
    output = tf.image.crop_to_bounding_box(input, 0, 0, tf.gather(shape, 1) - px, tf.gather(shape, 2) - py)
    return output


class ScaledDotProductRelativeAttention(Attention):
    def __init__(self,
                 max_relative_position=16,
                 max_relative_position_keys=-1,
                 max_relative_position_values=-1,
                 **kwargs,
                 ):
        super(ScaledDotProductRelativeAttention, self).__init__(
            name='scaled_dot_relative_attention',
            **kwargs)

        self.max_relative_position_keys = max_relative_position_keys if max_relative_position_keys > 0 else max_relative_position
        self.max_relative_position_values = max_relative_position_values if max_relative_position_values > 0 else max_relative_position
        if self.max_relative_position_keys <= 0:
            raise ValueError(f"Max relative position ({self.max_relative_position_keys}) must be > 0")
        if self.max_relative_position_values <= 0:
            raise ValueError(f"Max relative position ({self.max_relative_position_values}) must be > 0")

        self.rel_pos_lookup_k = None
        self.rel_pos_lookup_v = None

    def build(self, input_shape):
        q, k, v = input_shape
        depth = k[3]
        self.rel_pos_lookup_k = tf.keras.layers.Embedding(self.max_relative_position_keys * 2 + 1, depth, name="embedding_k")
        self.rel_pos_lookup_v = tf.keras.layers.Embedding(self.max_relative_position_values * 2 + 1, depth, name="embedding_v")

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
        relative_pos_keys = relative_positions(query_length, keys_length, self.max_relative_position_keys)
        relative_pos_values = relative_positions(query_length, keys_length, self.max_relative_position_values)

        relative_repr_keys = self.rel_pos_lookup_k(relative_pos_keys)
        relative_repr_values = self.rel_pos_lookup_v(relative_pos_values)
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

class WindowedSelfRelativeAttention(Attention):
    def __init__(self,
                 look_ahead=True,
                 width=10,
                 max_relative_position=16,
                 max_relative_position_keys=-1,
                 max_relative_position_values=-1,
                 **kwargs):
        super(WindowedSelfRelativeAttention, self).__init__(name='windowed_self_relative_attention', **kwargs)
        self.look_ahead = look_ahead
        self.width = width
        # if look ahead is set, we can only observe values from the past (negative values)
        self.rng = list(range(-(self.width // 2), (self.width // 2 if self.look_ahead else 0) + 1))
        self.max_relative_position_keys = max_relative_position_keys if max_relative_position_keys > 0 else max_relative_position
        self.max_relative_position_values = max_relative_position_values if max_relative_position_values > 0 else max_relative_position
        if self.max_relative_position_keys <= 0:
            raise ValueError(f"Max relative position ({self.max_relative_position_keys}) must be > 0")
        if self.max_relative_position_values <= 0:
            raise ValueError(f"Max relative position ({self.max_relative_position_values}) must be > 0")

        self.rel_pos_lookup_k = None
        self.rel_pos_lookup_v = None

    def build(self, input_shape):
        q, k, v = input_shape
        depth = k[3]
        self.rel_pos_lookup_k = tf.keras.layers.Embedding(self.max_relative_position_keys * 2 + 1, depth, name="embedding_k")
        self.rel_pos_lookup_v = tf.keras.layers.Embedding(self.max_relative_position_values * 2 + 1, depth, name="embedding_v")

    def relative_window_positions(self,index,shape,max_position):
        ones=tf.ones([shape[0],shape[1],shape[2]],tf.int32)
        clipped_index=tf.clip_by_value(index, -max_position, max_position)+max_position
        return ones*clipped_index

    def call(self, inputs, mask=None, **kwargs):
        q, k, v = inputs

        seq_len = tf.shape(q)[2]
        rng = self.rng

        # speed up in T=1 case (decoder), roll is not required, just make the normal mat mul and keep the last n entries
        windowed_qk = tf.stack([tf.reduce_sum(q * (tf.roll(k, axis=2, shift=-i)+self.rel_pos_lookup_k(self.relative_window_positions(i,tf.shape(k),self.max_relative_position_keys))), axis=-1) for i in rng], axis=0)  # W x B x H x T
        windowed_qk = windowed_qk[:, :, :, -seq_len:]

        scalar = tf.math.reciprocal(tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32)))
        scaled_attention_logits = tf.math.scalar_mul(scalar, windowed_qk)

        if mask is not None:
            mask = mask[:, :, :, -seq_len:]
            scaled_attention_logits += mask * -1e9

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=0)  # W x B x H xT
        attention_weights = tf.unstack(tf.expand_dims(attention_weights, axis=-1), axis=0)

        outputs_f = tf.stack([(tf.roll(v, axis=2, shift=-i)+self.rel_pos_lookup_v(self.relative_window_positions(i,tf.shape(v),self.max_relative_position_values))) * d for i, d in zip(rng, attention_weights)])  # W X B X T x F
        outputs_f = outputs_f[:, :, :, -seq_len:]
        outputs = tf.reduce_sum(outputs_f, axis=0)

        return outputs, None


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
        def k_q():
            range_vec_k = tf.range(length_k)
            range_vec_q = range_vec_k[-length_q:]
            return range_vec_k, range_vec_q

        def q_k():
            range_vec_q = tf.range(length_q)
            range_vec_k = range_vec_q[-length_k:]
            return range_vec_k, range_vec_q

        range_vec_k, range_vec_q = tf.cond(tf.greater(length_k, length_q), k_q, q_k)
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
