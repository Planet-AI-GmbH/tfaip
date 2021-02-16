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

from tfaip.base.model.components.attention.multiheadattention import MultiHeadAttention, AttentionType


class SelfMutualAttentionPFFLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, num_heads, rate=0.1, return_attn_coef=False, name="self_mutual_attention_pff",
                 layer_norm=True, residual=True, self_attention_type=AttentionType.DotProduct,
                 **kwargs):
        super(SelfMutualAttentionPFFLayer, self).__init__(name=name, **kwargs)
        self.self_attention = SelfAttentionLayer(d_model, num_heads, rate, True, layer_norm=layer_norm, residual=residual, attention_type=self_attention_type)
        self.mutual_attention = MutualAttentionLayer(d_model, num_heads, rate, True, layer_norm=layer_norm, residual=residual)
        self.pff_layer = PFFLayer(d_model, dff, rate)

        self.return_attn_coef = return_attn_coef

    def call(self, inputs, single_step=False, mask_padding=None, mask_look_ahead=None):
        x, y = inputs
        self_att, self_att_coef = self.self_attention(x, single_step=single_step, mask=mask_look_ahead)
        att, mut_att_coef = self.mutual_attention({'q': self_att, 'kv': y}, mask=mask_padding)
        out = self.pff_layer(att)
        if self.return_attn_coef:
            return out, self_att_coef, mut_att_coef
        else:
            return out


class MutualAttentionPFFLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, num_heads, rate=0.1, return_attn_coef=False, name='mutual_attention_pff',
                 layer_norm=True, residual=True, attention_type=AttentionType.DotProduct,
                 **kwargs):
        super(MutualAttentionPFFLayer, self).__init__(name=name, **kwargs)
        self.mutual_attention = MutualAttentionLayer(d_model, num_heads, rate, True, layer_norm=layer_norm, residual=residual, attention_type=attention_type)
        self.pff_layer = PFFLayer(d_model, dff, rate, residual=residual, layer_norm=layer_norm)

        self.return_attn_coef = return_attn_coef

    def call(self, inputs, training=None, mask=None):
        att, att_coef = self.mutual_attention(inputs, training=training, mask=mask)
        out = self.pff_layer(att, training=training)
        if self.return_attn_coef:
            return out, att_coef
        else:
            return out


class SelfAttentionPFFLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, num_heads, rate=0.1, return_attn_coef=False, name='self_attention_pff',
                 layer_norm=True, residual=True, attention_type=AttentionType.DotProduct,
                 **kwargs):
        super(SelfAttentionPFFLayer, self).__init__(name=name, **kwargs)
        self.self_attention = SelfAttentionLayer(d_model=d_model, num_heads=num_heads, rate=rate,
                                                 return_attn_coef=True, layer_norm=layer_norm, residual=residual,
                                                 attention_type=attention_type)
        self.pff_layer = PFFLayer(d_model=d_model, dff=dff, rate=rate, layer_norm=layer_norm, residual=residual)

        self.return_attn_coef = return_attn_coef

    def call(self, inputs, training=None, mask=None):
        att, att_coef = self.self_attention(inputs, training=training, mask=mask)
        out = self.pff_layer(att, training=training)
        if self.return_attn_coef:
            return out, att_coef
        else:
            return out


class PFFLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, rate=0.1, name="pff",
                 layer_norm=True, residual=True,
                 **kwargs):
        super(PFFLayer, self).__init__(name=name, **kwargs)

        self.use_residual = residual

        self.ffn1 = tf.keras.layers.Dense(dff, activation='relu', name='ffn1')  # (batch_size, seq_len, dff)
        self.ffn2 = tf.keras.layers.Dense(d_model, name='ffn2')  # (batch_size, seq_len, d_model)

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm') if layer_norm else None
        self.dropout = tf.keras.layers.Dropout(rate, name='dropout')

    def call(self, inputs, **kwargs):
        ffn_output_pre = self.ffn1(inputs)  # (batch_size, input_seq_len, dff)
        ffn_output = self.ffn2(ffn_output_pre)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout(ffn_output)
        out = inputs + ffn_output if self.use_residual else ffn_output
        if self.layernorm:
            out = self.layernorm(out)  # (batch_size, input_seq_len, d_model)
        return out


class SelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, rate=0.1, return_attn_coef=False, name='self_attention',
                 layer_norm=True, residual=True, attention_type=AttentionType.DotProduct,
                 **kwargs):
        super(SelfAttentionLayer, self).__init__(name=name, **kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.rate = rate
        self.return_attn_coef = return_attn_coef
        self.use_residual = residual

        self.mha = MultiHeadAttention(d_model, num_heads, return_attn_coef=True, attention_type=attention_type)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm') if layer_norm else None
        self.dropout = tf.keras.layers.Dropout(rate, name='dropout')

    def call(self, inputs, single_step=False, mask=None):
        kv = inputs
        q = inputs[:, -1:, :] if single_step else inputs
        mask = mask[:, :, -1:, :] if single_step else mask

        attn_output, attn_weights = self.mha({'q': q, 'k': kv, 'v': kv}, mask=mask, single_step=single_step)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout(attn_output)
        out = q + attn_output if self.use_residual else attn_output
        if self.layernorm:
            out = self.layernorm(out)  # (batch_size, input_seq_len, d_model)

        if self.return_attn_coef:
            return out, attn_weights
        else:
            return out


class MutualAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, rate=0.1, return_attn_coef=False, name='mutual_attention',
                 layer_norm=True, residual=True, attention_type=AttentionType.DotProduct,
                 **kwargs):
        super(MutualAttentionLayer, self).__init__(name=name, **kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.rate = rate
        self.return_attn_coef = return_attn_coef
        self.residual = residual

        self.mha = MultiHeadAttention(d_model, num_heads, return_attn_coef=True, attention_type=attention_type)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6) if layer_norm else None
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, mask=None):
        kv = inputs['kv']
        q = inputs['q']

        attn_output, attn_weights = self.mha({'k': kv, 'v': kv, 'q': q}, mask=mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout(attn_output)
        out = q + attn_output if self.residual else attn_output
        if self.layernorm:
            out = self.layernorm(out)  # (batch_size, input_seq_len, d_model)

        if self.return_attn_coef:
            return out, attn_weights
        else:
            return out
