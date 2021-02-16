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
import tensorflow.keras.backend as K


def rnn_transpose_time(x):
    return K.permute_dimensions(x, [1, 0, 2])


class BRNNDirectionMergeSum(keras.layers.Layer):
    def __init__(self, name='brnn_merge_sum', time_major_in=True, time_major_out=True):
        super(BRNNDirectionMergeSum, self).__init__(name=name)
        self._time_major_in = time_major_in
        self._time_major_out = time_major_out

    def call(self, inputs, **kwargs):
        shape_static = inputs.get_shape().as_list()
        cell_size = shape_static[2] // 2
        shape_dynamic = K.shape(inputs)
        dim0 = shape_dynamic[0]
        dim1 = shape_dynamic[1]
        # [dim0, dim1, 2*cell_size] -> [dim0, dim1, 2, cell_size]
        graph_o = K.reshape(inputs, shape=[dim0, dim1, 2, cell_size])
        # [dim0, dim1, 2, cell_size] -> [dim0, dim1, cell_size]
        graph_o = K.sum(graph_o, axis=2)
        if self._time_major_in and self._time_major_out:
            return graph_o
        else:
            # Since time_major_in != time_major_out we flip the first two dimensions
            return rnn_transpose_time(graph_o)


class BRNNDirectionMergeSumToConv(keras.layers.Layer):
    def __init__(self, time_major_in=True, data_format='NHWC', name='brnn_merge_sum_conv'):
        super(BRNNDirectionMergeSumToConv, self).__init__(name=name)
        self._data_format = data_format

        self._merge_sum = BRNNDirectionMergeSum(time_major_in=time_major_in, time_major_out=False)

    def call(self, inputs, **kwargs):
        # [] -> [batch_size, max_time, cell_size]
        output = self._merge_sum(inputs)
        # [batch_size, max_time, cell_size] -> [batch_size, cell_size, max_time]
        output = K.permute_dimensions(output, [0, 2, 1])
        if self._data_format == 'NHWC':
            # [batch_size, cell_size, max_time] -> [batch_size, cell_size, max_time, 1]
            return K.expand_dims(output, axis=3)
            # [batch_size, cell_size, max_time, 1] corresponds to [batch_size, Y, X, Z], 'NHWC'
        else:
            # [batch_size, cell_size, max_time] -> [batch_size, 1, cell_size, max_time]
            return K.expand_dims(output, axis=1)
            # [batch_size, 1, cell_size, max_time] corresponds to [batch_size, Z, Y, X], 'NCHW'


class BRNNLayer(keras.layers.Layer):
    def __init__(self, n_hidden,
                 time_major=True,
                 cell_type='LSTM',
                 merge_mode='concat',
                 dropout=0.0,
                 name='b_rnn'):
        super(BRNNLayer, self).__init__(name=name)

        # activations, recurrent dropout, unroll, bias may not be changed to use cuDNN cells!
        rnn = getattr(keras.layers, cell_type)(n_hidden, time_major=time_major, return_sequences=True,
                                               dropout=dropout, name=f'{cell_type.lower()}_cell')
        self._bidirectional = keras.layers.Bidirectional(rnn, merge_mode=merge_mode, name='bidirectional')

    def call(self, inputs, **kwargs):
        out = self._bidirectional(inputs, **kwargs)
        return out
