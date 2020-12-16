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
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf

from tfaip.util.shape_utils import combined_static_and_dynamic_shape


def slice_from_last_dim(channel, tensor, shape_as_list=None, reduce_dim=False):
    if shape_as_list is None:
        shape_as_list = combined_static_and_dynamic_shape(tensor)
    flatten = tf.reshape(tensor, [-1, shape_as_list[-1]])
    sliced = tf.slice(flatten, [0, channel], [-1, 1])
    if reduce_dim:
        return tf.reshape(sliced, shape=shape_as_list[:-1])
    else:
        return tf.reshape(sliced, shape=[*shape_as_list[:-1], 1])
