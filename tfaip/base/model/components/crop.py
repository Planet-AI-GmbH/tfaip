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
from typing import List, Union

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tfaip.util.shape_utils import combined_static_and_dynamic_shape as to_shape
from tfaip.util.typing import AnyTensor


def _resize(tensor, diffshape, targetshape, origshape):
    offset = [e // 2 for e in diffshape]
    return array_ops.slice(
        tensor,
        array_ops.stack([0, *offset, 0]),
        array_ops.stack([origshape[0], *targetshape[1:-1], origshape[-1]])
    )


def crop_to_dimension(features: AnyTensor, shape: AnyTensor) -> AnyTensor:
    """
    crops features to the given shapes, whereas first and last dimension were not touched (because batch and #features)
    @param features: feature of format NWC NHWC NDHWC
    @param shape: same dimension as shape of features
    @return: feature cropped on BOTH sides for dimension W, HW or DHW to
    """
    if len(features.shape) + 1 == len(shape) and features.dtype.is_integer:
        # increase shape of first
        features = tf.expand_dims(features, axis=-1)
        features = crop_to_dimension(features, shape)
        features = tf.squeeze(features, axis=-1)
        return features
    shape_layer = to_shape(features)
    # assert shape_layer[0] == shape[0]  # first dimension should be size of batch und should be the same
    diff = [tf.maximum(0, x - y) for x, y in zip(shape_layer[1:-1], shape[1:-1])]

    return tf.cond(tf.reduce_sum(tf.stack(diff)) > 0, lambda: _resize(features, diff, shape, shape_layer),
                   lambda: features)


def crop_to_same_dimension(layer1: AnyTensor, layer2: AnyTensor) -> (AnyTensor, AnyTensor):
    """
    Crops both layers to same dimension
    When one tensor is 1 dimension lower than the other and the type is integer (other is of type float),
    the cropping also works since the lower-dimensional tensor is assumed to be an index-tensor which corresponds to the float vector.
    Assumes to have first dimension to be the batch-size and the last dimension to be the feature size.
    So, first and last order will stay untouched.
    for dimensions 1,...,-1 it crops diff//2 from left and diff-(diff//2) from right side.
    These dimensions can be dynamic.
    The order of the layers does not play any role.
    :param layer1:
    :param layer2:
    :return:
    """
    shape1 = to_shape(layer1)
    shape2 = to_shape(layer2)
    if len(shape1) + 1 == len(shape2) and layer1.dtype.is_integer and layer2.dtype.is_floating:
        # increase shape of first
        layer1 = tf.expand_dims(layer1, axis=-1)
        layer1, layer2 = crop_to_same_dimension(layer1, layer2)
        layer1 = tf.squeeze(layer1, axis=-1)
        return layer1, layer2
    if len(shape2) + 1 == len(shape1) and layer2.dtype.is_integer and layer1.dtype.is_floating:
        # increase shape of first
        layer2 = tf.expand_dims(layer2, axis=-1)
        layer1, layer2 = crop_to_same_dimension(layer1, layer2)
        layer2 = tf.squeeze(layer2, axis=-1)
        return layer1, layer2
    # diff = [(x - y) for x, y in zip(shape1, shape2)]
    diff1 = [tf.maximum(0, x - y) for x, y in zip(shape1[1:-1], shape2[1:-1])]
    diff2 = [tf.maximum(0, x - y) for x, y in zip(shape2[1:-1], shape1[1:-1])]

    layer1 = tf.cond(tf.reduce_sum(tf.stack(diff1)) > 0, lambda: _resize(layer1, diff1, shape2, shape1), lambda: layer1)
    layer2 = tf.cond(tf.reduce_sum(tf.stack(diff2)) > 0, lambda: _resize(layer2, diff2, shape1, shape2), lambda: layer2)
    # if sum(diff1) > 0:
    #     # crop_left = shape_diff // 2
    #     layer1 = resize(layer1, diff1, shape2)
    # if sum(diff2) > 0:
    #     layer2 = resize(layer2, diff2, shape1)
    return layer1, layer2


def crop_valid_after_upsampling(kernel: List[int], stride: List[int], layer_orig: AnyTensor, layer_up: AnyTensor,
                                layer_down: AnyTensor) -> (AnyTensor, AnyTensor):
    """
    crops layers layer_orig and layer_up in spartial dimension to only contain valid pixels
    @param kernel: kernes-size of upsampling-layer
    @param stride: stride of up/sub sampling
    @param layer_orig: layer on which downsampling was applied
    @param layer_up: layer after upsampling was applied
    @param layer_down: layer before upsampling was applied
    @return: layer_orig and layer_up with cropped borders so that the fit to each other
    """
    shape_orig = to_shape(layer_orig)
    shape_down = to_shape(layer_down)
    shape_up = to_shape(layer_up)
    kernel = np.asarray(kernel)
    stride = np.asarray(stride)
    assert len(shape_orig) == len(shape_down)
    assert len(shape_orig) == len(shape_up)
    assert len(shape_orig) == len(kernel) + 2  # +1 batchsize +1 featuresize
    assert len(shape_orig) == len(stride) + 2  # +1 batchsize +1 featuresize
    shape, offset_orig, offset_up = _shape_for_up(kernel, stride, shape_orig[1:-1], shape_up[1:-1], shape_down[1:-1])

    layer_orig = crop(layer_orig, offset_orig, shape)
    layer_up = crop(layer_up, offset_up, shape)
    return layer_orig, layer_up


def crop(tensor: AnyTensor, offset: List[AnyTensor], targetshape: List[AnyTensor]) -> AnyTensor:
    """
    crops the tensor with the given dimension.
    Note that the first and the last dimension says fixed,
    since the cropping should be applied on the spartial dimensions between them
    @param tensor: a tensor of data_format NWC, NHWC or NHWDC
    @param offset: offset of start position, should have shape 1 for NWC, 2 for NHWC and 3 for NHWDC
    @param targetshape: size of desired tensor, shoud have shape 1 for NWC, 2 for NHWC and 3 for NHWDC
    @return: part of @tensor with shape @targetshape
    """
    tensor_shape = to_shape(tensor)
    return array_ops.slice(
        tensor,
        array_ops.stack([0, *offset, 0]),
        array_ops.stack([tensor_shape[0], *targetshape, tensor_shape[-1]])
    )


def _shape_for_up(kernel, stride, shape_orig, shape_up, shape_down):
    #only static shapes!!!
    # magic formuala see https://datascience.stackexchange.com/questions/26451/how-to-calculate-the-output-shape-of-conv2d-transpose for 'full'
    shape = shape_down * stride - (stride + kernel - 2)
    shape = np.minimum(shape, shape_orig)
    for i, dim in enumerate(shape):
        if dim < 1:
            raise Exception(
                f"target shape '{shape}' at dimension '{i}' has value '{dim}' calculated for stride = {stride} kernel = {kernel} and shape_down = {shape_down}")
    # shape = shape_orig - (kernel - 1) * stride
    offset_up = (shape_up - shape) // 2
    offset_orig = (shape_orig - shape) // 2
    return shape, offset_orig, offset_up


class Slicer(object):

    def __init__(self, size_in: List[int], rec_field: List[int], with_batch: bool = True):
        """
        object that can either return a tensorflow-operation to slice a tensor
        for each dimension the following holds
        |<--------------------size_in---------------------->|
        |<---cut_left--->|<---size_out--->|<---cut_right--->|
        0                 start            end               size_in
        cut_left + cut_right = rec_field
        cut_left<=cut_right<=cut_left+1

        @param size:
        @param rec_field: should be at least 1 in each dimension and at most as large as size_in
        @param with_batch:
        """
        # maybe +1 not +0 if rec_file = [1,1,1], left = [0,0,0], right = params.size

        cut_left = start = [(b - 1) // 2 for b in rec_field]
        cut_right = [r - s - 1 for r, s in zip(rec_field, start)]
        size_out = [s - r + 1 for s, r in zip(size_in, rec_field)]
        end = [si + st for si, st in zip(size_out, start)]
        self._start, self._start_class = self.combine(cut_left, 0, 0 if with_batch else None)
        self._size_out, self._size_out_class = self.combine(size_out, -1, -1 if with_batch else None)
        self._end, self._end_class = self.combine(end, -1, -1 if with_batch else None)

    @staticmethod
    def combine(vec: List[int], last: int, batch: Union[None, int]):
        res = []
        if batch is not None:
            res.append(batch)
        res.extend(vec)
        res_class = res.copy()
        res_class.append(last)
        return res, res_class

    def get_start(self, add_class_dimension: bool = False) -> List[int]:
        return self._start_class if add_class_dimension else self._start

    def get_size_out(self, add_class_dimension: bool = False) -> List[int]:
        return self._size_out_class if add_class_dimension else self._size_out

    def get_end(self, add_class_dimension: bool = False) -> List[int]:
        return self._end_class if add_class_dimension else self._end

    def slice_as_tf(self, tensor, add_class_dimension: bool = False):
        if add_class_dimension:
            return tf.slice(tensor, self._start_class, self._size_out_class)
        return tf.slice(tensor, self._start, self._size_out)

    def shift(self, coord: np.ndarray, add_class_dimension: bool = False):
        # [DIMENSION][#START,#END(excl)]
        # shape = [#DIMENSIONS][2]
        coord = coord.copy().transpose()
        start = coord[0, :] + np.array(self.get_start(add_class_dimension))
        end = coord[0, :] + np.array(self.get_end(add_class_dimension))
        # end2 = start[0, :] + np.array(self.get_size_out(add_class_dimension))
        return np.transpose(np.reshape(np.concatenate([start, end], axis=0), [2, -1]))

    # def get_start_end(self, start, add_class_dimension=False):
    #     off_start = self.get_start(add_class_dimension)
    #     # off_length = self.get_length(add_class_dimension)
    #     # s = start + off_start
    #     e = end + off_start - off_length
    #     return (s, e)

    def slice_as_np(self, tensor, add_class_dimension: bool = False):
        start = self.get_start(False)
        end = self.get_end(False)
        if add_class_dimension:
            assert len(tensor.shape) == len(end) + 1
            if len(start) == 0:
                return tensor
            if len(start) == 1:
                return tensor[
                       start[0]:end[0],
                       :]
            if len(start) == 2:
                return tensor[
                       start[0]:end[0],
                       start[1]:end[1],
                       :]
            if len(start) == 3:
                return tensor[
                       start[0]:end[0],
                       start[1]:end[1],
                       start[2]:end[2],
                       :]
            if len(start) == 4:
                return tensor[
                       start[0]:end[0],
                       start[1]:end[1],
                       start[2]:end[2],
                       start[3]:end[3],
                       :]

        assert len(tensor.shape) == len(end)
        if len(start) == 1:
            return tensor[
                   start[0]:end[0]]
        if len(start) == 2:
            return tensor[
                   start[0]:end[0],
                   start[1]:end[1]]
        if len(start) == 3:
            return tensor[
                   start[0]:end[0],
                   start[1]:end[1],
                   start[2]:end[2]]
        if len(start) == 4:
            return tensor[
                   start[0]:end[0],
                   start[1]:end[1],
                   start[2]:end[2],
                   start[3]:end[3]]
        if len(start) == 5:
            return tensor[
                   start[0]:end[0],
                   start[1]:end[1],
                   start[2]:end[2],
                   start[3]:end[3],
                   start[4]:end[4]]
        raise Exception(f"for length {len(start)} not implemented.")
