# Copyright 2021 The tfaip authors. All Rights Reserved.
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
from dataclasses import dataclass, field
from typing import List

import numpy as np
import tensorflow as tf
from paiargparse import pai_dataclass, pai_meta
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten

from examples.tutorial.full.graphs.backend import TutorialBackendParams, TutorialBackend
from tfaip.model.tensorboardwriter import TensorboardWriter


@pai_dataclass(alt="CNN")
@dataclass
class ConvGraphParams(TutorialBackendParams):
    filters: List[int] = field(
        default_factory=lambda: [16, 32],
        metadata=pai_meta(
            help="List of filters to set up the conv layers. Each conv layer is followed by a max pooling layer."
        ),
    )
    dense: List[int] = field(
        default_factory=lambda: [128], metadata=pai_meta(help="Definition of the hidden dense layers after the conv")
    )
    activation: str = field(default="relu", metadata=pai_meta(help="Activation function of the hidden layers"))

    def cls(self):
        return ConvLayers(self)


def handle(name: str, value: np.ndarray, step: int):
    # Create the image data as numpy array
    b, w, h, c = value.shape
    ax_dims = int(np.ceil(np.sqrt(c)))
    out_conv_v = np.zeros([b, w * ax_dims, h * ax_dims, 1])
    for i in range(c):
        x = i % ax_dims
        y = i // ax_dims
        out_conv_v[:, x * w : (x + 1) * w, y * h : (y + 1) * h, 0] = value[:, :, :, i]

    # Write the image (use 'name_for_tb' and step)
    tf.summary.image(name, out_conv_v, step=step)


class ConvLayers(TutorialBackend[ConvGraphParams]):
    def __init__(self, params: ConvGraphParams, name="conv", **kwargs):
        super(ConvLayers, self).__init__(params, name=name, **kwargs)
        self.conv_layers = [
            Conv2D(filters=filters, kernel_size=(2, 2), strides=(1, 1), padding="same", activation="relu")
            for filters in params.filters
        ]
        self.pool_layers = [MaxPool2D(pool_size=(2, 2), strides=(2, 2)) for _ in params.filters]
        self.flatten = Flatten()
        self.dense_layers = [Dense(nodes, activation="relu") for nodes in params.dense]
        self.conv_mat_tb_writer = TensorboardWriter(func=handle, dtype="float32", name="conv_mat")

    def call(self, images, **kwargs):
        conv_out = images
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            conv_out = pool(conv(conv_out))
        dense_out = self.flatten(conv_out)
        for dense in self.dense_layers:
            dense_out = dense(dense_out)

        self.add_tensorboard(self.conv_mat_tb_writer, conv_out)
        return {
            "out": dense_out,
            "conv_out": conv_out,
        }
