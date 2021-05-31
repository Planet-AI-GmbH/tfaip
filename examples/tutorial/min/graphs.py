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

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten

from examples.tutorial.min.model import TutorialModelParams
from tfaip.imports import GraphBase


class TutorialGraph(GraphBase[TutorialModelParams]):
    def __init__(self, params: TutorialModelParams, name="conv", **kwargs):
        super(TutorialGraph, self).__init__(params, name=name, **kwargs)
        # Create all layers
        self.conv1 = Conv2D(kernel_size=(2, 2), filters=16, padding="same", name="conv1")
        self.pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="pool1")
        self.conv2 = Conv2D(kernel_size=(2, 2), filters=32, padding="same", name="conv2")
        self.pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="pool2")
        self.flatten = Flatten()
        self.ff = Dense(128, name="f_ff", activation="relu")
        self.logits = Dense(self._params.n_classes, activation=None, name="classify")

    def build_graph(self, inputs, training=None):
        # Connect all layers and return a dict of the outputs
        rescaled_img = K.cast(inputs["img"], dtype="float32") / 255
        if len(rescaled_img.shape) == 3:
            rescaled_img = K.expand_dims(rescaled_img, axis=-1)  # add missing channels dimension
        conv_out = self.pool2(self.conv2(self.pool1(self.conv1(rescaled_img))))
        logits = self.logits(self.ff(self.flatten(conv_out)))
        pred = K.softmax(logits, axis=-1)
        cls = K.argmax(pred, axis=-1)
        out = {"pred": pred, "logits": logits, "class": cls}
        return out
