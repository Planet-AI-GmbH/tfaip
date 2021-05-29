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
from abc import ABC

import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tfaip.model.graphbase import GraphBase


class TutorialGraph(GraphBase, ABC):
    def __init__(self, params, **kwargs):
        super(TutorialGraph, self).__init__(params, **kwargs)
        self.acc = keras.metrics.Accuracy(name="internal_acc")
        self.backend = params.graph.cls()
        self.logits = keras.layers.Dense(self._params.n_classes, activation=None, name="classify")

    def build_graph(self, inputs, training=None):
        # Optionally add a training attribute to check if the graph is in training or validation mode
        # To design a different behaviour of the prediction graph, check if GT is available in the inputs
        # call function that is shared by all other graphs
        rescaled_img = K.expand_dims(K.cast(inputs["img"], dtype="float32") / 255, -1)
        backbone_out = self.backend(rescaled_img)  # call the actual graph (MLP or CNN)
        logits = self.logits(backbone_out["out"])
        pred = K.softmax(logits, axis=-1)
        cls = K.argmax(pred, axis=-1)
        out = {"pred": pred, "logits": logits, "class": cls}

        # Add conv out to outputs to show how to visualize using tensorboard
        if "conv_out" in backbone_out:
            out["conv_out"] = backbone_out["conv_out"]

        # Add a metric within the graph in the training model.
        # This metric will however not be used in LAV
        if "gt" in inputs:
            self.add_metric(self.acc(out["class"], inputs["gt"]))
        return out
