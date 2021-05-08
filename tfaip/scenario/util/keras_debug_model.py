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
"""Definition of the KerasDebugModel"""
import tensorflow.keras as keras

from tfaip.model.modelbase import ModelBase


class KerasDebugModel(keras.Model):
    """Keras model that creates the actual model in is call to allow eager execution during graph construction

    The outputs of this model are the outputs of the actual `ModelBase`, its extended losses and metrics.
    """
    def get_config(self):
        raise NotImplementedError

    def __init__(self, model: ModelBase):
        super().__init__()
        self.model = model
        model.setup()

    def call(self, inputs_targets, training=None, mask=None):
        outputs = self.model.build(inputs_targets)
        extended_outputs = {**outputs, **self.model.additional_outputs(inputs_targets, outputs)}
        losses = {name: loss.loss(inputs_targets[loss.target], extended_outputs[loss.output]) for name, loss in
                  self.model.loss().items()}
        extended_losses = self.model.extended_loss(inputs_targets, extended_outputs)
        metrics = self.model.extended_metric(inputs_targets, extended_outputs)
        return {**outputs, **extended_losses, **losses, **metrics}
