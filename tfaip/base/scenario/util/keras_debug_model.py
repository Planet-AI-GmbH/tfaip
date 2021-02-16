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

from tfaip.base.model import ModelBase


class KerasDebugModel(keras.Model):
    def get_config(self):
        raise NotImplementedError

    def __init__(self, model: 'ModelBase'):
        super(KerasDebugModel, self).__init__()
        self.model = model

    def call(self, inputs, training=None, mask=None):
        outputs = self.model.build(inputs)
        extended_outputs = {**outputs, **self.model.additional_outputs(inputs, outputs)}
        losses = self.model.loss(inputs, extended_outputs)
        metrics = self.model.extended_metric(inputs, extended_outputs)
        return {**outputs, **losses, **metrics}
