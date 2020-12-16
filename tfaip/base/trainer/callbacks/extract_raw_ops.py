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

__author__ = "Gundram Leifert"
__copyright__ = "Copyright 2020, Planet AI GmbH"
__credits__ = ["Gundram Leifert","Christoph Wick"]
__email__ = "gundram.leifert@planet-ai.de"
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
import tensorflow.keras.callbacks as cb


class ExtractRawOps(cb.Callback):
    def __init__(self):
        super(ExtractRawOps, self).__init__()
        self._supports_tf_logs = True
        self.extracted_logs = {}

    def on_train_begin(self, logs=None):
        self.extracted_logs = {}

    def on_epoch_begin(self, epoch, logs=None):
        self.extracted_logs = {}

    def on_epoch_end(self, epoch, logs=None):
        self.fix(logs)

    def on_train_batch_end(self, batch, logs=None):
        self.fix(logs)

    def on_predict_batch_end(self, batch, logs=None):
        self.fix(logs)

    def on_test_batch_end(self, batch, logs=None):
        self.fix(logs, prefix="val_")

    def fix(self, logs, prefix=''):
        if logs is None:
            return
        for k in list(logs.keys()):
            if isinstance(logs[k], tf.Tensor) and logs[k].dtype == tf.string:
                self.extracted_logs[prefix + k] = logs[k].numpy()
                del logs[k]
