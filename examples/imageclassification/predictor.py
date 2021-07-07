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
from tfaip import Sample
from tfaip.predict.predictor import Predictor

from examples.imageclassification.params import Keys


class ICPredictor(Predictor):
    def _print_prediction(self, sample: Sample, print_fn):
        class_name = self.data.params.classes[sample.outputs[Keys.OutputClass]]
        conf = sample.outputs[Keys.OutputSoftmax][sample.outputs[Keys.OutputClass]]
        print_fn(f"This image most likely belongs to {class_name} with a {conf:.2f} percent confidence.")
