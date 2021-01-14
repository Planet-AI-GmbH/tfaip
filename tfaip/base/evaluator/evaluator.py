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
from typing import Dict

from tfaip.base.data.pipeline.definitions import Sample
from tfaip.base.evaluator.params import EvaluatorParams
from tfaip.util.typing import AnyNumpy


class Evaluator:
    @classmethod
    def default_params(cls) -> EvaluatorParams:
        return EvaluatorParams()

    def __init__(self, params: EvaluatorParams):
        self.params = params

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def update_state(self, sample: Sample):
        pass

    def result(self) -> Dict[str, AnyNumpy]:
        return {}
