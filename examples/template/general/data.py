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
import logging

from examples.template.general.datapipeline.firstdataprocessor import FirstDataProcessorParams
from examples.template.general.datapipeline.seconddataprocessor import SecondDataProcessorParams
from examples.template.general.params import TemplateDataParams
from tfaip.data.data import DataBase

logger = logging.getLogger(__name__)


class TemplateData(DataBase[TemplateDataParams]):
    @classmethod
    def default_params(cls) -> TemplateDataParams:
        p = super().default_params()
        # Setup the data pipeline
        p.pre_proc.processors = [
            FirstDataProcessorParams(),
            SecondDataProcessorParams(),
            # ...
        ]
        return p

    def _input_layer_specs(self):
        raise NotImplementedError

    def _target_layer_specs(self):
        raise NotImplementedError
