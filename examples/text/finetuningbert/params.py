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
import os
from dataclasses import dataclass
from typing import Type

from paiargparse import pai_dataclass
from tfaip import DataBaseParams

this_dir = os.path.dirname(os.path.realpath(__file__))


class Keys:
    InputSentence1 = "sentence1"
    InputSentence2 = "sentence2"
    InputWordIds = "input_ids"
    InputMask = "attention_mask"
    InputTypeIds = "token_type_ids"
    Target = "label"
    OutputLogits = "logits"
    OutputSoftmax = "softmax"
    OutputClass = "class"


@pai_dataclass
@dataclass
class FTBertDataParams(DataBaseParams):
    @staticmethod
    def cls():
        from examples.text.finetuningbert.data import FTBertData

        return FTBertData
