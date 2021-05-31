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
