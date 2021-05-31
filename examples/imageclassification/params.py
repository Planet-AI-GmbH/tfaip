import os
from dataclasses import dataclass, field
from typing import List, Type

from paiargparse import pai_dataclass
from tfaip import DataBaseParams

this_dir = os.path.dirname(os.path.realpath(__file__))


class Keys:
    Target = "target"
    Image = "image"
    OutputLogits = "logits"
    OutputSoftmax = "softmax"
    OutputClass = "class"
    OutputClassName = "class_name"


@pai_dataclass
@dataclass
class ICDataParams(DataBaseParams):
    @staticmethod
    def cls():
        from examples.imageclassification.data import ICData

        return ICData

    classes: List[str] = field(default_factory=list)
    image_height: int = 180
    image_width: int = 180
