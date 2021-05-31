import glob
import os
from dataclasses import dataclass, field
from typing import Iterable, Type, List

from paiargparse import pai_dataclass, pai_meta
from tfaip import DataGeneratorParams, Sample
from tfaip.data.pipeline.datagenerator import DataGenerator


@pai_dataclass
@dataclass
class ATRDataGeneratorParams(DataGeneratorParams):
    @staticmethod
    def cls() -> Type["DataGenerator"]:
        return ATRDataGenerator

    image_files: List[str] = field(default_factory=list, metadata=pai_meta(required=True))

    def __post_init__(self):
        if len(self.image_files) == 1:
            self.image_files = glob.glob(self.image_files[0])


def split_all_ext(path):
    path, basename = os.path.split(path)
    pos = basename.find(".")
    return os.path.join(path, basename[:pos]), basename[pos:]


class ATRDataGenerator(DataGenerator[ATRDataGeneratorParams]):
    def __len__(self):
        return len(self.params.image_files)

    def generate(self) -> Iterable[Sample]:
        return (
            Sample(inputs=fn, targets=split_all_ext(fn)[0] + ".gt.txt", meta={"filename": fn})
            for fn in self.params.image_files
        )
