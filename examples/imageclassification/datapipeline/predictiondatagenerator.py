from dataclasses import dataclass, field
from typing import List, Type, Iterable

from paiargparse import pai_dataclass, pai_meta

from tfaip import DataGeneratorParams, Sample
from tfaip.data.pipeline.datagenerator import DataGenerator


@pai_dataclass
@dataclass
class ICPredictionDataGeneratorParams(DataGeneratorParams):
    @staticmethod
    def cls() -> Type["DataGenerator"]:
        return ICPredictionDataGenerator

    image_files: List[str] = field(default_factory=list, metadata=pai_meta(required=True))


# Custom data generator for prediction since the classes are unknown
class ICPredictionDataGenerator(DataGenerator[ICPredictionDataGeneratorParams]):
    def __len__(self):
        return len(self.params.image_files)

    def generate(self) -> Iterable[Sample]:
        return (Sample(inputs=fn) for fn in self.params.image_files)
