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
from dataclasses import dataclass
from typing import Optional, Type, Iterable

import tensorflow_datasets as tfds
from paiargparse import pai_dataclass
from tfaip import TrainerPipelineParamsBase, DataGeneratorParams, Sample, PipelineMode
from tfaip.data.pipeline.datagenerator import DataGenerator
from tfaip.trainer.params import TDataGeneratorVal, TDataGeneratorTrain

from examples.text.finetuningbert.params import Keys


@pai_dataclass
@dataclass
class GlueGeneratorParams(DataGeneratorParams):
    split: str = ""

    @staticmethod
    def cls() -> Type["DataGenerator"]:
        return GlueDataGenerator


class GlueDataGenerator(DataGenerator[GlueGeneratorParams]):
    def __init__(self, *args, **kwargs):
        super(GlueDataGenerator, self).__init__(*args, **kwargs)
        assert len(self.params.split) > 0
        self.glue = tfds.load("glue/mrpc", split=self.params.split, batch_size=None)

    def __len__(self):
        return len(self.glue)

    def generate(self) -> Iterable[Sample]:
        def to_sample(d: dict) -> Sample:
            return Sample(
                inputs={
                    Keys.InputSentence1: d["sentence1"].decode("utf-8"),
                    Keys.InputSentence2: d["sentence2"].decode("utf-8"),
                },
                targets={Keys.Target: [d["label"]]},
                meta={"index": int(d["idx"])},
            )

        return map(to_sample, self.glue.as_numpy_iterator())


@pai_dataclass
@dataclass
class GlueTrainerPipelineParams(TrainerPipelineParamsBase[GlueGeneratorParams, GlueGeneratorParams]):
    def train_gen(self) -> TDataGeneratorTrain:
        return GlueGeneratorParams(split="train")

    def val_gen(self) -> Optional[TDataGeneratorVal]:
        return GlueGeneratorParams(split="validation")


if __name__ == "__main__":
    # check if data can be generated correctly
    gen = GlueGeneratorParams(split="train").create(PipelineMode.Training)
    print(len(gen))
    for p in gen.generate():
        print(p)
