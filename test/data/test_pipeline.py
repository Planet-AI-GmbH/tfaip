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
import unittest
from dataclasses import dataclass, field
from typing import Iterable, Dict, Type, List

import tensorflow as tf
import numpy as np
from paiargparse import pai_dataclass

from tfaip import DataGeneratorParams, DataBaseParams
from tfaip.data.data import DataBase
from tfaip.data.databaseparams import DataPipelineParams
from tfaip import Sample, PipelineMode
from tfaip.data.pipeline.datagenerator import DataGenerator
from tfaip.data.pipeline.processor.dataprocessor import (
    DataProcessorParams,
    MappingDataProcessor,
    DataProcessorBase,
    GeneratingDataProcessor,
)
from tfaip.data.pipeline.processor.params import SequentialProcessorPipelineParams, ComposedProcessorPipelineParams


@pai_dataclass
@dataclass
class PrepareParams(DataProcessorParams):
    @staticmethod
    def cls() -> Type["DataProcessorBase"]:
        class Prepare(MappingDataProcessor):
            def apply(self, sample: Sample) -> Sample:
                return sample.new_inputs({"n": np.asarray([sample.inputs])}).new_targets(
                    {"n": np.asarray([sample.targets])}
                )

        return Prepare


@pai_dataclass
@dataclass
class AddProcessorParams(DataProcessorParams):
    v: int = 1

    @staticmethod
    def cls() -> Type["DataProcessorBase"]:
        return AddProcessor


class AddProcessor(MappingDataProcessor[AddProcessorParams]):
    def apply(self, sample: Sample) -> Sample:
        return sample.new_inputs(sample.inputs + self.params.v)


@pai_dataclass
@dataclass
class RepeatSampleProcessorParams(DataProcessorParams):
    f: int = 1
    add_per_step: int = 7

    @staticmethod
    def cls() -> Type["DataProcessorBase"]:
        return RepeatSampleProcessor


class RepeatSampleProcessor(GeneratingDataProcessor[RepeatSampleProcessorParams]):
    def generate(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        for s in samples:
            for i in range(self.params.f):
                yield s.new_inputs(s.inputs + i * self.params.add_per_step)


@pai_dataclass
@dataclass
class DropIfEvenProcessorParams(DataProcessorParams):
    @staticmethod
    def cls():
        return DropIfEven


class DropIfEven(GeneratingDataProcessor[DropIfEvenProcessorParams]):
    def generate(self, samples: Iterable[Sample]) -> Iterable[Sample]:
        for s in samples:
            if s.inputs % 2 == 0:
                continue
            yield s


@pai_dataclass
@dataclass
class MultiplyProcessorParams(DataProcessorParams):
    f: int = 1

    @staticmethod
    def cls() -> Type["DataProcessorBase"]:
        return MultiplyProcessor


class MultiplyProcessor(MappingDataProcessor[MultiplyProcessorParams]):
    def apply(self, sample: Sample) -> Sample:
        return sample.new_inputs(sample.inputs * self.params.f)


@dataclass
class SimpleDataGeneratorParams(DataGeneratorParams):
    numbers_to_generate: List[int] = field(default_factory=list)

    @staticmethod
    def cls() -> Type["DataGenerator"]:
        return SimpleDataGenerator


class SimpleDataGenerator(DataGenerator[SimpleDataGeneratorParams]):
    def __len__(self):
        return len(self.params.numbers_to_generate)

    def generate(self) -> Iterable[Sample]:
        return map(lambda s: Sample(inputs=s, targets=s), self.params.numbers_to_generate)


@pai_dataclass
@dataclass
class DataParams(DataBaseParams):
    @staticmethod
    def cls() -> Type["DataBase"]:
        return Data


class Data(DataBase):
    def _input_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        return {"n": tf.TensorSpec([1], dtype="int32", name="n")}

    def _target_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        return {}


class TestDataPipeline(unittest.TestCase):
    def test_data_sequential_pipeline(self):
        numbers = list(range(10))
        target_numbers = []
        for n in numbers:
            target_numbers.append((n + 1 + 3))
            target_numbers.append((n + 1 + 3 + 7))
        target_numbers = [n * 3 + 1 for n in target_numbers]
        target_numbers = [n for n in target_numbers if n % 2 == 1]

        data_params = DataParams(
            pre_proc=SequentialProcessorPipelineParams(
                run_parallel=False,
                num_threads=3,
                processors=[
                    AddProcessorParams(v=1),
                    AddProcessorParams(v=3),
                    RepeatSampleProcessorParams(f=2, add_per_step=7),
                    MultiplyProcessorParams(f=3),
                    AddProcessorParams(v=1),
                    DropIfEvenProcessorParams(),
                ],
            )
        )
        data = data_params.create()
        with data.create_pipeline(
            DataPipelineParams(mode=PipelineMode.TRAINING), SimpleDataGeneratorParams(numbers_to_generate=numbers)
        ) as pipeline:
            out = [s.inputs for s in pipeline.generate_input_samples(auto_repeat=False)]
            self.assertListEqual(target_numbers, out)

    def test_data_composed_pipeline(self):
        numbers = list(range(10))
        target_numbers = []
        for n in numbers:
            target_numbers.append((n + 1 + 3))
            target_numbers.append((n + 1 + 3 + 7))
        target_numbers = [n * 2 + 1 for n in target_numbers]
        target_numbers = [n for n in target_numbers if n % 2 == 1]

        data_params = DataParams(
            pre_proc=ComposedProcessorPipelineParams(
                pipelines=[
                    SequentialProcessorPipelineParams(
                        run_parallel=False,
                        num_threads=3,
                        processors=[
                            AddProcessorParams(v=1),
                            AddProcessorParams(v=3),
                        ],
                    ),
                    SequentialProcessorPipelineParams(
                        num_threads=1,  # Deterministic
                        processors=[
                            RepeatSampleProcessorParams(f=2, add_per_step=7),
                        ],
                    ),
                    SequentialProcessorPipelineParams(
                        run_parallel=True,
                        processors=[
                            MultiplyProcessorParams(f=2),
                            AddProcessorParams(v=1),
                        ],
                    ),
                    SequentialProcessorPipelineParams(
                        num_threads=1,  # Deterministic
                        processors=[
                            DropIfEvenProcessorParams(),
                        ],
                    ),
                ]
            )
        )
        data = data_params.create()
        with data.create_pipeline(
            DataPipelineParams(mode=PipelineMode.TRAINING), SimpleDataGeneratorParams(numbers_to_generate=numbers)
        ) as pipeline:
            out = [s.inputs for s in pipeline.generate_input_samples(auto_repeat=False)]
            self.assertListEqual(target_numbers, out)
