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
from typing import Iterable, Dict, Type, List, Union, Any

import numpy
import numpy as np
import tensorflow as tf
from paiargparse import pai_dataclass
from test.data.test_pipeline import AddProcessorParams, MultiplyProcessorParams, PrepareParams

from tfaip import DataGeneratorParams, DataBaseParams
from tfaip.data.data import DataBase
from tfaip.data.databaseparams import DataPipelineParams
from tfaip import Sample, PipelineMode
from tfaip.data.pipeline.datagenerator import DataGenerator
from tfaip.data.pipeline.processor.params import SequentialProcessorPipelineParams


def groups_into_samples(inputs: List[Any]) -> List[List[Any]]:
    # create batches
    # from [0, 1, 2, 3, 4, 5, ...] create
    # [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9], ...]
    inputs = inputs[:]
    batches = []
    target_size = 1
    while inputs:
        n = inputs.pop()
        sample = n
        if len(batches) == 0:
            batches.append([sample])
            target_size += 1
        else:
            if len(batches[-1]) < target_size:
                batches[-1].append(sample)
            else:
                batches.append([sample])
                target_size += 1

    return batches


@dataclass
class BatchedDataGeneratorParams(DataGeneratorParams):
    numbers_to_generate: List[int] = field(default_factory=list)

    @staticmethod
    def cls() -> Type["DataGenerator"]:
        return BatchedDataGenerator


class BatchedDataGenerator(DataGenerator[BatchedDataGeneratorParams]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batches = groups_into_samples(self.params.numbers_to_generate)
        self.batches = [[Sample(inputs=x, targets=x, meta={}) for x in b] for b in self.batches]

    def __len__(self):
        return len(self.batches)

    def generate(self) -> Iterable[Union[Sample, List[Sample]]]:
        return self.batches

    def yields_batches(self) -> bool:
        return True


@dataclass
class BatchedPadDataGeneratorParams(DataGeneratorParams):
    numbers_to_generate: List[int] = field(default_factory=list)

    @staticmethod
    def cls() -> Type["DataGenerator"]:
        return BatchedPadDataGenerator


class BatchedPadDataGenerator(DataGenerator[BatchedDataGeneratorParams]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batches = groups_into_samples(self.params.numbers_to_generate)
        self.batches = [Sample(inputs={"n": np.asarray(x)}, targets={"n": np.asarray(x)}) for x in self.batches]
        self.batches = groups_into_samples(self.batches)

    def __len__(self):
        return len(self.batches)

    def generate(self) -> Iterable[Union[Sample, List[Sample]]]:
        return self.batches

    def yields_batches(self) -> bool:
        return True


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
        return {"n": tf.TensorSpec([1], dtype="int32", name="n")}


@pai_dataclass
@dataclass
class DataPadParams(DataBaseParams):
    @staticmethod
    def cls() -> Type["DataBase"]:
        return DataPad


class DataPad(DataBase):
    def _input_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        return {"n": tf.TensorSpec([None], dtype="int32", name="n")}

    def _target_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        return {"n": tf.TensorSpec([None], dtype="int32", name="n")}


def run_pad_test(test, n_numbers=1000):
    numbers = list(range(n_numbers))
    data_params = DataPadParams()
    data = data_params.create()

    def to_tuple(s):
        return s.inputs, s.targets, s.meta

    with data.create_pipeline(
        DataPipelineParams(mode=PipelineMode.TRAINING), BatchedPadDataGeneratorParams(numbers_to_generate=numbers)
    ) as pipeline:
        # Test generate input samples
        batched_samples = pipeline.generate_input_samples(auto_repeat=False)

        # Test dataset
        batched_samples_with_ds = list(pipeline.input_dataset(auto_repeat=False).as_numpy_iterator())
        for (i1, t1, m1), (i2, t2, m2) in zip(map(to_tuple, batched_samples), batched_samples_with_ds):
            np.testing.assert_array_equal(i1["n"], i2["n"])
            np.testing.assert_array_equal(t1["n"], t2["n"])


def run_test(test, parallel, n_numbers=100):
    numbers = list(range(n_numbers))
    target_numbers = []
    for n in numbers:
        target_numbers.append((n + 1 + 3))
    target_numbers = [n * 3 + 1 for n in target_numbers]
    target_numbers = groups_into_samples(target_numbers)

    data_params = DataParams(
        pre_proc=SequentialProcessorPipelineParams(
            run_parallel=parallel,
            num_threads=3,
            processors=[
                AddProcessorParams(v=1),
                AddProcessorParams(v=3),
                MultiplyProcessorParams(f=3),
                AddProcessorParams(v=1),
                PrepareParams(),
            ],
        )
    )
    data = data_params.create()
    with data.create_pipeline(
        DataPipelineParams(mode=PipelineMode.TRAINING), BatchedDataGeneratorParams(numbers_to_generate=numbers)
    ) as pipeline:
        # Test generate input samples
        batched_samples = list(pipeline.generate_input_samples(auto_repeat=False))
        out = [list(np.squeeze(x.inputs["n"], axis=-1)) for x in batched_samples]
        test.assertListEqual(target_numbers, out)

        # Test dataset
        batched_samples_ds = list(pipeline.input_dataset(auto_repeat=False).as_numpy_iterator())
        out = [list(np.squeeze(i["n"], axis=-1)) for i, t, m in batched_samples_ds]
        test.assertListEqual(target_numbers, out)

        for s in batched_samples_ds:
            s[2]["meta"] = np.array([[d[0].decode("utf-8")] for d in s[2]["meta"]])

        def check_equal(s1: dict, s2: dict):
            test.assertListEqual(list(s1.keys()), list(s2.keys()))
            for k in s1.keys():
                numpy.testing.assert_array_equal(s1[k], s2[k])

        for s1, s2 in zip(batched_samples, batched_samples_ds):
            check_equal(s1.inputs, s2[0])
            check_equal(s1.targets, s2[1])
            check_equal(s1.meta, s2[2])


class TestManualBatching(unittest.TestCase):
    def test_sequential(self):
        run_test(self, False)

    def test_parallel(self):
        run_test(self, True)

    def test_padding(self):
        run_pad_test(self)
