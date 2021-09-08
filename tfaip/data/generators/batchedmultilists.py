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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from random import shuffle
from typing import TypeVar, Dict, Union, List, Tuple, Optional, Iterable

import numpy as np
from paiargparse import pai_dataclass

from tfaip import DataGeneratorParams, Sample, PipelineMode
from tfaip.data.pipeline.datagenerator import DataGenerator

logger = logging.getLogger(__name__)


@pai_dataclass
@dataclass
class BatchedMultiListDataGeneratorParams(DataGeneratorParams, ABC):
    shape_files: Optional[List[str]] = None
    batch_bins: int = 140000000


T = TypeVar("T", bound=BatchedMultiListDataGeneratorParams)


class BatchedMultiListDataGenerator(DataGenerator[T]):
    """Generate already batched samples (optional) based on shape files that define the size of each sample

    Use cases:
      - In speech recognition or ATR to have batches of equal (similar) length or number of tokens which allows
        to make optimal use of the space

    Based on:
      - Based on the Files of ESPNet

    Example (Speech Scenario, see ESPNet):
        - 4 files: wav, text, wav_shape, text_shape
        - wav a list of "id path_to_wav_file"
        - text a list of "id GROUND_TRUTH"
        - wav_shape a list of "id len", where len is the precomputed length of the wav file
        - text_shape a list of "id len", where len is the length of the text
        - specify the batch_bins to set the number of tokens/wav per batch
    """

    @abstractmethod
    def lists(self) -> Dict[str, Path]:
        raise NotImplementedError

    @abstractmethod
    def list_types(self) -> Dict[str, str]:
        raise NotImplementedError

    def make_sample(self, file_id: str):
        sample = Sample(
            inputs={k: self.parsed_files[k][file_id] for k in self._input_keys},
            targets={k: self.parsed_files[k][file_id] for k in self._target_keys},
            meta={"id": file_id, **{k + "_filename": v[file_id] for k, v in self.parsed_files.items()}},
        )
        if len(sample.inputs) == 1:
            sample = sample.new_inputs(list(sample.inputs.values())[0])
        if len(sample.targets) == 1:
            sample = sample.new_targets(list(sample.targets.values())[0])

        return sample

    def load_sample(self, sample: Sample) -> Sample:
        return sample

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert all(
            v in {"input", "target"} for v in self.list_types().values()
        ), "list_types must be 'input' or 'target'"
        self._input_keys = [k for k, v in self.list_types().items() if v == "input"]
        self._target_keys = [k for k, v in self.list_types().items() if v == "target"]

        self.parsed_files: Dict[str, Dict[str, str]] = {k: parse_file(v) for k, v in self.lists().items()}
        assert len(self.parsed_files) > 0, "No lists provided."
        self.first_parsed_file = next(iter(self.parsed_files.values()))
        self.sample_ids = list(self.first_parsed_file.keys())

        assert all(
            len(self.sample_ids) == len(v) for v in self.parsed_files.values()
        ), "Lists with different number of samples"

        self.samples = {k: self.make_sample(k) for k in self.sample_ids}
        if self.yields_batches():
            sampler = NumElementsBatchSampler(
                batch_bins=self.params.batch_bins,
                shape_files=self.params.shape_files,
            )
            self.batched_samples = sampler.batch_list
        else:
            self.batched_samples = self.samples

    def yields_batches(self) -> bool:
        return self.params.shape_files is not None

    def __len__(self):
        return len(self.samples)

    def steps_per_epoch(self, batch_size: int, scale_factor: float = 1):
        if self.yields_batches():
            return int(scale_factor * len(self.batched_samples))
        else:
            return super().steps_per_epoch(batch_size, scale_factor)

    def generate(self) -> Iterable[Sample]:
        if self.yields_batches():
            if self.mode == PipelineMode.TRAINING:
                shuffle(self.batched_samples)
            for batch in self.batched_samples:
                sample_batch = []
                for s_id in batch:
                    sample = self.load_sample(self.samples[s_id])
                    sample_batch.append(sample)

                yield sample_batch
        else:
            random_samples = list(self.samples.values())
            if self.mode == PipelineMode.TRAINING:
                shuffle(random_samples)
            for sample in random_samples:
                yield self.load_sample(sample)


def parse_file(fn):
    out = {}
    with open(fn) as f:
        for line in f.readlines():
            line = line.strip()
            idx = line.find(" ")
            assert idx > 0, "invalid entry " + line
            out[line[:idx]] = line[idx + 1 :]
    return out


def read_2column_text(path: Union[Path, str]) -> Dict[str, str]:
    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) == 1:
                k, v = sps[0], ""
            else:
                k, v = sps
            if k in data:
                raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
            data[k] = v
    return data


def load_num_sequence_text(path: Union[Path, str], loader_type: str = "csv_int") -> Dict[str, List[Union[float, int]]]:
    if loader_type == "text_int":
        delimiter = " "
        dtype = int
    elif loader_type == "text_float":
        delimiter = " "
        dtype = float
    elif loader_type == "csv_int":
        delimiter = ","
        dtype = int
    elif loader_type == "csv_float":
        delimiter = ","
        dtype = float
    else:
        raise ValueError(f"Not supported loader_type={loader_type}")

    # path looks like:
    #   utta 1,0
    #   uttb 3,4,5
    # -> return {'utta': np.ndarray([1, 0]),
    #            'uttb': np.ndarray([3, 4, 5])}
    d = read_2column_text(path)

    # Using for-loop instead of dict-comprehension for debuggability
    retval = {}
    for k, v in d.items():
        try:
            retval[k] = [dtype(i) for i in v.split(delimiter)]
        except TypeError:
            logger.error(f'Error happened with path="{path}", id="{k}", value="{v}"')
            raise
    return retval


class NumElementsBatchSampler:
    def __init__(
        self,
        batch_bins: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        min_batch_size: int = 1,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
        padding: bool = True,
    ):
        assert batch_bins > 0
        if sort_batch != "ascending" and sort_batch != "descending":
            raise ValueError(f"sort_batch must be ascending or descending: {sort_batch}")
        if sort_in_batch != "descending" and sort_in_batch != "ascending":
            raise ValueError(f"sort_in_batch must be ascending or descending: {sort_in_batch}")

        self.batch_bins = batch_bins
        self.shape_files = shape_files
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last

        # utt2shape: (Length, ...)
        #    uttA 100,...
        #    uttB 201,...
        utt2shapes = [load_num_sequence_text(s, loader_type="csv_int") for s in shape_files]

        first_utt2shape = utt2shapes[0]
        for s, d in zip(shape_files, utt2shapes):
            if set(d) != set(first_utt2shape):
                raise RuntimeError(f"keys are mismatched between {s} != {shape_files[0]}")

        # Sort samples in ascending order
        # (shape order should be like (Length, Dim))
        keys = sorted(first_utt2shape, key=lambda k: first_utt2shape[k][0])
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {shape_files[0]}")
        if padding:
            # If padding case, the feat-dim must be same over whole corpus,
            # therefore the first sample is referred
            feat_dims = [np.prod(d[keys[0]][1:]) for d in utt2shapes]
        else:
            feat_dims = None

        # Decide batch-sizes
        batch_sizes = []
        current_batch_keys = []
        for key in keys:
            current_batch_keys.append(key)
            # shape: (Length, dim1, dim2, ...)
            if padding:
                for d, s in zip(utt2shapes, shape_files):
                    if tuple(d[key][1:]) != tuple(d[keys[0]][1:]):
                        raise RuntimeError(
                            "If padding=True, the " f"feature dimension must be unified: {s}",
                        )
                bins = sum(len(current_batch_keys) * sh[key][0] * d for sh, d in zip(utt2shapes, feat_dims))
            else:
                bins = sum(np.prod(d[k]) for k in current_batch_keys for d in utt2shapes)

            if bins > batch_bins and len(current_batch_keys) >= min_batch_size:
                batch_sizes.append(len(current_batch_keys))
                current_batch_keys = []
        else:
            if len(current_batch_keys) != 0 and (not self.drop_last or len(batch_sizes) == 0):
                batch_sizes.append(len(current_batch_keys))

        if len(batch_sizes) == 0:
            # Maybe we can't reach here
            raise RuntimeError("0 batches")

        # If the last batch-size is smaller than minimum batch_size,
        # the samples are redistributed to the other mini-batches
        if len(batch_sizes) > 1 and batch_sizes[-1] < min_batch_size:
            for i in range(batch_sizes.pop(-1)):
                batch_sizes[-(i % len(batch_sizes)) - 1] += 1

        if not self.drop_last:
            # Bug check
            assert sum(batch_sizes) == len(keys), f"{sum(batch_sizes)} != {len(keys)}"

        # Set mini-batch
        self.batch_list = []
        iter_bs = iter(batch_sizes)
        bs = next(iter_bs)
        minibatch_keys = []
        for key in keys:
            minibatch_keys.append(key)
            if len(minibatch_keys) == bs:
                if sort_in_batch == "descending":
                    minibatch_keys.reverse()
                elif sort_in_batch == "ascending":
                    # Key are already sorted in ascending
                    pass
                else:
                    raise ValueError("sort_in_batch must be ascending" f" or descending: {sort_in_batch}")

                self.batch_list.append(tuple(minibatch_keys))
                minibatch_keys = []
                try:
                    bs = next(iter_bs)
                except StopIteration:
                    break

        if sort_batch == "ascending":
            pass
        elif sort_batch == "descending":
            self.batch_list.reverse()
        else:
            raise ValueError(f"sort_batch must be ascending or descending: {sort_batch}")
