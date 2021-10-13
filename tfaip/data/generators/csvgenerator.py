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
"""Generator for CSV files.

The path of the location of the CSV-file will be prepended to the filename references, so that the working dir does
not need to be changed.

Example:
    filename.jpg,139902,THIS IS THE GT,2021-08-19

    1. The `filename.jpg` is a path (or text).
    2. The number is the length of the data within the path
    3. The ground truth text
    4. The date (when created)

Alternatives:
    - TSVGeneratorParams for tab separated files ("\t")
    - SCSVGeneratorParams for semi colon separated files (";")
"""
import itertools
import logging
import multiprocessing
from dataclasses import dataclass, field
from pathlib import Path
from random import shuffle
from typing import Type, Iterable, Union, List, Optional, NamedTuple, TypeVar, Dict

from paiargparse import pai_dataclass, pai_meta

from tfaip import DataGeneratorParams, Sample, PipelineMode
from tfaip.data.generators.autobatching.autobatcher import auto_batch_to_same_samples_per_batch
from tfaip.data.pipeline.datagenerator import DataGenerator
from tfaip.util.file.glob_util import glob_all

logger = logging.getLogger(__name__)


@pai_dataclass
@dataclass
class CSVGeneratorParams(DataGeneratorParams):
    files: List[str] = field(default_factory=list)
    batch_bins: Optional[int] = field(
        default=None,
        metadata=pai_meta(
            help="Generate batches based on the csv sizes. Number of bins in a batch <= batch_bins. "
            "If None, no batching will be performed by the generator, instead by the tf.data.Dataset"
            " (the DataPipeline)."
        ),
    )

    @staticmethod
    def cls() -> Type["DataGenerator"]:
        return CSVGenerator

    def __post_init__(self):
        self.files = glob_all(self.files)


class CSVTuple(NamedTuple):
    data_path: str  # path to the actual data
    data_bins: int  # Length of the data
    gt: str  # parsed ground truth
    date: str  # parsed creation date of the sample

    @staticmethod
    def parse(values: List[str], prefix: Path = None) -> "CSVTuple":
        path = Path(values[0])
        if prefix is not None:
            path = prefix / path
        return CSVTuple(path.as_posix(), int(values[1]), values[2], values[3])

    def to_sample(self) -> Sample:
        return Sample(
            inputs=self.data_path, targets=self.gt, meta={"data_path": self.data_path, "data_bins": self.data_bins}
        )


def _parse_file(filename: Path, separator: str) -> List[CSVTuple]:
    with open(filename.as_posix()) as f:
        return [CSVTuple.parse(line.split(separator), prefix=filename.parent.resolve()) for line in f.readlines()]


T = TypeVar("T", bound=CSVGeneratorParams)


class CSVGenerator(DataGenerator[T]):
    """DataGenerator for CSV files

    `with_threading` allows to load the samples in a process pool (i.e. multiple samples a processed simultaneously)
    which is advantageous to faster load samples from the hard drive (e.g. audio or images).

    Subclasses can adapt the `separator` of the parsed files. See, e.g., the TSVGenerator which uses '\t' instead of ','.
    """

    def __init__(self, separator=",", with_threading=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.params.files) > 0, "No files given."
        self._separator = separator
        self._with_threading = with_threading

        logger.info(f"Parsing files {self.params.files}")
        self._entries: List[CSVTuple] = list(
            itertools.chain(*(_parse_file(Path(fn), self._separator) for fn in self.params.files))
        )
        if self.yields_batches():
            logger.info("Computing initial batches")
            self._initial_batches = self._get_batched_entries()

    def yields_batches(self) -> bool:
        return self.params.batch_bins is not None

    def __len__(self):
        return len(self._entries)

    def steps_per_epoch(self, batch_size: int, scale_factor: float = 1):
        if self.yields_batches():
            return int(scale_factor * len(self._initial_batches))
        else:
            return super().steps_per_epoch(batch_size, scale_factor)

    def load_sample(self, sample: Sample) -> Sample:
        """Method that actually loads the sample.

        This function can/should be overwritten by a subclass.
        """
        return sample

    def _get_batched_entries(self):
        id_entry: Dict[str, CSVTuple] = {f"{i}": d for i, d in enumerate(self._entries)}
        data_to_shape = {k: (v.data_bins,) for k, v in id_entry.items()}
        gt_to_shape = {k: (len(v.gt),) for k, v in id_entry.items()}
        batched_ids = auto_batch_to_same_samples_per_batch(self.params.batch_bins, [data_to_shape, gt_to_shape])
        return [[id_entry[s] for s in batch] for batch in batched_ids]

    def generate(self) -> Iterable[Union[Sample, List[Sample]]]:
        if self.yields_batches():
            if self.mode == PipelineMode.TRAINING:
                shuffle(self._initial_batches)

            if self._with_threading:

                def loader(batch):
                    return [self.load_sample(s.to_sample()) for s in batch]

                with multiprocessing.pool.ThreadPool(processes=8) as p:
                    if self.mode == PipelineMode.TRAINING:
                        for batch in p.imap_unordered(loader, self._initial_batches):
                            yield batch
                    else:
                        for batch in p.imap(loader, self._initial_batches):
                            yield batch
            else:
                for batch in self._initial_batches:
                    yield [self.load_sample(s.to_sample()) for s in batch]
        else:
            if self.mode == PipelineMode.TRAINING:
                shuffle(self._entries)

            if self._with_threading:

                def loader(e):
                    return self.load_sample(e.to_sample())

                with multiprocessing.pool.ThreadPool(processes=8) as p:
                    if self.mode == PipelineMode.TRAINING:
                        for entry in p.imap_unordered(loader, self._entries):
                            yield entry
                    else:
                        for entry in p.imap(loader, self._entries):
                            yield entry
            else:
                for entry in self._entries:
                    yield self.load_sample(entry.to_sample())


@pai_dataclass
@dataclass
class TSVGeneratorParams(CSVGeneratorParams):
    @staticmethod
    def cls() -> Type["DataGenerator"]:
        return TSVGenerator


class TSVGenerator(CSVGenerator[T]):
    def __init__(self, **kwargs):
        super().__init__(separator="\t", **kwargs)


@pai_dataclass
@dataclass
class SCSVGeneratorParams(CSVGeneratorParams):
    @staticmethod
    def cls() -> Type["DataGenerator"]:
        return SCSVGenerator


class SCSVGenerator(CSVGenerator[T]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, separator=";", **kwargs)
