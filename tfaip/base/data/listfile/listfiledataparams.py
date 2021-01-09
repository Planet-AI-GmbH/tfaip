# Copyright 2020 The tfaip authors. All Rights Reserved.
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
from dataclasses import dataclass, field, fields
from typing import Optional, List

from dataclasses_json import dataclass_json

from tfaip.base.data.databaseparams import DataGeneratorParams, DataBaseParams
from tfaip.util.argumentparser import dc_meta


@dataclass_json
@dataclass
class ListsFilePipelineParams(DataGeneratorParams):
    lists: Optional[List[str]] = field(default_factory=list, metadata=dc_meta(
        help="Training list files."
    ))
    list_ratios: Optional[List[float]] = field(default=None, metadata=dc_meta(
        help="Ratios of picking list files. Must be supported by the scenario"
    ))

    def validate(self):
        super(ListsFilePipelineParams, self).validate()

        if self.lists:
            if not self.list_ratios:
                self.list_ratios = [1.0] * len(self.lists)
            else:
                if len(self.list_ratios) != len(self.lists):
                    raise ValueError(f"Length of list_ratios must be equals to number of lists. Got {self.list_ratios}!={self.lists}")

    def split(self) -> List['ListFilePipelineParams']:
        out = []
        for l in self.lists:
            t = ListFilePipelineParams()
            for f in fields(self.__class__):
                if f.name in {'lists', 'list_ratios'}:
                    continue
                setattr(t, f.name, getattr(self, f.name))
            t.list = l
            out.append(t)
        return out


@dataclass_json
@dataclass
class ListFilePipelineParams(DataGeneratorParams):
    list: str = field(default=None, metadata=dc_meta(
        help="The validation list for testing the model during training."
    ))


@dataclass_json
@dataclass
class ListFileDataParams(DataBaseParams):
    train: ListsFilePipelineParams = field(default_factory=ListsFilePipelineParams, metadata=dc_meta(arg_mode='snake'))
    val: ListFilePipelineParams = field(default_factory=ListFilePipelineParams, metadata=dc_meta(arg_mode='snake'))
    lav: ListsFilePipelineParams = field(default_factory=ListsFilePipelineParams, metadata=dc_meta(arg_mode='snake'))

