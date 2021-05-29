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
from typing import NamedTuple, List

from pandas import read_excel
import numpy as np
import os
from subprocess import run, PIPE
from dataclasses import dataclass


@dataclass()
class Command:
    id: str
    command: str
    scenario: str
    cleanup: bool
    default_commands: dict
    grouped_commands: dict


@dataclass()
class Parameter:
    flag: str
    sub_param: bool
    split: bool = False
    delim: str = " "
    allow_empty: bool = False

    @staticmethod
    def parse(s: str, is_sub_param: bool):
        params = s.split(":")
        p = Parameter(params[0], is_sub_param)
        for param in params[1:]:
            setattr(p, param, True)

        return p

    def to_str(self, value) -> List[str]:
        if self.sub_param:
            if value is None:
                if self.allow_empty:
                    return [self.flag]
                return [""]
            value = "{}".format(value)
            if self.split:
                return [self.flag] + value.split(self.delim)

            return [self.flag, value]
        else:
            if value is None:
                if self.allow_empty:
                    return ["--" + self.flag]
                return []
            value = "{}".format(value)
            if self.split:
                return ["--" + self.flag] + value.split(self.delim)

            return ["--" + self.flag, value]


class Device(NamedTuple):
    device_id: int
    label: str
    is_gpu: bool


class XLSXExperimenter:
    def __init__(self, xlsx_path, gpus=None, cpus=None, dry_run=False, python=None, use_ts=False):
        gpus = gpus if gpus else []
        cpus = cpus if cpus else []
        assert not (use_ts and not gpus and not cpus)
        assert not (gpus and cpus)

        self.xlsx_path = xlsx_path
        self.with_gpu = gpus and len(gpus) > 0
        self.devices = [Device(int(gpu[0]), gpu, True) for gpu in gpus] + [Device(int(d[0]), d, False) for d in cpus]
        self.dry_run = dry_run
        self.python = python
        self.use_ts = use_ts

    def run(self):
        sheet = read_excel(self.xlsx_path)
        header = sheet.head(0)
        # ID column
        id = None
        params = None
        scenario = None
        skip_idx = None
        cleanup_idx = None
        command = None
        for i, v in enumerate(header.columns):
            if v == "ID":
                id = i
            elif v == "COMMAND":
                command = i
            elif v == "PARAMS":
                params = i
            elif v == "SCENARIO":
                scenario = i
            elif v == "SKIP":
                skip_idx = i
            elif v == "CLEANUP":
                cleanup_idx = i

        assert skip_idx is not None
        assert id is not None
        assert params is not None
        assert scenario is not None
        assert cleanup_idx is not None
        assert command is not None

        # read param groups
        groups = []
        last_group = None
        for i, group in enumerate(sheet.iloc[0]):
            if i < params:
                groups.append(None)
                continue

            if isinstance(group, str):
                last_group = group

            groups.append(last_group)

        parameters = []
        for i, parameter in enumerate(sheet.iloc[1]):
            if i < params:
                parameters.append(None)
                continue

            if isinstance(parameter, str):
                parameters.append(Parameter.parse(parameter, groups[i] != "default"))
            else:
                parameters.append(None)

        def get_parameter_by_flag(flag):
            return next(p for p in parameters if p and p.flag == flag)

        all_commands = []
        for index, row in sheet.iterrows():
            if index < 2:
                continue
            if not (isinstance(row[skip_idx], float) and np.isnan(row[skip_idx])):
                print("Skipping: {}".format(row[id]))
                continue

            cmd = Command(str(row[id]), row[command], row[scenario], row[cleanup_idx], {}, {})
            selected_group = cmd.default_commands
            for group, param, value in zip(groups, parameters, row):
                if param is None:
                    continue

                if group == "default":
                    selected_group = cmd.default_commands
                else:
                    if group not in cmd.grouped_commands:
                        cmd.grouped_commands[group] = {}

                    selected_group = cmd.grouped_commands[group]

                if isinstance(value, float) and np.isnan(value):
                    selected_group[param.flag] = None
                else:
                    selected_group[param.flag] = value

            all_commands.append(cmd)

        print("Starting {} calls".format(len(all_commands)))

        device_idx = 0
        for c in all_commands:
            if not self.use_ts:
                ts_socket = None
            elif self.with_gpu:
                ts_socket = "gpu{}".format(self.devices[device_idx].label)
            else:
                ts_socket = "cpu{}".format(self.devices[device_idx].label)

            tsp_call = []
            env = os.environ.copy()
            env["ID"] = str(c.id)
            env["PYTHON"] = self.python
            if self.use_ts:
                env["TS_SOCKET"] = ts_socket
                tsp_call = ["tsp", "-L", c.id]

            def single_param(k, v):
                params = get_parameter_by_flag(k).to_str(v)
                if params[0].endswith("__cls__"):
                    # Override class type of parameter
                    params[0] = params[0][: -len("__cls__")]
                return params

            def group_param(group_name, k, v):
                params = single_param(k, v)
                if len(params[0]) == 0:
                    return []  # empty parameter, skip
                else:
                    params[0] = "--" + group_name + "." + params[0]
                return params

            call = (
                tsp_call
                + [c.command, c.scenario]
                + (["--device.gpus", str(self.devices[device_idx].device_id)] if self.with_gpu else [])
                + sum([single_param(k, v) for k, v in c.default_commands.items()], [])
                + sum(
                    sum(
                        [
                            [group_param(group_name, k, v) for k, v in group.items()]
                            for group_name, group in c.grouped_commands.items()
                        ],
                        [],
                    ),
                    [],
                )
            )
            call = [str(c) for c in call if c]
            if not self.dry_run:
                if c.cleanup:
                    print("Cleanup not implemented yet")
                print("CALL [{}, {}]>> {}".format(ts_socket, c.id, " ".join(call)))
                out = run(call, env=env, check=True, stderr=PIPE, stdout=PIPE, universal_newlines=True)
                print(out.stderr)
                print(out.stdout)

                if self.use_ts:
                    result = run(["tsp", "-i"], env=env, check=True, capture_output=True)
                    if result.stdout.startswith(b"Exit status: died with exit code -1"):
                        raise Exception("Error in cmd")
            else:
                print("DRY RUN [{}, {}]>> {}".format(ts_socket, c.id, " ".join(call)))

            if self.use_ts:
                device_idx = (device_idx + 1) % len(self.devices)
