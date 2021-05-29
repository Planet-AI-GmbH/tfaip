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
import os
import platform
from subprocess import check_call

this_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(this_dir, "..", "..")


def workdir_path(name: str, *args):
    # name expected to be a path .../test/scenario/{SCENARIO_NAME}/workdir
    # or .../test/scenario/workdir (to support single scenario setups)
    scenario_dir = os.path.join("test", "scenario")
    name = os.path.abspath(name)
    if scenario_dir not in name:
        scenario_dir = "tfaip_scenario_test"
    return workdir_path_with_path(name, scenario_dir, *args)


def workdir_path_with_path(name: str, scenario_dir: str, *args):
    assert scenario_dir in name
    pos = name.find("/", name.rfind(scenario_dir) + len(scenario_dir) + 1)
    if pos >= 0:
        base_dir = name[:pos]
    else:
        base_dir = os.path.dirname(name)
    wd = os.path.join(base_dir, "workdir")
    assert os.path.exists(wd)
    assert os.path.isdir(wd)
    return os.path.join(wd, *args)


def call_in_root(args, env=None):
    if env is None:
        env = os.environ.copy()
    sep = ";" if platform.system() == "Windows" else ":"
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] += sep + os.path.join(root_dir)
    else:
        env["PYTHONPATH"] = os.path.join(root_dir)
    return check_call(args, env=env)
