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
import os

this_dir = os.path.dirname(os.path.realpath(__file__))


def get_workdir(name: str, *args):
    # name expected to be a path .../test/scenario/{SCENARIO_NAME}/workdir
    # or .../test/scenario/workdir (to support single scenario setups)
    scenario_dir = os.path.join('test', 'scenario')
    assert(scenario_dir in name)
    pos = name.find('/', name.rfind(scenario_dir) + len(scenario_dir) + 1)
    if pos >= 0:
        base_dir = name[:pos]
    else:
        base_dir = os.path.dirname(name)
    wd = os.path.join(base_dir, 'workdir')
    assert(os.path.exists(wd))
    assert(os.path.isdir(wd))
    return os.path.join(wd, *args)
