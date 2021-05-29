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
import tempfile
import unittest

from test.util.workdir import call_in_root
from tfaip.util.file.oshelper import ChDir

this_dir = os.path.dirname(os.path.realpath(__file__))
tfaip_dir = os.path.abspath(os.path.join(this_dir, "..", "..", "tfaip"))


class TestExperimenterScript(unittest.TestCase):
    def test_experimenter_example_no_tsp(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with ChDir(tmp_dir):
                call_in_root(
                    [
                        "tfaip-experimenter",
                        "--xlsx",
                        os.path.join(tfaip_dir, "scripts", "xlsxexperimenter", "example.xlsx"),
                        "--no_use_tsp",
                        "--dry_run",
                    ]
                )
                call_in_root(
                    [
                        "tfaip-experimenter",
                        "--xlsx",
                        os.path.join(tfaip_dir, "scripts", "xlsxexperimenter", "example.xlsx"),
                        "--no_use_tsp",
                    ]
                )

    # Disabled test since weired errors are thrown sometimes
    # def test_experimenter_example_with_tsp(self):
    #     try:
    #         subprocess.call(['tsp',  '-h'])
    #     except OSError:
    #         print("You need task-spooler (tsp) to be installed on your system to run this test! "
    #               "Install via 'apt install task-spooler' on Ubuntu/Debian.")
    #         try:
    #             # Install task spooler
    #             subprocess.call(['apt', 'install', '-y', 'task-spooler'])
    #         except OSError:
    #             print("Could not install task-spooler automatically. Stopping this test.")
    #             return

    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         with ChDir(tmp_dir):
    #             check_call(['tfaip-experimenter',
    #                         '--xlsx', os.path.join(tfaip_dir, 'scripts', 'xlsxexperimenter', 'example.xlsx'),
    #                         '--cpus', '0', '1',
    #                         '--dry_run'])
    #             check_call(['tfaip-experimenter',
    #                         '--xlsx', os.path.join(tfaip_dir, 'scripts', 'xlsxexperimenter', 'example.xlsx'),
    #                         '--cpus', '0', '1',
    #                         ])

    #             env = os.environ.copy()
    #             env['TS_SOCKET'] = 'cpu0'
    #             check_call(['tsp', '-c'], env=env)
    #             env['TS_SOCKET'] = 'cpu1'
    #             check_call(['tsp', '-c'], env=env)
