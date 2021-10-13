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

from tfaip.scripts.xlsxexperimenter.run_xlsx_experimenter import XLSXExperimenter
from tfaip.util.testing.workdir import call_in_root
from tfaip.util.file.oshelper import ChDir

this_dir = os.path.dirname(os.path.realpath(__file__))
tfaip_dir = os.path.abspath(os.path.join(this_dir, "..", "..", "tfaip"))
test_dir = os.path.abspath(os.path.join(this_dir, "..", "..", "test"))


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

    def test_experimenter_example_update_mode(self):
        from shutil import copy2
        from pandas import ExcelFile

        with tempfile.TemporaryDirectory() as tmp_dir:
            with ChDir(tmp_dir):
                # copy relevant files to tmp dir, necessary due to relative paths in logger
                os.mkdir("experimenter_test_files")
                os.mkdir("experimenter_test_files/1")
                os.mkdir("experimenter_test_files/2")
                path_src = os.path.join(test_dir, "scripts", "experimenter_test_files")
                copy2(os.path.join(path_src, "test_experimenter.xlsx"), "experimenter_test_files")
                copy2(os.path.join(path_src, "1", "train.log"), os.path.join("experimenter_test_files", "1"))
                copy2(os.path.join(path_src, "2", "train.log"), os.path.join("experimenter_test_files", "2"))

                call_in_root(
                    [
                        "tfaip-experimenter",
                        "--xlsx",
                        os.path.join("experimenter_test_files", "test_experimenter.xlsx"),
                        "--update",
                    ]
                )

                # verify existence of second worksheet in tmp xlsx
                xlsx_file = ExcelFile(os.path.join("experimenter_test_files", "test_experimenter.xlsx"))
                assert len(xlsx_file.sheet_names) == 2

    def test_experimenter_non_existent_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with ChDir(tmp_dir):
                with self.assertRaises(FileNotFoundError):
                    XLSXExperimenter("non_existent.xlsx")

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
