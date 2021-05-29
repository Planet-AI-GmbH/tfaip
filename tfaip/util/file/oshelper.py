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
"""os utilities"""
import os


class ChDir:
    """
    Utility class to change the working directory in a `with` block and restoring the old dir afterwards.

    Usage:
    ```
    with ChDir("my_path"):
        # Working dir is my_path

    # working dir is reset
    ```
    """

    def __init__(self, path):
        self._cd = os.getcwd()  # store the current working dir to reset it on exit
        self.path = path

    def __enter__(self):
        self._cd = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self._cd)
