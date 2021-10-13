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
import glob
import os
from typing import Iterable, Union, List, Optional


def glob_all(paths: Union[str, Iterable[str]], resolve_ending: Optional[str] = None) -> List[str]:
    """Resolves a list of files with wildcards.

    Basically this function applies pythons `glob.glob` to all path in `paths`.

    Optionally, specify `resolve_ending` which will read all files that end with `resolve_ending` and parse their
    content as additional paths.
    """
    if isinstance(paths, str):
        return glob_all([paths])
    else:
        out = []
        for p in paths:
            if resolve_ending is not None and p.endswith(resolve_ending):
                with open(p, "r") as f:
                    for line in f:
                        out += glob.glob(line)
            else:
                out += glob.glob(os.path.expanduser(p))

        return out
