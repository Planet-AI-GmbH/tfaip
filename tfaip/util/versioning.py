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
import subprocess
import logging
from typing import Optional
import os


logger = logging.getLogger(__name__)
this_dir = os.path.dirname(os.path.realpath(__file__))


def get_commit_hash() -> Optional[str]:
    try:
        h = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=this_dir).strip()
        logger.debug(f"Git commit hash {h}")
        return h
    except subprocess.CalledProcessError:
        logger.debug("tfaip not running from git repository")
        # Not a git repo
        return None
