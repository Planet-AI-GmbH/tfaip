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
"""Utils for versioning of the code"""
import logging
from typing import Optional
import os

logger = logging.getLogger(__name__)
this_dir = os.path.dirname(os.path.realpath(__file__))


def get_commit_hash() -> Optional[str]:
    """
    Retrieve the commit hash of tfaip or None if tfaip is not installed via git, e.g. directly from the pip packages.
    """
    try:
        import git  # local import so that no crash occurs if git is not installed

        repo = git.Repo(path=this_dir, search_parent_directories=True)
        h = repo.head.object.hexsha
        logger.debug(f"Git commit hash {h}")
        return h
    except Exception:  # pylint: disable=broad-except
        logger.debug("Could not read commit hash. Maybe tfaip not running from git repository or git not installed")
        # Not a git repo
        return None
