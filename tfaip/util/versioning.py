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
