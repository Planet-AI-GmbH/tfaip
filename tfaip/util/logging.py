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
"""Definitions for logging in tfaip"""
import os
import logging

# Initialize logging
import re
import sys

import numpy as np

FORMAT = "{levelname:<8s} {asctime} {name:>30.30s}: {message}"
formatter = logging.Formatter(FORMAT, style="{")
TFAIP_LOG_LEVEL = getattr(logging, os.environ.get("TFAIP_LOG_LEVEL", "INFO").upper())
this_logger = logging.getLogger(__name__)
logging.basicConfig(level=TFAIP_LOG_LEVEL)
logging.getLogger().handlers[0].setFormatter(formatter)

for handler in logging.getLogger("tensorflow").handlers:
    handler.setFormatter(formatter)


# Define a custom extension handler so that the exceptions are logged to logging
def handle_exception(exc_type, exc_value, exc_traceback):
    if exc_type is not None and issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    this_logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception  # Overwrite the excepthook


def logger(name):
    return logging.getLogger(name)


class WriteToLogFile:
    """Context to write logging to a log file

    When exit the stack the log file gets closed and the handler is removed
    """

    def __init__(self, log_dir: str, append: bool, log_name="train.log"):
        assert log_dir is not None
        os.makedirs(log_dir, exist_ok=True)
        self.filename = os.path.join(log_dir, log_name)
        self.file_handler = None
        self.append = append

    def __enter__(self):
        self.file_handler = logging.FileHandler(self.filename, "a" if self.append else "w", encoding="utf-8")
        self.file_handler.setFormatter(formatter)
        self.file_handler.setLevel(level=TFAIP_LOG_LEVEL)
        logging.getLogger().addHandler(self.file_handler)
        this_logger.info(f"Logging to '{self.filename}'")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            handle_exception(exc_type, exc_val, exc_tb)  # Log exception before log gets closed
        logging.getLogger().removeHandler(self.file_handler)
        self.file_handler.flush()
        self.file_handler.close()


class ParseLogFile:
    def __init__(self, log_dir: str, log_name="train.log"):
        assert log_dir is not None
        self.filename = os.path.join(log_dir, log_name)

    def get_metrics(self) -> dict:
        if not os.path.isfile(self.filename):
            return dict()
        with open(self.filename, "r") as f:
            # get last epoch line index
            res_lines = []
            for [i, line] in enumerate(f, 0):
                if "Results of epoch" in line:
                    res_lines.append([i, line])

            assert len(res_lines) > 0, "Found no result lines in log-file"
            last_res = res_lines[-1][1]

            # parse losses and metrics
            metrics_re = re.findall(r"([\w/]+): ([-0-9.]+)", last_res)
            metrics = {ele[0]: ele[1] for ele in metrics_re}
        return metrics
