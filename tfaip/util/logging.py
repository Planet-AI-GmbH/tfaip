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
import logging

# Initialize logging
FORMAT = '{levelname:<8s} {asctime} {name:>30.30s}: {message}'
formatter = logging.Formatter(FORMAT, style='{')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger().handlers[0].setFormatter(formatter)

for handler in logging.getLogger("tensorflow").handlers:
    handler.setFormatter(formatter)


def setup_log(log_dir, append, log_name='train.log'):
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, log_name)
    file_handler = logging.FileHandler(filename, 'a' if append else 'w')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=logging.INFO)
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging training progress to '{filename}'")
