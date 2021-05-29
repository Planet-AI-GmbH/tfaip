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
"""Utils for writing tests with tfaip

In custom tests, call ``set_test_init`` in the root ``__init__.py`` of the tests (not the root dir of the project!)
"""
import os

import tfaip.util.logging as logging
from tfaip.device.device_config import DeviceConfigParams, DeviceConfig

# Initialize "Devices" (for all tests), using the same config!
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable gpu usage


def setup_test_init():
    """Function that should be called in the root __init__ of all tests

    The call ensures that the Devices and the logging are initialized correctly
    """

    DeviceConfig(DeviceConfigParams())

    logging.logger(__name__).debug("Set up device config for testing")
