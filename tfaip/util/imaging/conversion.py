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
import logging
import numpy as np


logger = logging.getLogger(__name__)


def make_uint8(img):
    if img.dtype == np.uint8:
        logger.debug('image_io_helpers: make_uint8: already uint8 -> returning input')
        return img
    elif img.dtype == np.float64:
        logger.debug('image_io_helpers: make_uint8: converting image float64 -> uint8')
        if np.min(img) < 0 or np.max(img) > 1:
            raise Exception("invalid float64 image values - must be in [0.0, 1.0] ")
        retval = (255.9999999999999 * img).astype(np.uint8)
        logger.debug('image_io_helpers: make_uint8: [' + str(np.min(img)) + ',' + str(np.max(img)) + '] -> [' + str(
            np.min(retval)) + ',' + str(np.max(retval)) + ']')
        return retval
    elif img.dtype == np.float16:
        logger.debug('image_io_helpers: make_uint8: converting image float16 -> uint8')
        if np.min(img) < 0 or np.max(img) > 1:
            raise Exception("invalid float64 image values - must be in [0.0, 1.0] ")
        retval = (255.9 * img).astype(np.uint8)
        logger.debug('image_io_helpers: make_uint8: [' + str(np.min(img)) + ',' + str(np.max(img)) + '] -> [' + str(
            np.min(retval)) + ',' + str(np.max(retval)) + ']')
        return retval
    elif img.dtype == np.float32:
        logger.debug('image_io_helpers: make_uint8: converting image float32 -> uint8')
        if np.min(img) < 0 or np.max(img) > 1:
            raise Exception("invalid float64 image values - must be in [0.0, 1.0] ")
        retval = (255.99999 * img).astype(np.uint8)
        logger.debug('image_io_helpers: make_uint8: [' + str(np.min(img)) + ',' + str(np.max(img)) + '] -> [' + str(
            np.min(retval)) + ',' + str(np.max(retval)) + ']')
        return retval
    else:
        raise Exception('unsupported image dtype: ' + str(img.dtype))
