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
import numpy as np
from numpy import ndarray
from typing import List


class HexByteConverter(object):
    def __init__(self):
        lut_16 = []
        for i in range(16):
            lut_16.append(hex(i)[2].upper())
        self._int_hex_lut_256 = []
        for j in range(16):
            for i in range(16):
                self._int_hex_lut_256.append(lut_16[j] + lut_16[i])
        self._hex_int_lut_256 = []
        for j in range(128):
            tmp = [0] * 128
            self._hex_int_lut_256.append(tmp)
        self._hex_int_lut_256_flat = [0] * 65536
        for j in range(16):
            for i in range(16):
                val = 16 * j + i
                lj = lut_16[j]
                li = lut_16[i]
                ljl = lj.lower()
                lil = li.lower()
                olj = ord(lj)
                oli = ord(li)
                oljl = ord(ljl)
                olil = ord(lil)
                self._hex_int_lut_256[olj][oli] = val
                self._hex_int_lut_256[oljl][olil] = val
                self._hex_int_lut_256_flat[olj * 256 + oli] = val
                self._hex_int_lut_256_flat[oljl * 256 + olil] = val

    def bytes2hex1d(self, bytes1d: List[int]) -> str:
        hx = ""
        for byte in bytes1d:
            hx += self._int_hex_lut_256[byte]
        return hx

    def bytes2hex2d(self, bytes2d: List[List[int]]) -> str:
        hx = ""
        for bytes1d in bytes2d:
            hx += self.bytes2hex1d(bytes1d) + "\n"
        return hx

    def hex2bytes1d(self, hex1d: str) -> List[int]:
        retval = []
        for k in range(0, len(hex1d), 2):
            j = ord(hex1d[k])
            i = ord(hex1d[k + 1])
            retval.append(self._hex_int_lut_256[j][i])
        return retval

    def hex2bytes2d(self, hex2d: str) -> np.ndarray:
        retval = []
        lines = hex2d.splitlines(False)
        for line in lines:
            if len(line) > 0:
                retval.append(self.hex2bytes1d(line))
        return np.asarray(retval).astype(np.uint8)


hex_byte_converter = HexByteConverter()
