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
import bisect
from typing import Dict


def limit(min_val, val, max_val):
    return min(max_val, max(min_val, val))


class KeyHelper(object):
    def __init__(self, dct: Dict):
        self._keys = sorted(dct)
        self._size = len(self._keys)

    def get_keys(self, key):
        if key is None:
            raise TypeError("key is None")
        elif key in self._keys:
            return key
        else:
            idx = bisect.bisect(self._keys, key)
            if 0 < idx < self._size:
                return self._keys[idx - 1], self._keys[idx]
            elif idx > 0:
                return self._keys[idx - 1]
            elif idx < self._size:
                return self._keys[idx]

    def get_key(self, key):
        if key is None:
            raise TypeError("key is None")
        elif key in self._keys:
            return key
        else:
            idx = bisect.bisect(self._keys, key)
            idx = limit(0, idx, self._size - 1)
            return self._keys[idx]
