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
            return key,
        else:
            idx = bisect.bisect(self._keys, key)
            if 0 < idx < self._size:
                return self._keys[idx - 1], self._keys[idx]
            elif idx > 0:
                return self._keys[idx - 1],
            elif idx < self._size:
                return self._keys[idx],

    def get_key(self, key):
        if key is None:
            raise TypeError("key is None")
        elif key in self._keys:
            return key,
        else:
            idx = bisect.bisect(self._keys, key)
            idx = limit(0, idx, self._size - 1)
            return self._keys[idx]
#
# # if __name__ == '__main__':
# #     from random import Random
# #
# #     r = Random(12345)
# #     val_map = {r.randint(0, 100): x for x in range(10)}
# #     kh = KeyHelper(val_map)
# #     print kh.get_keys(15)
# #     print kh.get_keys(0)
# #     print kh.get_keys(98)
# #     print kh.get_keys(1)
# #     print kh.get_keys(57)
# #     print kh.get_keys(-57)
