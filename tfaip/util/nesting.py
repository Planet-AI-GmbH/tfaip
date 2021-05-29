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
def to_flat(nested_dict, including_lists=True):
    # Flats an arbitrarily deep nested dictionary of anything. If 'including_lists' is true, lists are also replaced by dictionary entries
    def _add(key, value, _flat_dict):
        if isinstance(value, dict):
            for k, v in value.items():
                _add(combine_keys(key, k), v, _flat_dict)
        elif including_lists and isinstance(value, list):
            for idx, v in enumerate(value):
                _add(combine_key_and_list_idx(key, idx), v, _flat_dict)
        else:
            _flat_dict[key] = value

    flat_dict = dict()
    for k, v in nested_dict.items():
        _add(k, v, flat_dict)
    return flat_dict


def to_nested(flat_dict):
    # Reconstructs the original nested dictionary from the flattened dictionary.

    nested_dict = dict()
    for k, v in flat_dict.items():
        key_path = split_key(k)
        dst = nested_dict
        last_key = False
        for key_idx, _subkey in enumerate(key_path):
            if key_idx == (len(key_path) - 1):
                last_key = True
            subkey_and_indices = split_key_and_list_indices(_subkey)
            subkey = subkey_and_indices[0]
            list_indices = subkey_and_indices[1:]
            if len(list_indices) > 0:
                if subkey not in dst:
                    dst[subkey] = list()
                dst = dst[subkey]
                for idx, list_idx_str in enumerate(list_indices):
                    list_idx = int(list_idx_str)
                    lst_length = len(dst)
                    dst.extend([None for _ in range(lst_length, list_idx + 1)])
                    if dst[list_idx] == None:
                        if idx == (len(list_indices) - 1):
                            if last_key:
                                dst[list_idx] = v
                            else:
                                dst[list_idx] = dict()
                        else:
                            dst[list_idx] = list()
                    dst = dst[list_idx]
            else:
                if last_key:
                    dst[subkey] = v
                else:
                    if subkey not in dst:
                        dst[subkey] = dict()
                    dst = dst[subkey]
    return nested_dict


def combine_keys(kp, kc):
    # combines parent key (kp: str) and child key (kc: str)
    return kp + "/" + kc


def split_key(k):
    # splits into subkeys
    return k.split("/")


def combine_key_and_list_idx(k, idx):
    # combines a key (k: str) with a list index (idx: int)
    return k + ":" + str(idx)


def split_key_and_list_indices(k):
    # splits into key and list indices
    return k.split(":")


if __name__ == "__main__":

    nested_dict = dict()
    nested_dict["a"] = 1
    nested_dict["b"] = dict()
    nested_dict["c"] = [{"c1": 10, "c2": 11}, [12, 13, 14], 3]

    nested_dict["b"]["b1"] = dict()
    nested_dict["b"]["b2"] = None
    nested_dict["b"]["b3"] = 5

    nested_dict["b"]["b1"]["b11"] = 6
    nested_dict["b"]["b1"]["b12"] = 7
    nested_dict["b"]["b1"]["b13"] = 8
    nested_dict["b"]["b1"]["b14"] = 9

    flat_dict = to_flat(nested_dict)
    reconst_nested_dict = to_nested(flat_dict)

    print("Original nested dict: ", nested_dict)
    print("Flattened dict: ", flat_dict)
    print("Reconstructed nested dict: ", reconst_nested_dict)
