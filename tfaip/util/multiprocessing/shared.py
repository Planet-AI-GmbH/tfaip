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
from dataclasses import field

import dataclasses_json
from paiargparse import pai_meta


def shared_value_field(**kwargs):
    """
    Usage in dataclass:
        param: multiprocessing.Value = shared_value_field(default_factory=lambda: mp_manager().Value('i', 0))
    """
    assert "default_factory" in kwargs

    def decode(x):
        default = kwargs["default_factory"]()
        default.value = x
        return default

    kwargs["metadata"] = {
        **kwargs.get("metadata", {}),
        **dataclasses_json.config(encoder=lambda x: x.value, decoder=decode),
        **pai_meta(mode="ignore"),
    }
    return field(**kwargs)
