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
"""Utils for JSON-Encoding/Decoding"""
import importlib
import json
from dataclasses import is_dataclass
from typing import Any


class TFAIPJsonEncoder(json.JSONEncoder):
    """
    Json Encoder class that supports the encoding of dataclasses.
    Dataclasses are assumed to have @pai_dataclass or @dataclass_json annotation so that the function to_dict() and
    from_dict() are provided.
    Encoding decoding is enabled by storing an additional field "__json_dc_type" in the dict that is the module that
    must be imported to obtain the original class.

    See Also:
        TFAIPJsonDecoder
    """

    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            d = obj.to_dict()
            d["__json_dc_type__"] = obj.__class__.__module__ + ":" + obj.__class__.__name__
            return d
        else:
            return super().default(obj)


class TFAIPJsonDecoder(json.JSONDecoder):
    """
    Json Decoder class that supports the decoding of dataclasses if encoded with TFAIPJsonEncoder.

    See Also:
        TFAIPJsonEncoder
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, object_hook=self.object_hook_fn, **kwargs)

    def object_hook_fn(self, obj):
        if "__json_dc_type__" in obj:
            module, name = obj["__json_dc_type__"].split(":")
            del obj["__json_dc_type__"]
            cls = getattr(importlib.import_module(module), name)
            return cls.from_dict(obj)
        else:
            return obj
