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
"""Definition of derived types and type checking"""
import inspect
import typing as typing_native
from contextlib import suppress
from functools import wraps
from typing import Union

import numpy as np

AnyNumpy = Union[
    np.ndarray, np.int, np.int8, np.int16, np.int32, np.int64, np.float, np.float16, np.float32, np.float64, np.bool]


def enforce_types(callable):
    spec = inspect.getfullargspec(callable)

    def check_types(*args, **kwargs):
        parameters = dict(zip(spec.args, args))
        parameters.update(kwargs)
        for name, value in parameters.items():
            with suppress(KeyError):  # Assume un-annotated parameters can be any type
                type_hint = spec.annotations[name]
                if isinstance(type_hint, typing_native._SpecialForm):  # pylint: disable=protected-access
                    # No check for typing.Any, typing.Union, typing.ClassVar (without parameters)
                    continue
                try:
                    actual_type = type_hint.__origin__
                except AttributeError:
                    # In case of non-typing types (such as <class 'int'>, for instance)
                    actual_type = type_hint
                # In Python 3.8 one would replace the try/except with
                # actual_type = typing.get_origin(type_hint) or type_hint
                if isinstance(actual_type, typing_native._SpecialForm):  # pylint: disable=protected-access
                    # case of typing.Union[…] or typing.ClassVar[…]
                    actual_type = type_hint.__args__
                # print(value)
                # print(actual_type)
                # print(isinstance(value,actual_type))
                try:
                    ok = isinstance(value, actual_type)
                except TypeError:
                    # TODO: better handling for example isinstance([],List[str]) because str leads to generics
                    pass
                if not ok:
                    raise TypeError(
                        'Unexpected type for \'{}\' (expected {} but found {})'.format(name, type_hint, type(value)))

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            check_types(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    if inspect.isclass(callable):
        callable.__init__ = decorate(callable.__init__)
        return callable

    return decorate(callable)
