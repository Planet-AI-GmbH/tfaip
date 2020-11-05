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
import enum
from argparse import ArgumentParser, Action
from typing import Optional, List, Union, Any, Type
import Levenshtein


def dc_meta(*, help: str = None):
    return locals()


def str2bool(v: str) -> bool:
    return v.lower() == 'true' or (v.isdigit() and int(v) > 0)


def str2int_enum(v: str, enum_cls: Type[enum.IntEnum]):
    if v.isnumeric():
        return enum_cls(int(v))

    for k, e in enum_cls.__members__.items():
        if k.lower() == v.lower():
            return e

    raise ValueError(f"Could not match {v} to any valid key in {enum_cls.__members__.keys()}.")


def str2str_enum(v: str, enum_cls: Type[enum.Enum]):
    for k, e in enum_cls.__members__.items():
        if k.lower() == v.lower() or e.value.lower() == v.lower():
            return e

    raise ValueError(f"Could not match {v} to any valid key in {enum_cls.__members__.keys()}.")


def parse_list_arg(val, formatter):
    if len(val) == 0:
        return []

    if ',' not in val:
        # only one argument, may also have no brackets at all
        if val[0] == '[' and val[-1] == ']':
            return [formatter(val[1:-1])]
        return [formatter(val)]

    if val[0] != '[':
        raise ValueError("List argument must start with '[' but got '{}'".format(val))
    elif val[-1] != ']':
        raise ValueError("List argument must end with ']' but got '{}'".format(val))

    return [formatter(s.strip()) for s in val[1:-1].split(',') if len(s.strip()) > 0]


def is_int_enum(t):
    try:
        return issubclass(t, enum.IntEnum)
    except TypeError:
        return False


def is_str_enum(t):
    try:
        return issubclass(t, enum.Enum)
    except TypeError:
        return False


def enum_class_from_field(field) -> Optional[Type[enum.Enum]]:
    if is_int_enum(field.type) or is_str_enum(field.type):
        return field.type
    elif is_enum_list(field.type, enum.Enum):
        return extract_list_enum_type(field.type)
    return None


def extract_list_enum_type(t):
    try:
        base_type = t.__reduce__()[1][0]
    except IndexError:
        return None

    if base_type == Union:
        if len(t.__args__) != 2 and t.__args__[1] != type(None):
            return
        t = t.__args__[0]
        try:
            base_type = t.__reduce__()[1][0]
        except IndexError:
            return

    if base_type != List:
        return None

    return t.__args__[0]


def is_enum_list(t, enum_t):
    try:
        return issubclass(extract_list_enum_type(t), enum_t)
    except TypeError:
        return False


def make_store_dataclass_action(data_cls: Any):
    class DataClassAction(Action):
        def __call__(self, parser, args, values, option_string=None):
            params = getattr(args, self.dest)
            for kv in values:
                if len(kv.split('=')) == 2:
                    key, val = kv.split("=")
                else:
                    raise ValueError(f"Could not parse '{kv}' must by KEY=VALUE")

                # get parameter of data_cls
                try:
                    field = next(f for name, f in data_cls.__dataclass_fields__.items() if f.name == key)
                except StopIteration:
                    arguments = generate_argument_list(data_cls)
                    closest = None
                    for arg in arguments:
                        if key in arg:
                            closest = key

                    if not closest and len(arguments) > 0:
                        distances = {a: Levenshtein.distance(a, key) for a in arguments}
                        closest = sorted(arguments, key=lambda a: distances[a])[0]

                    str_args = argument_list_to_str(arguments)
                    raise AttributeError(f'Invalid argument {key}. Did you mean "{closest}={val}"? Available arguments {str_args}')

                # Single
                if field.type == Optional[str] or field.type == str:
                    setattr(params, key, val)
                elif field.type == Optional[int] or field.type == int:
                    setattr(params, key, int(val))
                elif field.type == Optional[float] or field.type == float:
                    setattr(params, key, float(val))
                elif field.type == Optional[bool] or field.type == bool:
                    setattr(params, key, str2bool(val))

                # Lists
                elif field.type == Optional[List[str]] or field.type == List[str]:
                    setattr(params, key, parse_list_arg(val, str))
                elif field.type == Optional[List[int]] or field.type == List[int]:
                    setattr(params, key, parse_list_arg(val, int))
                elif field.type == Optional[List[float]] or field.type == List[float]:
                    setattr(params, key, parse_list_arg(val, float))

                # Enum
                elif is_int_enum(field.type):
                    setattr(params, key, str2int_enum(val, field.type))
                elif is_str_enum(field.type):
                    setattr(params, key, str2str_enum(val, field.type))

                # Enum Lists
                elif is_enum_list(field.type, enum.Enum):
                    t = extract_list_enum_type(field.type)
                    l = parse_list_arg(val, str)
                    if is_int_enum(field.type):
                        setattr(params, key, list([str2int_enum(v, t) for v in l]))
                    else:
                        setattr(params, key, list([str2str_enum(v, t) for v in l]))


                # Unknown
                else:
                    raise TypeError("Unknown type of field {}".format(field.type))

    return DataClassAction


def field_is_dataclass(field):
    return hasattr(field.type, "__dataclass_fields__") or \
           (hasattr(field.type, '__origin__') and field.type.__origin__ == Union and hasattr(field.type.__args__[0], '__dataclass_fields__'))


def extract_dataclass_from_field(field):
    if hasattr(field.type, "__dataclass_fields__"):
        return field.type
    else:
        return field.type.__args__[0]


def generate_help(data_cls: Any):
    default = data_cls()
    all_fields = {name: f for name, f in data_cls.__dataclass_fields__.items() if not field_is_dataclass(f) and not name.endswith('_')}

    def generate_help_for_field(name):
        field = all_fields[name]
        enum_class = enum_class_from_field(field)
        msg = f"{name}={getattr(default, name)}"
        if enum_class:
            vs = [str(e.value) for e in enum_class]
            msg += " ∈{" + ', '.join(vs) + "}"
        if field.metadata:
            if 'help' in field.metadata:
                msg += f"\n    {field.metadata['help']}"
        return msg

    return "\n".join([generate_help_for_field(name) for name in sorted(all_fields.keys())])


def generate_argument_list(data_cls: Any):
    return [name for name, f in data_cls.__dataclass_fields__.items() if not field_is_dataclass(f) and not name.endswith('_')]


def argument_list_to_str(arguments):
    return "[" + ', '.join(f"{name}" for name in arguments) + "]"


def add_args_group(parser: ArgumentParser, group: str, params_cls: Any, default=None):
    default = default if default else params_cls()
    params_cls = default.__class__
    parser.add_argument("--" + group,
                        action=make_store_dataclass_action(params_cls),
                        default=default,
                        nargs='*',
                        metavar="KEY=VAL",
                        help=generate_help(params_cls))

    for name, field in params_cls.__dataclass_fields__.items():
        if field_is_dataclass(field) and not name.endswith("_"):
            add_args_group(parser, group=name, params_cls=extract_dataclass_from_field(field), default=getattr(default, name))
