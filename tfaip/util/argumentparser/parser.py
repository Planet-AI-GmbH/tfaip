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
from typing import Optional, List, Union, Any, Type, Dict
import Levenshtein
import logging

from tfaip.base.resource.resource import Resource

logger = logging.getLogger(__name__)


class TFAIPArgumentParser(ArgumentParser):
    def __init__(self, ignore_required=False, *args, **kwargs):
        super(TFAIPArgumentParser, self).__init__(*args, **kwargs)
        self._required_fields = {}
        self._ignore_required = ignore_required

    def get_required_fields(self, group):
        if group not in self._required_fields:
            self._required_fields[group] = []
        return self._required_fields[group]

    def parse_known_args(self, *args, **kwargs):
        r = super(TFAIPArgumentParser, self).parse_known_args(*args, **kwargs)

        if not self._ignore_required:
            not_set_fields = {k: v for k, v in self._required_fields.items() if len(v) > 0}
            if len(not_set_fields) > 0:
                s = [f"{group} {' '.join([f.name for f in fields])}" for group, fields in not_set_fields.items()]
                raise ValueError(f"Required arguments '{' '.join(s)}' were not set.")
        return r


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


def parse_list_arg(val, formatter, splitter=','):
    if len(val) == 0:
        return []

    if splitter not in val:
        # only one argument, may also have no brackets at all
        if val[0] == '[' and val[-1] == ']':
            val = val[1:-1]
        if len(val) == 0:
            return []
        return [formatter(val)]

    if val[0] != '[':
        raise ValueError("List argument must start with '[' but got '{}'".format(val))
    elif val[-1] != ']':
        raise ValueError("List argument must end with ']' but got '{}'".format(val))

    return [formatter(s.strip()) for s in val[1:-1].split(splitter) if len(s.strip()) > 0]


def parse_list_list_arg(val, formatter):
    return [parse_list_arg(sub, formatter) for sub in parse_list_arg(val.replace('],[', ']|['), str, '|')]


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


def is_optional_field(field):
    return (hasattr(field.type, "__args__")
            and len(field.type.__args__) == 2
            and field.type.__args__[-1] is type(None)
            )


def is_list_field(field):
    if isinstance(field, type):
        return
    if hasattr(field, "__args__"):
        ftype = field
    else:
        ftype = field.type
    return (hasattr(ftype, "__args__")
            and len(ftype.__args__) == 1
            and ftype._name == 'List'
            )


def make_store_dataclass_action(data_cls: Any, required_fields: List, exclude_field_names: List[str] = None):
    if exclude_field_names is None:
        exclude_field_names = []
    safe_separator = ":"
    all_fields = fields_to_dict(data_cls, safe_separator=safe_separator)
    for name in exclude_field_names:
        if name in all_fields:
            del all_fields[name]
    required_fields.extend([field for name, field in all_fields.items() if field.metadata.get('required', False)])

    class DataClassAction(Action):
        def __call__(self, parser, args, values, option_string=None):
            source_params = getattr(args, self.dest)
            for kv in values:
                try:
                    if len(kv.split('=')) == 2:
                        key, val = kv.split("=")
                    else:
                        raise ValueError(f"Could not parse '{kv}' must by KEY=VALUE")

                    # get parameter of data_cls
                    try:
                        name, field = None, None
                        for name, f in all_fields.items():
                            if name.replace(safe_separator, f.metadata.get('arg_mode_separator', '.')) == key:
                                field = f
                                break
                        if field is None:
                            raise KeyError()
                    except KeyError:
                        arguments = [name.replace(safe_separator, field.metadata.get('arg_mode_separator', '.')) for name, field in all_fields.items()]
                        closest = None
                        for arg in arguments:
                            if key in arg:
                                closest = arg

                        if not closest and len(arguments) > 0:
                            distances = {a: Levenshtein.distance(a, key) for a in arguments}
                            closest = sorted(arguments, key=lambda a: distances[a])[0]

                        str_args = argument_list_to_str(arguments)
                        raise AttributeError(f'Invalid argument {key}. Did you mean "{closest}={val}"? Available arguments {str_args}')

                    # get the correct params and set the non prefixed key (snake mode)
                    params = source_params
                    split_name = name.split(safe_separator)
                    key = split_name[-1]
                    for sub in split_name[:-1]:
                        params = getattr(params, sub)

                    def cast(v, cast_fn, f):
                        if is_optional_field(f) and v.lower() in {'none', 'null'}:
                            return None
                        return cast_fn(v)

                    def cast_list(v, cast_type_fn, cast_list_fn, f):
                        if is_optional_field(f) and v.lower() in {'none', 'null'}:
                            return None
                        return cast_list_fn(v, cast_type_fn)

                    # Single
                    if field.type == Optional[str] or field.type == str:
                        setattr(params, key, cast(val, str, field))
                    elif field.type == Optional[int] or field.type == int:
                        setattr(params, key, cast(val, int, field))
                    elif field.type == Optional[float] or field.type == float:
                        setattr(params, key, cast(val, float, field))
                    elif field.type == Optional[bool] or field.type == bool:
                        setattr(params, key, cast(val, str2bool, field))
                    elif field.type == Optional[Resource] or field.type == Resource:
                        setattr(params, key, cast(val, Resource, field))

                    # Lists
                    elif field.type == Optional[List[str]] or field.type == List[str]:
                        setattr(params, key, cast_list(val, str, parse_list_arg, field))
                    elif field.type == Optional[List[int]] or field.type == List[int]:
                        setattr(params, key, cast_list(val, int, parse_list_arg, field))
                    elif field.type == Optional[List[float]] or field.type == List[float]:
                        setattr(params, key, cast_list(val, float, parse_list_arg, field))

                    # Lists of Lists
                    elif field.type == List[List[float]]:
                        setattr(params, key, cast_list(val, float, parse_list_list_arg, field))

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

                    if field in required_fields:
                        required_fields.remove(field)

                except Exception as e:
                    logger.error(f"During the parsing of field {key}={val} an error occurred.")
                    raise e

    return DataClassAction


def field_is_dataclass(field):
    return hasattr(field.type, "__dataclass_fields__") or \
           (hasattr(field.type, '__origin__') and field.type.__origin__ == Union and hasattr(field.type.__args__[0], '__dataclass_fields__'))


def extract_dataclass_from_field(field):
    if hasattr(field.type, "__dataclass_fields__"):
        return field.type
    else:
        return field.type.__args__[0]


def fields_to_dict(data_cls, include_snake=True, d=None, prefix: str = '', safe_separator=":") -> Dict[str, Any]:
    d = d if d is not None else {}
    for name, f in data_cls.__dataclass_fields__.items():
        if name.endswith('_') or ('arg_mode' in f.metadata and f.metadata['arg_mode'] == 'ignore'):
            continue
        if field_is_dataclass(f):
            if include_snake and 'arg_mode' in f.metadata and f.metadata['arg_mode'] == 'snake':
                fields_to_dict(extract_dataclass_from_field(f), include_snake, d, prefix + name + safe_separator)
            continue
        d[prefix + name] = f

    return d


def convert_value(p):
    if isinstance(p, Resource):
        return p.initial_path
    elif issubclass(p.__class__, enum.Enum):
        return p.value
    return str(p)

def get_default(p, name, safe_separator=":"):
    for sub in name.split(safe_separator):
        p = getattr(p, sub)
    if isinstance(p, list):
        return '[' + ','.join([convert_value(x) for x in p]) + ']'
    return convert_value(p)

def generate_help_for_field(field, name, default, safe_separator=":"):
    enum_class = enum_class_from_field(field)
    msg = f"{name.replace(safe_separator, field.metadata.get('arg_mode_separator', '.'))}={default}"
    if enum_class:
        vs = [str(e.value) for e in enum_class]
        msg += " ∈ {" + ', '.join(vs) + "}"
    if field.metadata:
        if 'help' in field.metadata:
            msg += f"\n    {field.metadata['help']}"
    return msg

def generate_help(data_cls: Any):
    default = data_cls()
    safe_separator = ":"
    all_fields = fields_to_dict(data_cls, safe_separator=safe_separator)
    return "\n".join([generate_help_for_field(all_fields[name], name, get_default(default, name)) for name in sorted(all_fields.keys())])


def argument_list_to_str(arguments):
    return "[" + ', '.join(f"{name}" for name in arguments) + "]"


def add_args_group(parser: TFAIPArgumentParser, group: str, params_cls: Any, default=None, exclude_field_names=None):
    if exclude_field_names is None:
        exclude_field_names = []
    assert(isinstance(parser, TFAIPArgumentParser))
    default = default if default else params_cls()
    params_cls = default.__class__
    parser.add_argument("--" + group,
                        action=make_store_dataclass_action(params_cls, parser.get_required_fields("--" + group),
                                                           exclude_field_names=exclude_field_names),
                        default=default,
                        nargs='*',
                        metavar="KEY=VAL",
                        help=generate_help(params_cls))

    for name, field in params_cls.__dataclass_fields__.items():
        if name in exclude_field_names:
            continue
        if field_is_dataclass(field) and not name.endswith("_"):
            if 'arg_mode' not in field.metadata or field.metadata['arg_mode'] == 'flat':
                add_args_group(parser, group=name, params_cls=extract_dataclass_from_field(field), default=getattr(default, name))
            elif field.metadata['arg_mode'] == 'snake':
                pass
            elif field.metadata['arg_mode'] == 'ignore':
                pass
            else:
                raise ValueError(f"Unknown arg_mode {field.metadata['arg_mode']}")
