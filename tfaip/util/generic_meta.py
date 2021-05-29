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
"""Definition of Meta-Classes used to determine Generics of a class"""
from abc import ABCMeta
from copy import copy
from typing import TypeVar, List, Any, Dict


class CollectGenericTypes(ABCMeta):
    """
    Meta-Class to automatically track and store all Generics and its actuall (final) type of a class.

    This class will add __generic_types__ and __generic_typevars__ attributes to a class that store:
    - __generic_types__: A dictionary tracking the type of a given typevar (tracked by its name)
    - __generic_typevars__: A dictionary tracking all typevars of this Generic class (tracked by its name)

    Tracking is based on the origin attribute of a TypeVar which is why only one Generic per BaseClass is allowed.
    An exception to this rule is if multiple TypeVars of the same base are overwritten simultanously. Matchis is then
    based on the order.
    """

    def __subclasscheck__(cls, subclass):
        # Custom subclass check for correct 'issubclass' support
        # I do not yet know why this is required...
        return cls == subclass or any(cls == sc for sc in subclass.__mro__)

    def __new__(mcs, *args, **kwargs):
        # Automatically discover generic types of class by iterating base classes and current __orig_bases__
        c = type.__new__(mcs, *args, **kwargs)
        # create fields
        c.__generic_types__: Dict[Any] = {}
        c.__generic_typevars__ = {}

        # copy from all super classes
        for super_cls in args[1]:
            if hasattr(super_cls, "__generic_types__"):
                c.__generic_types__.update(super_cls.__generic_types__)
            if hasattr(super_cls, "__generic_typevars__"):
                c.__generic_typevars__.update(super_cls.__generic_typevars__)

        # loop though all super classes of this class and detect if this class comprises replacements for a Generic
        if hasattr(c, "__orig_bases__"):
            non_matched = []
            for ob in c.__orig_bases__:
                if not hasattr(ob, "__args__"):
                    continue
                for arg in ob.__args__:
                    # Loop though through the args, this is where the actual Types are written (the square brackets)
                    if isinstance(arg, TypeVar):
                        c.__generic_typevars__[arg.__name__] = arg

                    matching_values = []
                    for t in c.__generic_typevars__.values():
                        # Find all matching types (usually there should only be one)
                        if isinstance(arg, TypeVar):
                            if arg == t:
                                matching_values.append((t.__name__, arg.__bound__))
                        elif issubclass(arg, t.__bound__):
                            matching_values.append((t.__name__, arg))

                    if len(matching_values) == 1:
                        # Update the type
                        c.__generic_types__[matching_values[0][0]] = matching_values[0][1]
                    elif len(matching_values) > 1:
                        # Collect all that match multiple types, they will be assigned in order
                        non_matched.append(matching_values)
                    else:
                        raise TypeError(f"Could not find {arg} in {c.__generic_typevars__.values()}")

            if non_matched:
                # check if ambiguous matches are all identical and if its the same length as the matches, then
                # match them. Thus take the type of the matching values, by the order of non_matched for the target
                if all(len(values) == len(non_matched) for values in non_matched):
                    for i, matching_values in enumerate(non_matched):
                        c.__generic_types__[non_matched[0][i][0]] = matching_values[0][1]
                else:
                    raise TypeError("Could not determining corresponding TypeVar.")
        return c


class ReplaceDefaultDataClassFieldsMeta(CollectGenericTypes):
    """
    This meta class will not only track the Generics of a dataclass but also replace Generic default values.
    This enables usage of, e.g.:

    dc_field: TSub = field(default_factory=TSub)

    by replacing the default factory with the actual type of the Generic
    """

    def __new__(mcs, *args, field_names: List[str], **kwargs):
        # Set default types for TrainerPipelineParams
        c = super().__new__(mcs, *args, **kwargs)

        # store mapping from field name to typevar id by obtaining the type of the field
        c.__generic_field_name_to_typevar__ = {}

        # collect all from super classes
        for super_cls in args[1]:
            if hasattr(super_cls, "__generic_field_name_to_typevar__"):
                c.__generic_field_name_to_typevar__.update(super_cls.__generic_field_name_to_typevar__)

        if hasattr(c, "__dataclass_fields__"):
            # track non yet tracked fields
            for name, field in c.__dataclass_fields__.items():
                if name in field_names:
                    if name not in c.__generic_field_name_to_typevar__:
                        c.__generic_field_name_to_typevar__[name] = field.type.__name__

                    # replace factory
                    field.default_factory = c.__generic_types__[c.__generic_field_name_to_typevar__[name]]

        return c

    @staticmethod
    def generic_types(ob, i):
        arg = ob.__args__[i]
        if isinstance(arg, TypeVar):
            return arg.__bound__  # default
        return arg
