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
import pkgutil
import os
from typing import List, TYPE_CHECKING, Type
import logging
from dataclasses import dataclass
import importlib


if TYPE_CHECKING:
    from tfaip.base.scenario import ScenarioBase


@dataclass
class ScenarioDefinition:
    path: str
    name: str
    scenario: Type['ScenarioBase']


_scenarios: List[ScenarioDefinition] = []
logger = logging.getLogger(__name__)
this_dir = os.path.dirname(os.path.realpath(__file__))


def register_scenario(defintion: ScenarioDefinition):
    for scen in _scenarios:
        if scen.name == defintion.name and scen.scenario != defintion.scenario:
            raise KeyError(f"Scenario with name {scen.name} already registered at {scen.path} while attempting to add scenario at {defintion.path}")

    _scenarios.append(defintion)


def load_all_scenarios(dir_name, root_package='', root_dir=None):
    from tfaip.base.scenario.scenariobase import list_scenarios_in_module
    root_dir = root_dir if root_dir is not None else dir_name
    logger.debug(f"Searching for scenarios in {dir_name}")
    for importer, package_name, _ in pkgutil.iter_modules([dir_name]):
        if package_name == 'scenario':
            module = importer.find_module(package_name)
            if module:
                logger.debug(f"Found scenario file at {dir_name}. Attempting to load Scenario classes")
                imported = importlib.import_module('.scenario', root_package)
                all_scenario_cls = list_scenarios_in_module(imported)
                for name, scenario_cls in all_scenario_cls:
                    logger.info(f"Added scenario {dir_name}.{scenario_cls.__name__}")
                    register_scenario(ScenarioDefinition(
                        dir_name,
                        os.path.relpath(dir_name, root_dir).replace("/", '.'),
                        scenario_cls
                    ))

        load_all_scenarios(os.path.join(dir_name, package_name), '.'.join([root_package, package_name]), root_dir)


def scenarios():
    if not _scenarios:
        load_all_scenarios(this_dir, 'tfaip.scenario')

        # load external scenarios
        external_dirs = os.environ.get('TFAIP_SCENARIOS', '').split(';')
        for external_dir in external_dirs:
            if len(external_dir) == 0:
                continue
            logger.info(f"Loading external scenarios from {external_dir}")
            load_all_scenarios(external_dir, '', root_dir=external_dir)

    return _scenarios
