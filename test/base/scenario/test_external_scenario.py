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
import unittest
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
tfaip_dir = os.path.abspath(os.path.join(this_dir, '..', '..', '..', 'tfaip'))


class TestExternalScenario(unittest.TestCase):
    def test_find_scenarios_via_env_var(self):
        os.environ['TFAIP_SCENARIOS'] = tfaip_dir
        from tfaip.scenario import scenarios
        all_scenarios = scenarios()
        print(all_scenarios)

        original_scenarios = []
        external_scenarios = []
        for scenario in scenarios():
            if scenario.name.startswith('scenario'):
                external_scenarios.append(scenario)
            else:
                original_scenarios.append(scenario)

        self.assertEqual(len(original_scenarios), len(external_scenarios))
        for a, b in zip(original_scenarios, external_scenarios):
            self.assertEqual('scenario.' + a.name, b.name)
            self.assertEqual(a.path, b.path)
            self.assertEqual(a.scenario, b.scenario)
