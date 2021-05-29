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
import json
import os

from tfaip.lav.lav import LAV
from tfaip.scenario.listfile.params import ListsFileGeneratorParams


class ListFileLAV(LAV):
    def _on_lav_end(self, data_generator_params: ListsFileGeneratorParams, result):
        if self._params.store_results:
            dump_dict = {
                "metrics": {
                    k: v for k, v in result.items() if (not isinstance(v, bytes) and not type(v).__module__ == "numpy")
                },
                "lav_params": self._params.to_dict(),
                "data_params": self._data.params.to_dict(),
                "model_params": self._model.params.to_dict(),
            }
            json_fn = os.path.join(
                self._params.model_path,
                f"lav_results_{'_'.join([os.path.basename(l) for l in data_generator_params.lists])}.json",
            )

            with open(json_fn, "w") as json_fp:
                json.dump(dump_dict, json_fp, indent=2)
