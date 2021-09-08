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
from typing import List

from tfaip.lav.callbacks.lav_callback import LAVCallback

from tfaip.lav.lav import LAV


class ListFileLAVCallback(LAVCallback):
    def on_lav_end(self, result):
        ...

    def _on_lav_end(self, result):
        if self.lav.params.store_results:
            dump_dict = {
                "metrics": {
                    k: v for k, v in result.items() if (not isinstance(v, bytes) and not type(v).__module__ == "numpy")
                },
                "lav_params": self.lav.params.to_dict(),
                "data_params": self.data.params.to_dict(),
                "model_params": self.model.params.to_dict(),
            }
            json_fn = os.path.join(
                self.lav.params.model_path,
                f"lav_results_{'_'.join([os.path.basename(l) for l in self.current_data_generator_params.lists])}.json",
            )

            with open(json_fn, "w") as json_fp:
                json.dump(dump_dict, json_fp, indent=2)


class ListFileLAV(LAV):
    def _custom_callbacks(self) -> List[LAVCallback]:
        return [ListFileLAVCallback()]
