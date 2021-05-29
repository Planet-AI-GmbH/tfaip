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
import logging
import os
from argparse import ArgumentParser

from tensorflow import keras

from tfaip.imports import Trainer, WarmStartParams, WarmStarter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main():
    parser = ArgumentParser()

    parser.add_argument("output_dir", type=str, help="path to the checkpoint dir to resume from")

    args = parser.parse_args()

    with open(os.path.join(args.output_dir, "trainer_params.json")) as f:
        d = json.load(f)
        d.get("device", {})["gpus"] = []

    trainer = Trainer.restore_trainer(d)
    scenario = trainer.scenario
    data = scenario.create_data()
    model = scenario.create_model()

    inputs = data.create_input_layers()
    outputs = model.build(inputs)
    keras_model = keras.models.Model(inputs=inputs, outputs=outputs)
    scenario._keras_predict_model = keras_model
    scenario._export_graphs = model.export_graphs(inputs, outputs, data.create_target_as_input_layers())

    def store(path):
        print(f"Converting {path}")
        if os.path.exists(path):
            warmstart_params = WarmStartParams(
                model=os.path.join(path, "serve"),
                rename_targets=[f"BLSTM_{i}->BLSTM{i + 1}" for i in range(3)],
            )
            warmstarter = WarmStarter(warmstart_params)
            warmstarter.warmstart(scenario.keras_predict_model)
            scenario.export(path, export_resources=False)
            print(f"{path} successfully written.")
        else:
            print(f"{path} not found. Skipping")

    for p in ["best", "export"]:
        store(os.path.join(args.output_dir, p))


if __name__ == "__main__":
    main()
