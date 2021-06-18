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

from tfaip.imports import Trainer, WarmStartParams, WarmStarter
from tfaip_addons.seq2seq.decoder import BeamSearchDecoderParams, BasicDecoderParams

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main():
    parser = ArgumentParser()

    parser.add_argument("output_dir", type=str, help="path to the checkpoint dir to resume from")

    args = parser.parse_args()

    # reset gpu usage
    with open(os.path.join(args.output_dir, "trainer_params.json")) as f:
        d = json.load(f)
        d.get("device", {})["gpus"] = []

    trainer_params, scenario = Trainer.parse_trainer_params(d)
    # trainer_params.scenario.model.inference_decoder = BeamSearchDecoderParams(beam_width=5)
    # trainer_params.scenario.model.ctc_decoder_weight = 0
    # trainer_params.scenario.model.length_bonus = 0.6
    trainer_params.scenario.model.inference_decoder = BasicDecoderParams()
    trainer = scenario.create_trainer(trainer_params, restore=True)
    scenario = trainer.scenario

    # setup training and prediction graph with dummy settings
    scenario.params.print_eval_limit = 0
    scenario.setup_training("adam", skip_model_load_test=True)

    def store(path, to_path):
        print(f"Converting from {path} to {to_path}")
        if os.path.exists(path):
            # warmstart_params = WarmStartParams(
            #     model=os.path.join(path, 'serve'),
            #     rename_targets=[f"BLSTM_{i}->BLSTM{i + 1}" for i in range(3)],
            # )
            # warmstarter = WarmStarter(warmstart_params)
            # warmstarter.warmstart(scenario.keras_predict_model)
            scenario.keras_predict_model.load_weights(os.path.join(path, "serve", "variables", "variables"))
            scenario.export(to_path, trainer_params=trainer_params, export_resources=False)
            print(f"{to_path} successfully written.")
        else:
            print(f"{path} not found. Skipping")

    for p, to_p in [("best", "best_path")]:
        store(os.path.join(args.output_dir, p), os.path.join(args.output_dir, to_p))


if __name__ == "__main__":
    main()
