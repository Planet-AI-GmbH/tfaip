import json
from argparse import ArgumentParser
from tfaip.base.trainer import Trainer
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main():
    parser = ArgumentParser()

    parser.add_argument('checkpoint_dir', type=str, help='path to the checkpoint dir to resume from')

    args = parser.parse_args()

    with open(os.path.join(args.checkpoint_dir, 'trainer_params.json')) as f:
        d = json.load(f)
        d['device_params']['gpus'] = []

    trainer = Trainer.restore_trainer(d)
    scenario = trainer.scenario
    params = trainer.params
    scenario.setup_training('adam', params.skip_model_load_test,
                            run_eagerly=params.force_eager,
                            no_train_scope=params.no_train_scope)

    def store(path):
        print(f"Converting {path}")
        if os.path.exists(path):
            scenario.keras_predict_model.load_weights(os.path.join(path, 'serve', 'variables', 'variables'))
            scenario.export(path, export_resources=False)
            print(f"{path} successfully written.")
        else:
            print(f"{path} not found. Skipping")

    for p in ['best', 'export']:
        store(os.path.join(args.checkpoint_dir, p))


if __name__ == '__main__':
    main()
