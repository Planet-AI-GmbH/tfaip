from argparse import ArgumentParser
from tfaip.base.trainer import Trainer
import logging

from tfaip.util.logging import setup_log

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()

    parser.add_argument('checkpoint_dir', type=str, help='path to the checkpoint dir to resume from')

    args = parser.parse_args()
    setup_log(args.checkpoint_dir, append=True)

    logger.info("=================================================================")
    logger.info(f"RESUMING TRAINING from {args.checkpoint_dir}")
    logger.info("=================================================================")

    trainer = Trainer.restore_trainer(args.checkpoint_dir)
    trainer.train()


if __name__ == '__main__':
    main()
