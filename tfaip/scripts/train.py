import logging
from tfaip.util.logging import setup_log
from argparse import ArgumentParser, RawTextHelpFormatter
from tfaip.scenario import scenarios
from tfaip.util.argument_parser import add_args_group


logger = logging.getLogger(__name__)


def run():
    main(parse_args())


def main(args):
    # parse the arguments
    scenario_meta = next(s.scenario for s in scenarios() if s.name == args.scenario)

    trainer_params = args.trainer_params
    if trainer_params.checkpoint_dir:
        setup_log(trainer_params.checkpoint_dir, append=False)

    logger.info("trainer_params=" + trainer_params.to_json(indent=2))

    # create the trainer and run it
    trainer = scenario_meta.create_trainer(trainer_params)
    trainer.train()


def parse_args(args=None):
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)

    # setup scenarios as subparsers (all available)
    sub_parsers = parser.add_subparsers(dest='scenario', required=True)

    # loop over all available scenarios
    for scenario_def in scenarios():
        # add scenario parameters as sub parameters
        p = sub_parsers.add_parser(scenario_def.name, formatter_class=parser.formatter_class)
        default_trainer_params = scenario_def.scenario.trainer_cls().get_params_cls()()
        default_trainer_params.scenario_params = scenario_def.scenario.default_params()
        add_args_group(p, group='trainer_params', default=default_trainer_params, params_cls=scenario_def.scenario.trainer_cls().get_params_cls())

    return parser.parse_args(args)


if __name__ == '__main__':
    run()
