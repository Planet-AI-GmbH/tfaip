from argparse import ArgumentParser
import sys

from tfaip.scripts.xlsxexperimenter.run_xlsx_experimenter import XLSXExperimenter


def main():
    parser = ArgumentParser()

    parser.add_argument('--xlsx', required=True)
    parser.add_argument('--gpus', nargs='+', type=str, required=False, help="The gpus to use. For multiple runs on the same gpu use e.g. --gpus 3a 3b 3b")
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--no_use_ts', action='store_true', default=False)

    args = parser.parse_args()

    exp = XLSXExperimenter(args.xlsx, args.gpus, args.dry_run, args.python, not args.no_use_ts)
    exp.run()


if __name__ == '__main__':
    main()