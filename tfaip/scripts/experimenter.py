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