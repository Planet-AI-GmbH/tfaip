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
from argparse import ArgumentParser
import sys

from tfaip.scripts.xlsxexperimenter.run_xlsx_experimenter import XLSXExperimenter


def main():
    parser = ArgumentParser()

    parser.add_argument("--xlsx", required=True)
    parser.add_argument("--no_use_tsp", action="store_true", default=False)
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=str,
        required=False,
        help="The gpus to use. For multiple runs on the same gpu use e.g. --gpus 3a 3b 3b",
    )
    parser.add_argument(
        "--cpus",
        nargs="+",
        type=str,
        required=False,
        help="The cpu devices. e.g. --cpus 0 1 2 3 4 to schedule on 5 cpus",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--python", default=sys.executable)

    args = parser.parse_args()

    if not args.no_use_tsp and not args.gpus and not args.cpus:
        raise ValueError("No devices (gpu or cpu) found. Disable task spooler (--use_no_tsp) or use --gpus --cpus")
    if args.gpus and args.cpus:
        raise ValueError("Do not mix gpu and cpu calls.")

    exp = XLSXExperimenter(args.xlsx, args.gpus, args.cpus, args.dry_run, args.python, not args.no_use_tsp)
    exp.run()


if __name__ == "__main__":
    main()
