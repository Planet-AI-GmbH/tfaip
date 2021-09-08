# (c) 2021, PLANET artificial intelligence GmbH
#
# @contact legal@planet-ai.de
# ==============================================================================
import argparse
import re


def update_source(filename, oldcopyright, copyright, dry_run=False, copy_rights_to_erase=None) -> bool:
    utfstr = chr(0xEF) + chr(0xBB) + chr(0xBF)
    with open(filename, "r") as f:
        fdata = f.read()
    is_utf = False
    if fdata.startswith(utfstr):
        is_utf = True
        fdata = fdata[3:]

    # erase any old copyright
    if copy_rights_to_erase:
        copy_rights_to_erase = [c for c in copy_rights_to_erase if c != oldcopyright and c != copyright]
        for c in copy_rights_to_erase:
            if fdata.startswith(c):
                fdata = fdata[len(c) :]

    if oldcopyright is not None:
        if fdata.startswith(oldcopyright):
            if oldcopyright == copyright:
                return True
            fdata = fdata[len(oldcopyright) :]
    if not fdata.startswith(copyright):
        print("Updating Copyright of " + filename)
        fdata = copyright + fdata
        if dry_run:
            return False

        with open(filename, "w") as f:
            if is_utf:
                f.write(utfstr + fdata)
            else:
                f.write(fdata)

        return False
    return True


def process_file(file: str, setup, old_copyrights) -> bool:
    copyright = setup["copyright"]
    return update_source(file, copyright, copyright, dry_run=False, copy_rights_to_erase=old_copyrights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", default=[])
    args = parser.parse_args()

    setup = [
        # Open source python files
        {
            "filter": [
                re.compile(r"tfaip/.*\.py"),
                re.compile(r"test/.*\.py"),
                re.compile(r"examples/.*\.py"),
                re.compile(r"setup\.py"),
                re.compile(r"docs/.*\.py"),
            ],
            "copyright": "# Copyright 2021 The tfaip authors. All Rights Reserved.\n"
            "#\n"
            "# This file is part of tfaip.\n"
            "#\n"
            "# tfaip is free software: you can redistribute it and/or modify\n"
            "# it under the terms of the GNU General Public License as published by the\n"
            "# Free Software Foundation, either version 3 of the License, or (at your\n"
            "# option) any later version.\n"
            "#\n"
            "# tfaip is distributed in the hope that it will be useful, but\n"
            "# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY\n"
            "# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for\n"
            "# more details.\n"
            "#\n"
            "# You should have received a copy of the GNU General Public License along with\n"
            "# tfaip. If not, see http://www.gnu.org/licenses/.\n"
            "# ==============================================================================\n",
        },
        # any other python file
        {
            "filter": [re.compile(r".*\.py")],
            "copyright": "# (c) 2021, PLANET artificial intelligence GmbH\n"
            "#\n"
            "# @contact legal@planet-ai.de\n"
            "# ==============================================================================\n",
        },
    ]

    all_copyrights_to_erase = [s["copyright"] for s in setup]

    def apply_on_file(file):
        for s in setup:
            if any(r.fullmatch(file) for r in s["filter"]):
                return process_file(file, s, all_copyrights_to_erase)
        return True

    results = []
    for file in args.files:
        results.append(apply_on_file(file))

    print(f"{len(results) - sum(results)} files reformatted, {sum(results)} files left unchanged.")

    if len(results) > 0 and not all(results):
        exit(1)
