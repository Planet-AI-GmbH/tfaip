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
import os
import platform
import pwd


def get_username():
    return pwd.getpwuid(os.getuid())[0]


def get_home_dir():
    system = platform.system()
    if system == "Linux":
        return "/home/" + get_username()
    elif system == "Darwin":
        return "/Users/" + get_username()
    else:
        raise NotImplementedError('get_home_dir not yet implemented for ' + system)


if __name__ == '__main__':
    # img_fn = "vol/my_path/img.jpg"
    # old_prefix = "vol/"
    # new_prefix = "truth/"
    # tfg = PrefixReplacingExtAppendingXMLTruthFilenameGenerator(old_prefix, new_prefix)
    # print img_fn + " -> " + tfg.generate_truth_filename(img_fn)
    print(get_home_dir())
