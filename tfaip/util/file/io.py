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
import errno
import os
import shutil


def cp_copy(src, dest, exist_ok=True):
    """create path copy"""
    if not os.path.isdir(os.path.dirname(dest)):
        try:
            os.makedirs(os.path.dirname(dest))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    if not exist_ok:
        if os.path.isfile(dest) or os.path.isdir(dest):
            raise IOError("File already exist! {}".format(dest))
    shutil.copy(src, dest)


def file_path_with_mkdirs(path_with_filen_name):
    """will create all dirs to save the file to the given path"""
    if not os.path.isdir(os.path.dirname(path_with_filen_name)):
        try:
            os.makedirs(os.path.dirname(path_with_filen_name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return path_with_filen_name


class IOContext(object):
    def get_io_filepath(self, filepath: str):
        raise NotImplementedError()


class DefaultIOContext(IOContext):
    def get_io_filepath(self, filepath: str):
        return os.path.abspath(filepath)


#
#
# class RootDirPathIOContext(IOContext):
#     def __init__(self, fs_root_dirpath=None):
#         self._fs_root_dirpath = fs_root_dirpath
#
#     def get_io_filepath(self, filepath):
#         return os.path.join(self._fs_root_dirpath, filepath) if self._fs_root_dirpath is not None else filepath
#
#
# class RelativeToListFileIOContext(RootDirPathIOContext):
#     def __init__(self, list_filepath=None):
#         super(RelativeToListFileIOContext, self).__init__(
#             fs_root_dirpath=os.path.dirname(os.path.abspath(list_filepath)))
#
#
# class CITnetRaceTrackIOContext(RootDirPathIOContext):
#     def __init__(self, list_filepath=None):
#         super(CITnetRaceTrackIOContext, self).__init__(
#             fs_root_dirpath=os.path.dirname(os.path.dirname(os.path.abspath(list_filepath))))
#
#

class ReplaceIOContext(IOContext):
    def __init__(self, old: str, new: str):
        self._old = old
        self._new = new

    def get_io_filepath(self, filepath):
        return filepath.replace(self._old, self._new, 1)
