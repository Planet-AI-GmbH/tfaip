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
# -*- coding: utf-8 -*-


import codecs
import os
import sys
import traceback

unknown_word = 'UNK'


def create_sm_txt(path, name=""):
    pass


def get_sm(path):
    """
    Creates and returns a StringMap. The mapping is loaded from the file given by `path`.

    Args:
        path: path to text file, should contain the mappings like 'character=key'.

    Returns:
        StringMap with loaded mapping.
    """
    sm = StringMapper()
    try:
        if path.endswith('tsv'):
            sm.load_mapping_from_freq(path)
        elif path.endswith('txt'):
            sm.load_mapping_from_txt(path)
        else:
            sm.load_mapping_from_sm(path)
    except Exception as inst:
        print('Original Exception: ', inst)
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        raise RuntimeError("Can't load StringMap! Check if this is a valid path: ", os.path.abspath(path))

    # TODO (@Jochen) make configurable, maybe not a good default
    for channel in range(sm.size()):
        if channel in sm.word_to_id_map:
            i_tag = sm.get_value(channel).replace("B-", "I-")
            assert sm.has_channel(i_tag), \
                f'Tag-Map-Error: channel: {channel}->{sm.get_value(channel)} has no corresponding I-Tag!'
    return sm


class StringMapper:
    def __init__(self):
        self.word_to_id_map = {}
        self.id_to_word_map = {}
        self.loaded = False

        self.unknown_id = -1
        # self.unknown_id = 0
        self._freq_map = {}

    def get_oov_id(self):
        return self.unknown_id

    def has_channel(self, string):
        return string in self.word_to_id_map

    def get_channel(self, string):
        if self.has_channel(string):
            return self.word_to_id_map[string]
        else:
            # print("unknown word: ", string, " returning ", self.unknown_id)
            return self.unknown_id

    def get_value(self, channel):
        if channel < self.size():
            return self.id_to_word_map[channel]
        else:
            return unknown_word

    def size(self):
        return len(self.id_to_word_map)

    def add(self, string, channel=None):
        if channel == None:
            channel = len(self.word_to_id_map)
            # print("len = " + str(index))
        self.word_to_id_map[string] = channel
        if channel not in self.id_to_word_map:
            self.id_to_word_map[channel] = string
        # print("pos = " + str(index))
        # self.unknown_id = len(self.id_to_word_map)
        return channel

    def get_mapping(self, file_path):
        with codecs.open(file_path, 'r', encoding='utf-8') as cm_file:
            raw = cm_file.readlines()
            return raw

    def get_freq_from_id(self, id):
        return self._freq_map[id]

    def get_freq_from_word(self, word):
        return self._freq_map[self.word_to_id_map[word]]

    def load_mapping_from_sm(self, file_path):
        if self.loaded:
            raise RuntimeError("map already loaded")
        with codecs.open(file_path, 'r', encoding='utf-8') as cm_file:
            raw = cm_file.readlines()
            for line in raw:
                if line[-1] == '\n':
                    line = line[:-1]
                split = line.rsplit('=', 1)
                key = split[0]
                index = int(split[1]) - 1
                if index < 0:
                    raise Exception(f"index of key {key} is {index + 1} but has to be at least 1")
                # specific values which are escaped by '\': delete '\'
                if key[0] == '\\':
                    key = key[1:]
                if key == unknown_word:
                    self.unknown_id = id
                # print("'" + key + "' ==> " + str(index))
                self.add(key, index)
        self.loaded = True
        # self.unknown_id = len(self.dictBwd)
        if self.unknown_id < 0:
            self.unknown_id = len(self.id_to_word_map)
            self.add(unknown_word, self.unknown_id)

    def load_mapping_from_freq(self, file_path):
        if self.loaded:
            raise RuntimeError("map already loaded")
        with codecs.open(file_path, 'r', encoding='utf-8') as cm_file:
            raw = cm_file.readlines()
            id = 0
            for line in raw:
                if line[-1] == '\n':
                    line = line[:-1]
                split = line.rsplit('\t', 1)
                key = split[0]
                if key == unknown_word:
                    self.unknown_id = id
                freq = float(split[1])

                # specific values which are escaped by '\': delete '\'
                if key[0] == '\\':
                    key = key[1:]
                # print("'" + key + "' ==> " + str(index))
                self.add(key, id)
                self._freq_map[id] = freq
                id += 1
        self.loaded = True
        # self.unknown_id = len(self.dictBwd)
        if self.unknown_id < 0:
            self.unknown_id = len(self.id_to_word_map)
            self.add(unknown_word, self.unknown_id)
        # print("unknown id: ", self.unknown_id)

    def load_mapping_from_txt(self, file_path):
        if self.loaded:
            raise RuntimeError("map already loaded")
        with codecs.open(file_path, 'r', encoding='utf-8') as cm_file:
            raw = cm_file.readlines()
            id = 0
            for line in raw:
                if line[-1] == '\n':
                    line = line[:-1]
                key = line
                # specific values which are escaped by '\': delete '\'
                if key[0] == '\\':
                    key = key[1:]
                # print("'" + key + "' ==> " + str(index))
                self.add(key, id)
                if key == unknown_word:
                    self.unknown_id = id
                id += 1
        self.loaded = True
        # self.unknown_id = len(self.dictBwd)
        if self.unknown_id < 0:
            self.unknown_id = len(self.id_to_word_map)
            self.add(unknown_word, self.unknown_id)

    def save_mapping(self, file_path):
        with codecs.open(file_path, 'w', encoding='utf-8') as cm_file:
            for key, value in self.word_to_id_map.iteritems():
                if key == '\\' or key == '=':
                    cm_file.write('\\')
                cm_file.write(key)
                cm_file.write('=')
                cm_file.write(str(value + 1))
                cm_file.write('\n')
                # file.write('NaC')
                # file.write('=')
                # file.write(str(len(self.dictBwd)))
        self.loaded = True

    def print2(self):
        # for key in self.dictBwd.keys():
        # print(key)
        # print(str(key)+" => "+(self.dictBwd.get(key)))
        print(self.word_to_id_map)
        print(self.id_to_word_map)


if __name__ == '__main__':
    os.chdir("..")
    # char_map = CharacterMapper()
    # char_map.load_mapping('./private/data/meganet/char_map.txt')
    # char_map.print()
    # print("value = " + str(char_map.get_channel('-')))
    # print("value of " + (u'\u2014') + " = " + str(char_map.get_channel(u'\u2014')))
    # print("value of " + (u'\xe4') + " = " + str(char_map.get_channel(u'\xe4')))
    # print("#############################################")
    # char_map = get_cm_lp()
    # char_map.print()
    # print("char_map = " + str(char_map.dictFwd))
    # print(char_map.get_channels(u'012-Ö4'))

    # barlach=get_cm_barlach()
    # print(barlach.size())

    create_sm_txt("./private/data/bentham/train_and_val.txt", "bentham")
    # char_map = CharacterMapper()
    # char_map.load_mapping("./private/data/cm_esposalles.txt")
    # char_map.print()
