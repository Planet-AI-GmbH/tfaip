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
__author__ = "Gundram Leifert"
__copyright__ = "Copyright 2020, Planet AI GmbH"
__credits__ = ["Gundram Leifert"]
__email__ = "gundram.leifert@planet-ai.de"

import ast
import json
import logging
from dataclasses import dataclass
from os import PathLike
from typing import Dict, Union, Any, Optional, List

from tfaip.util.file.stringmapper import StringMapper


def load_from_json(path: str):
    with open(path, 'r') as fp:
        struct = json.load(fp)
        return PaiHandler(struct)


@dataclass
class Region(object):
    classes: List[Union[int, str]]
    id: str
    # classes_names:List[str]
    bb: List[int]


def _default_struct():
    return {
        # "id": "",
        # "producer": "",
        # "srcImgPath": "",
        "pages": [
            {
                # "id": ""
            }]
    }


def _default_classification(label: Union[List[str], str], score: float = 1.0):
    labels = label if isinstance(label, list) else [label]
    results = []
    for l in labels:
        results.append({
            "label": l,
            "score": score,
            # "isValid": True
        })
    res = {}
    return {
        # "moduleId": "human_detector",
        # "moduleLabel": "full",
        # "threshold": 1.0,
        "results": results
    }


class PaiHandler(object):

    def __init__(self, struct: Optional[Dict[str, Any]] = None, category_mapper: Union[StringMapper, None] = None):
        self._struct = struct if struct is not None else _default_struct()
        self._category_mapper = category_mapper

    def save(self,
             path: Union[str, PathLike],
             indent: Union[int, None] = 4):
        with open(path, 'w') as fp:
            json.dump(self._struct, fp, indent=indent)

    def add_classification(self, label: str, score: float = 1.0):
        page = self._get_page_struct()
        if 'classifications' not in page:
            page['classifications'] = []
        page['classifications'].append(_default_classification(label, score))

    def add_property(self, key, value):
        if 'properties' not in self._struct:
            self._struct['properties'] = {}
        self._struct['properties'][key] = value

    def add_region(self, bb: List[int], clazz: Union[List[str], str], id: str = None, score: float = 1.0):
        page = self._get_page_struct()
        if 'regions' not in page:
            page['regions'] = []
        regions: List = page['regions']
        assert len(bb) == 4
        points = [
            [bb[0], bb[2]],
            [bb[1], bb[2]],
            [bb[1], bb[3]],
            [bb[0], bb[3]],
        ]
        classifications = []
        if not isinstance(clazz, list):
            clazz = [clazz]
        # for c in clazz:
        classifications.append(_default_classification(clazz, score))
        r = {
            'coordinates': ";".join([f"{b[0]},{b[1]}" for b in points]),
            'classifications': classifications,
        }
        if id is not None:
            r['id'] = id
        regions.append(r)

    def _get_page_struct(self):
        if len(self._struct['pages']) != 1:
            raise Exception(f"not implemented for pages != 1 ( = {len(self._struct['pages'])} )")
        return self._struct['pages'][0]

    def get_regions(self,
                    category_mapper: Union[StringMapper, None] = None,
                    allow_xfile: bool = True
                    ) -> List[Region]:
        content = []
        if category_mapper is None:
            category_mapper = self._category_mapper
        page = self._get_page_struct()
        if 'regions' in page:
            for region in page['regions']:
                _fill_regions(region, content, category_mapper, allow_xfile)
        return content

    def get_classifications(self,
                            category_mapper: Union[StringMapper, None] = None,
                            allow_xfile: bool = True
                            ) -> List[Dict[str, Any]]:
        """

        @rtype: object
        """
        if category_mapper is None:
            category_mapper = self._category_mapper
        if len(self._struct['pages']) != 1:
            raise Exception(f"not implemented for pages != 1 ( = {len(self._struct['pages'])} )")
        classifications = []
        for page in self._struct['pages']:
            classifications.extend(
                _get_classifications(page['classifications'], category_mapper))
        return classifications

    def get_text(self):
        pass


def _str_to_bb(str_from_json: str):
    if (str_from_json[0] == '['):
        reg_proc = ast.literal_eval(str_from_json.replace(';', ','))
        xs = [coord[0] for coord in reg_proc]
        ys = [coord[1] for coord in reg_proc]
    else:
        points_as_str = [e.split(',') for e in str_from_json.strip().split(';')]
        xs = [int(e[0]) for e in points_as_str]
        ys = [int(e[1]) for e in points_as_str]
    return [min(xs), max(xs), min(ys), max(ys)]


def _get_classifications(classification_list: List[Dict[str, Any]],
                         category_mapper: Optional[StringMapper],
                         threshold: float = 1.0
                         ) -> Optional[Dict[str, int]]:
    if not classification_list:
        return None
    classes = []
    for module_result in classification_list:
        for result in module_result['results']:
            lbl = result['label']
            if float(result['score']) < threshold:
                logging.info(f'ignore class {lbl} since score {result["score"]} > {threshold}')
                continue
            id = _get_index_or_name(lbl, category_mapper)
            if id is not None:
                classes.append(id)
    return classes if len(classes) > 0 else None


def _get_index_or_name(label: str, category_mapper: Optional[StringMapper]) -> Optional[Union[int, str]]:
    if category_mapper is None:
        return label
    id = category_mapper.get_channel(label)
    if id == category_mapper.get_oov_id():
        logging.info(f'ignore class {label} since it is not in the category-mapper')
        return None
    return id


def _fill_regions(reg: Dict[str, Any],
                  content: List[Region],
                  category_mapper: Optional[StringMapper],
                  allow_xfile: bool = True,
                  threshold: float = 1.0) -> None:
    if 'classifications' in reg:  # check if exist and ist not empty
        classes = _get_classifications(reg['classifications'], category_mapper, threshold)
        if classes is not None:
            content.append(Region(
                id=reg['id'] if 'id' in reg else None,
                classes=classes,
                bb=_str_to_bb(reg['coordinates'])))
        if 'regions' in reg:
            for r in reg['regions']:
                _fill_regions(r, content, category_mapper)
    elif allow_xfile and 'regionClassification' in reg:
        idx_or_name = _get_index_or_name(reg['regionClassification'], category_mapper)
        if idx_or_name is not None:
            content.append(Region(
                id=reg['id'] if 'id' in reg else None,
                classes=[idx_or_name],
                bb=_str_to_bb(reg['coords']['points'])))
    if 'regions' in reg and len(reg['regions']) != 0:
        for sub_reg in reg['regions']:
            _fill_regions(sub_reg, content, category_mapper, allow_xfile, threshold)
