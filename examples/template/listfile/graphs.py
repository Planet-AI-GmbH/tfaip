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
from examples.template.listfile.params import TemplateModelParams
from tfaip.model.graphbase import GraphBase


class TemplateGraph(GraphBase[TemplateModelParams]):
    def __init__(self, params: TemplateModelParams, name="template_graph", **kwargs):
        super(TemplateGraph, self).__init__(params, name=name, **kwargs)
        # Create all layers
        raise NotImplementedError

    def build_graph(self, inputs, training=None):
        # Connect all layers and return a dict of the outputs
        raise NotImplementedError
