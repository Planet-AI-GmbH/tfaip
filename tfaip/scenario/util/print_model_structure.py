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
from typing import NamedTuple, List, Any
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.layer_utils import count_params


def print_all_layers(root_layer, print_fn=None):
    if print_fn is None:
        print_fn = print

    line_length = 150  # 98
    positions = [0.5, 1.0]
    if positions[-1] <= 1:
        positions = [int(line_length * p) for p in positions]
    # header names for the different log elements
    to_display = ["Param # (output shape)", "Layer (type)"]

    print_fn("_" * line_length)
    print_row(to_display, positions, print_fn)
    print_fn("=" * line_length)

    layer_tree = LayerTreeNode(root_layer, list())
    unbuilt_layers = get_sub_layers(root_layer, layer_tree, "", layers_with_params_only=True)
    print_layers(layer_tree, "", [], positions, print_fn)
    print_fn("=" * line_length)

    trainable_count = count_params(root_layer.trainable_weights)
    non_trainable_count = count_params(root_layer.non_trainable_weights)

    print_fn("Total params: {:,}".format(trainable_count + non_trainable_count))
    print_fn("Trainable params: {:,}".format(trainable_count))
    print_fn("Non-trainable params: {:,}".format(non_trainable_count))
    if len(unbuilt_layers) > 0:
        print_fn("There are layers which are not yet built and accordingly the number of weights is unknown:")
    for unbuilt_layer in unbuilt_layers:
        print_fn("  " + unbuilt_layer)
    print_fn("_" * line_length)


def print_layers(layer_node, prefix, is_last, positions, print_fn):
    num_sublayers = len(layer_node.next)
    for node_idx in range(num_sublayers):
        is_last_sub = is_last + [node_idx == (num_sublayers - 1)]
        node = layer_node.next[node_idx]
        sub_layer = node.layer
        sub_layer_name = sub_layer.name
        full_layer_name = prefix + "/" + sub_layer_name if prefix else sub_layer_name
        if len(node.next) == 0:
            print_layer_summary(sub_layer, sub_layer_name, full_layer_name, positions, print_fn, is_last_sub, True)
        else:
            print_layer_summary(sub_layer, sub_layer_name, full_layer_name, positions, print_fn, is_last_sub, False)
            print_layers(node, full_layer_name, is_last_sub, positions, print_fn)


def get_sub_layers(layer, layer_node, prefix, layers_with_params_only):
    # Recursively loop through all layers
    unbuilt_layers = list()
    for sub_layer in layer._flatten_layers(False, False):
        full_layer_name = prefix + "/" + sub_layer.name if prefix else sub_layer.name
        if not sub_layer.built:
            unbuilt_layers.append(full_layer_name)
        if (not sub_layer.built) or (sub_layer.built and sub_layer.count_params() > 0) or (not layers_with_params_only):
            layer_node.next.append(LayerTreeNode(sub_layer, list()))
            unbuilt_layers.extend(
                get_sub_layers(sub_layer, layer_node.next[-1], full_layer_name, layers_with_params_only)
            )
    return unbuilt_layers


def print_row(fields, positions, print_fn):
    line = ""
    for i in range(len(fields)):
        if i > 0:
            line = line[:-1] + " "
        line += str(fields[i])
        line = line[: positions[i]]
        line += " " * (positions[i] - len(line))
    print_fn(line)


def print_layer_summary(layer, layer_name, full_layer_name, positions, print_fn, is_last, is_leaf, indent=2):

    cls_name = layer.__class__.__name__

    if layer.built:
        num_params = layer.count_params()

        trainable_weights = layer.trainable_weights
        non_trainable_weights = layer.non_trainable_weights
        num_trainable_weights = count_params(trainable_weights)
        num_non_trainable_weights = count_params(non_trainable_weights)

        if num_params > 0:
            if num_trainable_weights == num_params:
                numstring = ""
            elif num_trainable_weights == 0:
                numstring = "not "
            else:
                numstring = str(num_trainable_weights) + " "
            trainablestring = " [" + numstring + "trainable]"
        else:
            trainablestring = ""
    else:
        num_params = "?"
        trainablestring = " [? trainable]"

    if is_last[-1]:
        s = "\u2514"
    else:
        s = "\u251C"

    prefix = ""
    for b in is_last[:-1]:
        if not b:
            prefix += "\u2502" + " " * indent
        else:
            prefix += " " * (indent + 1)

    fields = [
        prefix + s + "\u2500" * (indent - 1) + " " + str(num_params) + trainablestring,
        prefix + s + "\u2500" * (indent - 1) + " " + layer_name + " (" + cls_name + ")",
    ]
    print_row(fields, positions, print_fn)

    if is_leaf:
        if layer.built:
            weight_names = [weight.name.split("/")[-1] for weight in layer.weights]
            num_trainable_as_list = [count_params([weight]) if weight.trainable else 0 for weight in layer.weights]
            weights = layer.get_weights()
            num_weights = len(weights)
            for weight_idx in range(num_weights):
                weight = weights[weight_idx]
                weight_name = weight_names[weight_idx]
                if not is_last[-1]:
                    sub_prefix = "\u2502" + " " * indent
                else:
                    sub_prefix = " " * (indent + 1)

                if weight_idx == (num_weights - 1):
                    ssub = "\u2514"
                else:
                    ssub = "\u251C"

                weight_size = np.size(weight)

                if weight_size > 0:
                    num_trainable_weights = num_trainable_as_list[weight_idx]
                    if num_trainable_weights == weight_size:
                        numstring = ""
                    elif num_trainable_weights == 0:
                        numstring = "not "
                    else:
                        numstring = str(num_trainable_weights) + " "
                    trainablestring = " [" + numstring + "trainable] " + str(np.shape(weight))
                else:
                    trainablestring = ""

                fields = [
                    prefix + sub_prefix + ssub + "\u2500" * (indent - 1) + " " + str(weight_size) + trainablestring,
                    prefix + sub_prefix + ssub + "\u2500" * (indent - 1) + " " + weight_name,
                ]
                print_row(fields, positions, print_fn)


class LayerTreeNode(NamedTuple):
    layer: tf.keras.layers.Layer
    next: List[Any]
