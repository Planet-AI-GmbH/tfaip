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
"""Definition of the WarmStarter"""
import logging
import re
from typing import List, Optional, NoReturn

import numpy as np
import tensorflow as tf

from tfaip import WarmStartParams

logger = logging.getLogger(__name__)


def longest_common_startstr(strs: List[str]) -> str:
    # Search for the longest common start string of each of the passed strs
    longest_common = ""
    for c in zip(*strs):
        if len(set(c)) == 1:
            longest_common += c[0]
        else:
            break

    return longest_common


class WarmStarter:
    """The WarmStarter handles the loading of a pretrained model an applies the weights to the current one.

    See WarmStartParams for configuration. Both SavedModels and Checkpoints are supported.
    """

    def __init__(self, params: WarmStartParams):
        self._params = params

        if (params.exclude or params.include) and params.allow_partial:
            raise ValueError("Allow partial is only allowed if neither exclude not include is specified")

    def _trim(self, names: List[str]) -> List[str]:
        if self._params.trim_graph_name:
            longest_common = longest_common_startstr([n for n in names if "print_limit" not in n])
            if "/" in longest_common:
                # find slash, only trim full var names
                longest_common = longest_common[: longest_common.rfind("/") + 1]
                return [s[len(longest_common) :] if "print_limit" not in s else s for s in names]
        return names

    @staticmethod
    def _replace_name(name, rename_rules: Optional[List[str]] = None):
        if rename_rules:
            for replace in rename_rules:
                from_to = replace.split("->")
                if len(from_to) != 2:
                    raise ValueError(f"Renaming rule {replace} must follow the 'from->to' schemata")

                name = name.replace(from_to[0], from_to[1])
        return name

    def _apply_renamings(self, names, rename_params: Optional[List[str]]):
        names = self._trim(names)
        if rename_params:
            names = [self._replace_name(n, rename_params) for n in names]

        names = [self._auto_replace_numbers(n) for n in names]

        return names

    def _auto_replace_numbers(self, name):
        for r in self._params.auto_remove_numbers_for:
            name = re.sub(f"(.*/){r}_\\d+(/.*)", f"\\1{r}\\2", name)

        return name

    def warmstart(self, target_model: tf.keras.Model, custom_objects=None):
        if not self._params.model:
            logger.debug("No warm start model provided")
            return

        # Names that will be ignored in both the loaded and target model (no real weights)
        names_to_ignore = {"print_limit:0", "count:0"}

        # 1. Load as saved model. if successful -> 3, if failed -> 2
        # 2. If OSError (-> no saved model), load weights directly fron checkpoint -> 5
        # 3. Load weights and assign (only works if the model is identical), if failed -> 4
        # 4. Load weights by name -> 5
        # 5. Apply renaming rules and match weights
        try:
            # 1. Load model
            src_model = tf.keras.models.load_model(self._params.model, compile=False, custom_objects=custom_objects)
        except OSError:
            # 2. load as checkpoint, then go to 5.
            logger.debug(f"Could not load '{self._params.model}' as saved model. Attempting to load as a checkpoint.")
            ckpt = tf.train.load_checkpoint(self._params.model)
            name_shapes = ckpt.get_variable_to_shape_map()
            model_to_load_var_names = name_shapes.keys()

            def rename_ckpt_var_name(name: str):
                name = name.replace("/.ATTRIBUTES/VARIABLE_VALUE", "")
                return name

            names = self._apply_renamings(model_to_load_var_names, self._params.rename)
            if self._params.add_suffix:
                names = [name + self._params.add_suffix for name in names]
            weights_ckpt = {
                rename_ckpt_var_name(pp_name): ckpt.get_tensor(name)
                for pp_name, name in zip(names, model_to_load_var_names)
            }
            all_loaded_weights = weights_ckpt
            trainable_name_to_loaded_var_name = {k: k for k in names}
        else:
            # 3. apply weights directly
            logger.info("Source model successfully loaded for warmstart.")
            skip_direct_apply = self._params.include or self._params.exclude
            if not skip_direct_apply:
                try:
                    self.apply_weights(
                        target_model, [np.asarray(w) for w in src_model.weights if w.name not in names_to_ignore]
                    )
                except Exception as e:
                    # 4. Load and rename weights
                    logger.exception(e)
                    logger.warning(
                        "Weights could not be copied directly. Retrying application of renaming rules to"
                        "match variable names."
                    )
                else:
                    # successful, nothing to do
                    return

            loaded_var_names = self._apply_renamings([w.name for w in src_model.weights], self._params.rename)
            model_to_load_var_names = [w.name for w in src_model.weights]
            loaded_weights = list(zip(loaded_var_names, src_model.weights, src_model.get_weights()))
            all_loaded_weights = {name: weight for name, var, weight in loaded_weights if name not in names_to_ignore}
            trainable_name_to_loaded_var_name = {
                k: v
                for k, v in zip(
                    self._apply_renamings(all_loaded_weights.keys(), self._params.rename_targets),
                    all_loaded_weights.keys(),
                )
            }

        # 5. Apply names with renaming rules
        target_var_names = self._apply_renamings([w.name for w in target_model.weights], self._params.rename_targets)
        target_weights = list(zip(target_var_names, target_model.weights, target_model.get_weights()))
        if len(set(name for name, var, weight in target_weights)) != len(target_weights):
            logger.critical(
                "Non unique names detected in model weight names. You can ignore this warning but the "
                "model will not be initialized correctly!"
            )
        all_trainable_target_weights = {
            name: weight for name, var, weight in target_weights if name not in names_to_ignore
        }
        # all_trainable_target_weights = {name: weight for name, var, weight in target_weights if var.trainable}
        trainable_name_to_target_var_name = {
            k: v
            for k, v in zip(
                self._apply_renamings(all_trainable_target_weights.keys(), self._params.rename_targets),
                all_trainable_target_weights.keys(),
            )
        }
        target_var_name_to_trainable_name = {v: k for k, v in trainable_name_to_target_var_name.items()}
        logger.info(f"Warm-starting from {self._params.model}")
        try:
            # First try to reinstantiate the model, and apply the renamings on the weights,
            # if this fails, load the weights as checkpoints, apply additional renamings, and then try to match
            # the source and target weights
            model = tf.keras.models.load_model(self._params.model, compile=False, custom_objects=custom_objects)
        except OSError:
            logger.debug(f"Could not load '{self._params.model}' as saved model. Attempting to load as a checkpoint.")

        # Filter the params and validate
        names_target = set(trainable_name_to_target_var_name.keys())
        names_loaded = set(trainable_name_to_loaded_var_name.keys())
        if self._params.exclude or self._params.include:
            names_to_load = names_loaded
            if self._params.include:
                inc = re.compile(self._params.include)
                names_to_load = [name for name in names_to_load if inc.fullmatch(name)]

            if self._params.exclude:
                exc = re.compile(self._params.exclude)
                names_to_load = [name for name in names_to_load if not exc.fullmatch(name)]

            if len(names_target.intersection(names_to_load)) == 0:
                raise NameError(f"Not a weight could be matched.\nLoaded: {names_to_load}\nTarget: {names_target}")
        elif self._params.allow_partial:
            names_to_load = names_target.intersection(names_loaded)
        else:
            diff_target = names_loaded.difference(names_target)
            diff_load = names_target.difference(names_loaded)
            if len(diff_target) > 0 or len(diff_load) > 0:
                raise NameError(
                    f"Not all weights could be matched:\nTargets '{diff_target}'\nLoaded: '{diff_load}'. "
                    f"\nUse allow_partial to allow partial loading"
                )

        names_to_load = names_target
        new_weights = []
        warm_weights_names = []
        cold_weights_names = []
        non_trainable_weights_names = []
        for weight_idx, name in enumerate(target_var_names):
            # access original weight via index because names might not be unique (e.g. in metrics)
            trainable_name = target_var_name_to_trainable_name.get(name, None)  # None == not existing
            if trainable_name in names_to_load:
                warm_weights_names.append(name)
                new_weights.append(all_loaded_weights[trainable_name_to_loaded_var_name[trainable_name]])
            else:
                if name in all_trainable_target_weights:
                    cold_weights_names.append(name)
                else:
                    non_trainable_weights_names.append(name)
                new_weights.append(target_weights[weight_idx][2])  # set to original weight
        not_loaded_weights = [name for name in names_loaded if name not in warm_weights_names]
        newline = "\n\t"
        logger.info(newline.join(["model-to-load weights:"] + model_to_load_var_names))
        logger.info(newline.join(["renamed unmached weights:"] + [str(x) for x in not_loaded_weights]))
        logger.info(newline.join(["Warm weights:"] + [str(x) for x in warm_weights_names]))
        logger.info(newline.join(["Cold weights:"] + [str(x) for x in cold_weights_names]))
        logger.info(f"There are {len(non_trainable_weights_names)} non trainable weights.")
        if len(names_to_load) == 0:
            raise ValueError("No warmstart weight could be matched! Set TFAIP_LOG_LEVEL=INFO for more information.")

        self.apply_weights(target_model, new_weights)

    def apply_weights(self, target_model, new_weights) -> NoReturn:
        """
        By default, all weights of the target model are set to the new weights.
        This function can be overwritte, to handle setting the parameters. E.g. in ATR, if a Codec adaption should be
        done, i.e. only a sub set of one weight matrix is selected.

        Args:
            target_model: Target model
            new_weights: New weights of the model
        """
        target_model.set_weights(new_weights)
