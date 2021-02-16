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
from tfaip.base.trainer.warmstart.warmstart_params import WarmstartParams
import tensorflow as tf
import logging
import re


logger = logging.getLogger(__name__)
logger.setLevel("INFO")


class Warmstarter:
    def __init__(self, params: WarmstartParams):
        self._params = params

        if (params.exclude or params.include) and params.allow_partial:
            raise ValueError("Allow partial is only allowed if neither exclude not include is specified")


    def _trim(self, name: str):
        if self._params.trim_graph_name:
            name = name[name.find('/') + 1:]
        return name

    def _replace_name(self, name):
        if self._params.rename:
            for replace in self._params.rename:
                from_to = replace.split('->')
                if len(from_to) != 2:
                    raise ValueError(f"Renaming rule {replace} must follow the 'from->to' schemata")

                name = name.replace(from_to[0], from_to[1])


        name = self._auto_replace_numbers(name)

        name += self._params.add_suffix


        return name

    def _auto_replace_numbers(self, name):
        for r in self._params.auto_remove_numbers_for:
            name = re.sub(f"(.*/){r}_\\d+(/.*)", f"\\1{r}\\2", name)

        return name

    def warmstart(self, target_model: tf.keras.Model, custom_objects=None):
        if not self._params.model:
            logger.debug("No warm start model provided")
            return

        target_var_names = [self._trim(self._auto_replace_numbers(w.name)) for w in target_model.weights]
        target_weights = list(zip(target_var_names, target_model.weights, target_model.get_weights()))
        all_trainable_target_weights = {name: weight for name, var, weight in target_weights if var.trainable}
        all_target_weights = {name: weight for name, var, weight in target_weights}
        logger.info(f"Warmstarting from {self._params.model}")
        try:
            model = tf.keras.models.load_model(self._params.model, compile=False, custom_objects=custom_objects)
            loaded_var_names = [self._trim(self._replace_name(w.name)) for w in model.weights]
            loaded_weights = list(zip(loaded_var_names, model.weights, model.get_weights()))
            all_loaded_weights = {name: weight for name, var, weight in loaded_weights if var.trainable}
        except OSError:
            logger.debug(f"Could not load '{self._params.model}' as saved model. Attempting to load as a checkpoint.")
            ckpt = tf.train.load_checkpoint(self._params.model)
            name_shapes = ckpt.get_variable_to_shape_map()
            var_names_ckpt = name_shapes.keys()

            def rename_ckpt_var_name(name: str):
                name = name.rstrip('/.ATTRIBUTES/VARIABLE_VALUE')
                name = self._replace_name(name)
                name = self._trim(name)
                return name
            weights_ckpt = {rename_ckpt_var_name(name): ckpt.get_tensor(name) for name in var_names_ckpt}
            all_loaded_weights = weights_ckpt

        names_target = set(all_trainable_target_weights.keys())
        names_loaded = set(all_loaded_weights.keys())
        if self._params.exclude or self._params.include:
            names_to_load = names_loaded
            if self._params.include:
                inc = re.compile(self._params.include)
                names_to_load = [name for name in names_to_load if inc.fullmatch(name)]

            if self._params.exclude:
                exc = re.compile(self._params.exclude)
                names_to_load = [name for name in names_to_load if not exc.fullmatch(name)]

        elif self._params.allow_partial:
            names_to_load = names_target.intersection(names_loaded)
        else:
            diff_target = names_loaded.difference(names_target)
            diff_load = names_target.difference(names_loaded)
            if len(diff_target) > 0 or len(diff_load) > 0:
                raise NameError(f"Not all weights could be matched: '{diff_target}' from '{diff_load}'. "
                                f"Use allow_partial to allow partial loading")

            names_to_load = names_target

        new_weights=[]
        warm_weights_names=[]
        cold_weights_names=[]
        for name in target_var_names:
            if name in names_to_load:
                warm_weights_names.append(name)
                new_weights.append(all_loaded_weights[name])
            else:
                cold_weights_names.append(name)
                new_weights.append(all_target_weights[name])
        not_loaded_weights=[]
        for name in names_loaded:
            if name not in warm_weights_names:
                not_loaded_weights.append(name)
        if self._params.debug_weight_names:
            newline='\n\t'
            logger.info(f'Not loaded weights:\n {newline.join([str(x) for x in not_loaded_weights])}')
            logger.info(f'Warm weights:\n {newline.join([str(x) for x in warm_weights_names])}')
            logger.info(f'Cold weights:\n {newline.join([str(x) for x in cold_weights_names])}')
        #new_weights = [all_loaded_weights[name] if name in names_to_load else all_target_weights[name] for name in target_var_names]
        target_model.set_weights(new_weights)
        logger.info(f"Warmstarted weights: {names_to_load}")
