# Copyright 2021, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""General utilities for adding hyperparameter getters and setters."""

import collections

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.types import computation_types


class GetHparamsTypeError(TypeError):
  """`TypeError` for incorrect input and output of get_hparams."""


class SetHparamsTypeError(TypeError):
  """`TypeError` for incorrect input and output of get_hparams."""


def type_check_get_hparams_fn(
    get_hparams_fn: computation_base.Computation,
    state_type: computation_types.Type,
):
  """Validates the type signature of `get_hparams_fn` in `ClientWorkProcess`."""
  py_typecheck.check_type(get_hparams_fn, computation_base.Computation)
  get_hparams_state_type = get_hparams_fn.type_signature.parameter
  if (
      get_hparams_state_type is None
      or not get_hparams_state_type.is_assignable_from(state_type)
  ):
    raise GetHparamsTypeError(
        'The input to get_hparams must be compatible with the state type '
        f'{state_type}, but found type {get_hparams_state_type}.'
    )


def type_check_set_hparams_fn(
    set_hparams_fn: computation_base.Computation,
    state_type: computation_types.Type,
):
  """Validates the type signature of `set_hparams_fn` in `ClientWorkProcess`."""
  py_typecheck.check_type(set_hparams_fn, computation_base.Computation)
  set_hparams_parameter = set_hparams_fn.type_signature.parameter
  if (
      not isinstance(set_hparams_parameter, computation_types.StructType)
      or len(set_hparams_parameter) != 2
  ):
    raise SetHparamsTypeError(
        'Expected two input arguments to set_hparams, but found '
        f'{set_hparams_parameter}.'
    )
  set_hparams_state_type = set_hparams_parameter[0]
  if not set_hparams_state_type.is_assignable_from(state_type):
    raise SetHparamsTypeError(
        'The first input to set_hparams must be compatible with the state '
        f'type {state_type}, but found {set_hparams_state_type}.'
    )
  set_hparams_result_type = set_hparams_fn.type_signature.result
  if not set_hparams_result_type.is_assignable_from(state_type):
    raise SetHparamsTypeError(
        'The output of set_hparams must be compatible with the state '
        f'of type {state_type} but found {set_hparams_result_type}.'
    )


def build_basic_hparams_getter(
    state_type: computation_types.Type,
) -> computation_base.Computation:
  """Creates a `tff.Computation` that returns an empty ordered dictionary."""

  @tensorflow_computation.tf_computation(state_type)
  def get_hparams_computation(state):
    del state
    return collections.OrderedDict()

  return get_hparams_computation


def build_basic_hparams_setter(
    state_type: computation_types.Type, hparams_type: computation_types.Type
) -> computation_base.Computation:
  """Creates a `tff.Computation` that returns the state, unchanged."""

  @tensorflow_computation.tf_computation(state_type, hparams_type)
  def set_hparams_computation(state, hparams):
    del hparams
    return state

  return set_hparams_computation
