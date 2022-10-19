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
#
# pytype: skip-file
# This module disables the PyType analyzer, see
# https://github.com/tensorflow/federated/blob/main/docs/pytype.md for more
# information.
"""Abstractions for client work in learning algorithms."""

from typing import Optional

import attr

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning.templates import hparams_base


@attr.s(frozen=True)
class ClientResult():
  """A structure containing the result of `ClientWorkProcess.next` computation.

  Attributes:
    update: The local update to model weights produced by clients.
    update_weight: A weight for weighted aggregation of the `update`.
  """
  update = attr.ib()
  update_weight = attr.ib()


class ClientDataTypeError(TypeError):
  """`TypeError` for incorrect type of client data."""


class ClientResultTypeError(TypeError):
  """`TypeError` for incorrect structure of result of client work."""


# TODO(b/240314933): Move this (or refactor this) to a more general location.
def _is_allowed_client_data_type(type_spec: computation_types.Type) -> bool:
  """Determines whether a given type is a (possibly nested) sequence type."""
  if type_spec.is_sequence():
    return type_analysis.is_tensorflow_compatible_type(type_spec.element)
  elif type_spec.is_struct():
    return all(
        _is_allowed_client_data_type(element_type)
        for element_type in type_spec.children())
  else:
    return False


# TODO(b/240314933): Move this (or refactor this) to a more general location.
def _type_check_initialize_fn(initialize_fn: computation_base.Computation):
  if not initialize_fn.type_signature.result.is_federated():
    raise errors.TemplateNotFederatedError(
        f'Provided `initialize_fn` must return a federated type, but found '
        f'return type:\n{initialize_fn.type_signature.result}\nTip: If you '
        f'see a collection of federated types, try wrapping the returned '
        f'value in `tff.federated_zip` before returning.')
  if initialize_fn.type_signature.result.placement != placements.SERVER:
    raise errors.TemplatePlacementError(
        f'The state controlled by a `ClientWorkProcess` must be placed at '
        f'the SERVER, but found type: {initialize_fn.type_signature.result}.')


# TODO(b/240314933): Move this (or refactor this) to a more general location.
def _check_next_fn_is_federated(next_fn: computation_base.Computation):
  """Checks that a given `next_fn` has federated inputs and outputs."""
  next_types = (
      structure.flatten(next_fn.type_signature.parameter) +
      structure.flatten(next_fn.type_signature.result))
  if not all([t.is_federated() for t in next_types]):
    offending_types = '\n- '.join(
        [t for t in next_types if not t.is_federated()])
    raise errors.TemplateNotFederatedError(
        f'Provided `next_fn` must be a *federated* computation, that is, '
        f'operate on `tff.FederatedType`s, but found\n'
        f'next_fn with type signature:\n{next_fn.type_signature}\n'
        f'The non-federated types are:\n {offending_types}.')


# TODO(b/240314933): Move this (or refactor this) to a more general location.
def _type_check_next_fn_parameters(next_fn: computation_base.Computation):
  """Validates the input types of `next_fn` in a `ClientWorkProcess`."""
  next_fn_param = next_fn.type_signature.parameter
  if not next_fn_param.is_struct():
    raise errors.TemplateNextFnNumArgsError(
        f'The `next_fn` must have exactly three input arguments, but found '
        f'the following input type which is not a Struct: {next_fn_param}.')
  if len(next_fn_param) != 3:
    next_param_str = '\n- '.join([str(t) for t in next_fn_param])
    raise errors.TemplateNextFnNumArgsError(
        f'The `next_fn` must have exactly three input arguments, but found '
        f'{len(next_fn_param)} input arguments:\n{next_param_str}')
  second_next_param = next_fn_param[1]
  client_data_param = next_fn_param[2]
  if second_next_param.placement != placements.CLIENTS:
    raise errors.TemplatePlacementError(
        f'The second input argument of `next_fn` must be placed at CLIENTS '
        f'but found {second_next_param}.')
  if client_data_param.placement != placements.CLIENTS:
    raise errors.TemplatePlacementError(
        f'The third input argument of `next_fn` must be placed at CLIENTS '
        f'but found {client_data_param}.')
  if not _is_allowed_client_data_type(client_data_param.member):
    raise ClientDataTypeError(
        f'The third input argument of `next_fn` must be a sequence or '
        f'a structure of squences, but found {client_data_param}.')


# TODO(b/240314933): Move this (or refactor this) to a more general location.
def _type_check_next_fn_result(next_fn: computation_base.Computation):
  """Validates the output types of `next_fn` in a `ClientWorkProcess`."""
  next_fn_result = next_fn.type_signature.result
  if (not next_fn_result.result.is_federated() or
      next_fn_result.result.placement != placements.CLIENTS):
    raise errors.TemplatePlacementError(
        f'The "result" attribute of the return type of `next_fn` must be '
        f'placed at CLIENTS, but found {next_fn_result.result}.')
  if (not next_fn_result.result.member.is_struct_with_python() or
      next_fn_result.result.member.python_container is not ClientResult):
    raise ClientResultTypeError(
        f'The "result" attribute of the return type of `next_fn` must have '
        f'the `ClientResult` container, but found {next_fn_result.result}.')
  if next_fn_result.measurements.placement != placements.SERVER:
    raise errors.TemplatePlacementError(
        f'The "measurements" attribute of return type of `next_fn` must be '
        f'placed at SERVER, but found {next_fn_result.measurements}.')


class ClientWorkProcess(measured_process.MeasuredProcess):
  """A stateful process capturing work at clients during learning.

  Client work encapsulates the main work performed by clients as part of a
  federated learning algorithm, such as several steps of gradient descent based
  on the client data, and returning a update to the initial model weights.

  A `ClientWorkProcess` is a `tff.templates.MeasuredProcess` that formalizes the
  type signature of `initialize` and `next` for the core work performed by
  clients in a learning process.
  """

  def __init__(self,
               initialize_fn: computation_base.Computation,
               next_fn: computation_base.Computation,
               *,
               get_hparams_fn: Optional[computation_base.Computation] = None,
               set_hparams_fn: Optional[computation_base.Computation] = None):
    """Initializes a `ClientWorkProcess`.

    The `initialize_fn` and `next_fn` must have the following type signatures:
    ```
      - initialize_fn: ( -> S@SERVER)
      - next_fn: (<S@SERVER,
                   A@CLIENTS,
                   {D*}@CLIENTS>
                  ->
                  <state=S@SERVER,
                   result=ClientResult(B, C)@CLIENTS,
                   measurements=M@SERVER>)
    ```
    with `A`, `B`, `C`, and `D` not dependent on other types here. `A`
    represents a parameter informing the client update (such as a client's
    model weights). `D*` is a `tff.SequenceType` of client data.

    Note that the output of `next_fn` must have a structure matching
    `tff.templates.MeasuredProcessOutput`. The `result` field of this output
    has type `tff.learning.templates.ClientResult(B, C)` where `B` represents a
    client's update (such as a model update) and `C` represents the weight of
    this update when using weighted aggregation across clients.

    If provided, the `get_hparams_fn` and `set_hparams_fn` must be non-federated
    computations with the following type signatures:
    ```
      - get_hparams_fn: (S -> H)
      - set_hparams_fn: (<S, H> -> S)
    ```
    Here, `S` must match the state `S` of `initialize_fn` and `next_fn`, and `H`
    represents the hyperparameter type.

    Args:
      initialize_fn: A `tff.Computation` matching the criteria above.
      next_fn: A `tff.Computation` matching the criteria above.
      get_hparams_fn: An optional `tff.Computation` matching the criteria above.
        If not provided, this defaults to a computation that returns an empty
        ordred dictionary, regardless of the contents of the state.
      set_hparams_fn: An optional `tff.Computation` matching the criteria above.
        If not provided, this defaults to a pass-through computation, that
        returns the input state regardless of the hparams passed in.

    Raises:
      TemplateNotFederatedError: If any of the federated computations provided
        do not return a federated type.
      TemplateNextFnNumArgsError: If the `next_fn` has an incorrect number
        of arguments.
      TemplatePlacementError: If any of the federated computations have an
        incorrect placement.
      ClientDataTypeError: If the third input of `next_fn` is not a sequence
        type placed at `CLIENTS`.
      ClientResultTypeError: If the second output of `next_fn` does not meet the
        criteria outlined above.
      GetHparamsTypeError: If the type signature of `get_hparams_fn` does not
        meet the criteria above.
      SetHparamsTypeError: If the type signature of `set_hparams_fn` does not
        meet the criteria above.
    """
    super().__init__(initialize_fn, next_fn, next_is_multi_arg=True)

    _type_check_initialize_fn(initialize_fn)
    _check_next_fn_is_federated(next_fn)
    _type_check_next_fn_parameters(next_fn)
    _type_check_next_fn_result(next_fn)

    state_type = initialize_fn.type_signature.result.member
    if get_hparams_fn is not None:
      hparams_base.type_check_get_hparams_fn(get_hparams_fn, state_type)
    else:
      get_hparams_fn = hparams_base.build_basic_hparams_getter(state_type)

    hparams_type = get_hparams_fn.type_signature.result

    if set_hparams_fn is not None:
      hparams_base.type_check_set_hparams_fn(set_hparams_fn, state_type)
    else:
      set_hparams_fn = hparams_base.build_basic_hparams_setter(
          state_type, hparams_type)

    self._get_hparams_fn = get_hparams_fn
    self._set_hparams_fn = set_hparams_fn

  @property
  def get_hparams(self) -> computation_base.Computation:
    return self._get_hparams_fn

  @property
  def set_hparams(self) -> computation_base.Computation:
    return self._set_hparams_fn
