# Copyright 2019, The TensorFlow Federated Authors.
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
"""Defines a template for a stateful process."""

from typing import Optional

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.templates import errors


def _is_nonempty_struct(type_signature) -> bool:
  return type_signature.is_struct() and type_signature


def _infer_state_type(initialize_result_type, next_parameter_type,
                      next_is_multi_arg):
  """Infers the state type from the `initialize` and `next` types."""
  if next_is_multi_arg is None:
    # `state_type` may be `next_parameter_type` or
    # `next_parameter_type[0]`, depending on which one was assignable from
    # `initialize_result_type`.
    if next_parameter_type.is_assignable_from(initialize_result_type):
      return next_parameter_type
    if (_is_nonempty_struct(next_parameter_type) and
        next_parameter_type[0].is_assignable_from(initialize_result_type)):
      return next_parameter_type[0]
    raise errors.TemplateStateNotAssignableError(
        'The return type of `initialize_fn` must be assignable to either\n'
        'the whole argument to `next_fn` or the first argument to `next_fn`,\n'
        'but found `initialize_fn` return type:\n'
        f'{initialize_result_type}\n'
        'and `next_fn` with whole argument type:\n'
        f'{next_parameter_type}')
  elif next_is_multi_arg:
    if not _is_nonempty_struct(next_parameter_type):
      raise errors.TemplateNextFnNumArgsError(
          'Expected `next_parameter_type` to be a structure type of at least '
          f'length one, but found type:\n{next_parameter_type}')
    if next_parameter_type[0].is_assignable_from(initialize_result_type):
      return next_parameter_type[0]
    raise errors.TemplateStateNotAssignableError(
        'The return type of `initialize_fn` must be assignable to the first\n'
        'argument to `next_fn`, but found `initialize_fn` return type:\n'
        f'{initialize_result_type}\n'
        'and `next_fn` whose first argument type is:\n'
        f'{next_parameter_type}')
  else:
    # `next_is_multi_arg` is `False`
    if next_parameter_type.is_assignable_from(initialize_result_type):
      return next_parameter_type
    raise errors.TemplateStateNotAssignableError(
        'The return type of `initialize_fn` must be assignable to the whole\n'
        'argument to `next_fn`, but found `initialize_fn` return type:\n'
        f'{initialize_result_type}\n'
        'and `next_fn` whose first argument type is:\n'
        f'{next_parameter_type}')


class IterativeProcess:
  """A process that includes an initialization and iterated computation.

  An iterated process will usually be driven by a control loop like:

  ```python
  def initialize_fn():
    ...

  def next_fn(state):
    ...

  iterative_process = IterativeProcess(initialize_fn, next_fn)
  state = iterative_process.initialize()
  for round in range(num_rounds):
    state = iterative_process.next(state)
  ```

  The `initialize_fn` function must return an object which is expected as input
  to and returned by the `next_fn` function. By convention, we refer to this
  object as `state`.

  The iteration step (`next_fn` function) can accept arguments in addition to
  `state` (which must be the first argument), and return additional arguments,
  with `state` being the first output argument:

  ```python
  def next_fn(state, round_num):
    ...

  iterative_process = ...
  state = iterative_process.initialize()
  for round in range(num_rounds):
    state, output = iterative_process.next(state, round)
  ```
  """

  def __init__(self,
               initialize_fn: computation_base.Computation,
               next_fn: computation_base.Computation,
               next_is_multi_arg: Optional[bool] = None):
    """Creates a `tff.templates.IterativeProcess`.

    Args:
      initialize_fn: A no-arg `tff.Computation` that returns the initial state
        of the iterative process. Let the type of this state be called `S`.
      next_fn: A `tff.Computation` that represents the iterated function. The
        first or only argument must match the state type `S`. The first or only
        return value must also match state type `S`.
      next_is_multi_arg: An optional boolean indicating that `next_fn` will
        receive more than just the state argument (if `True`) or only the state
        argument (if `False`). This parameter is primarily used to provide
        better error messages.

    Raises:
      TypeError: If `initialize_fn` and `next_fn` are not instances of
        `tff.Computation`.
      TemplateInitFnParamNotEmptyError: If `initialize_fn` has any input
        arguments.
      TemplateStateNotAssignableError: If the `state` returned by either
        `initialize_fn` or `next_fn` is not assignable to the first input
        argument of `next_fn`.
    """
    py_typecheck.check_type(initialize_fn, computation_base.Computation)
    if initialize_fn.type_signature.parameter is not None:
      raise errors.TemplateInitFnParamNotEmptyError(
          f'Provided `initialize_fn` must be a no-arg function, but found '
          f'input argument(s) {initialize_fn.type_signature.parameter}.')
    initialize_result_type = initialize_fn.type_signature.result

    py_typecheck.check_type(next_fn, computation_base.Computation)
    next_parameter_type = next_fn.type_signature.parameter
    state_type = _infer_state_type(initialize_result_type, next_parameter_type,
                                   next_is_multi_arg)

    next_result_type = next_fn.type_signature.result
    if state_type.is_assignable_from(next_result_type):
      # The whole return value is the state type
      pass
    elif (_is_nonempty_struct(next_result_type) and
          state_type.is_assignable_from(next_result_type[0])):
      # The first return value is state type
      pass
    else:
      raise errors.TemplateStateNotAssignableError(
          f'The first return argument of `next_fn` must be '
          f'assignable to its first input argument, but found\n'
          f'`next_fn` which returns type:\n{next_result_type}\n'
          f'which does not match its first input argument:\n{state_type}')

    self._state_type = state_type
    self._initialize_fn = initialize_fn
    self._next_fn = next_fn

  @property
  def initialize(self) -> computation_base.Computation:
    """A no-arg `tff.Computation` that returns the initial state."""
    return self._initialize_fn

  @property
  def next(self) -> computation_base.Computation:
    """A `tff.Computation` that produces the next state.

    Its first argument should always be the current state (originally produced
    by `tff.templates.IterativeProcess.initialize`), and the first (or only)
    returned value is the updated state.

    Returns:
      A `tff.Computation`.
    """
    return self._next_fn

  @property
  def state_type(self) -> computation_types.Type:
    """The `tff.Type` of the state of the process."""
    return self._state_type
