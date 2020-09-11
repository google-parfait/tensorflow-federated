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
"""Defines functions and classes for constructing a TFF iterative process."""

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_base


class InvalidConstructorArgumentsError(TypeError):
  pass


class InitializeFnHasArgsError(InvalidConstructorArgumentsError):

  def __init__(self, initialize_fn):
    message = ('`initialize_fn` must be a no-arg `tff.Computation`, but found '
               f'parameter type:\n{initialize_fn.type_signature.parameter}')
    super().__init__(message)


class NextMustAcceptStateFromInitializeError(InvalidConstructorArgumentsError):

  def __init__(self, initialize_result_type, next_parameter_type):
    message = (
        'The return type of `initialize_fn` must be assignable to the first '
        'parameter of `next_fn`, but \n'
        f'`initialize_fn` returned type:\n{initialize_result_type}\n'
        f'`next_fn`\'s argument type is:\n{next_parameter_type}')
    super().__init__(message)


class NextMustReturnStateError(InvalidConstructorArgumentsError):

  def __init__(self, next_result_type, state_type):
    message = (
        'The return type of `next_fn` must be assignable to the state type\n'
        'returned by `initialize_fn` and accepted by `next_fn` but found\n'
        f'`next_fn` which returns type:\n`{next_result_type}`\n'
        f'which does not match state type:\n`{state_type}`')
    super().__init__(message)


class IterativeProcess(object):
  """A process that includes an initialization and iterated computation.

  An iterated process will usually be driven by a control loop like:

  ```python
  def initialize():
    ...

  def next(state):
    ...

  iterative_process = IterativeProcess(initialize, next)
  state = iterative_process.initialize()
  for round in range(num_rounds):
    state = iterative_process.next(state)
  ```

  The iteration step can accept arguments in addition to `state` (which must be
  the first argument), and return additional arguments:

  ```python
  def next(state, item):
    ...

  iterative_process = ...
  state = iterative_process.initialize()
  for round in range(num_rounds):
    state, output = iterative_process.next(state, round)
  ```
  """

  def __init__(self, initialize_fn, next_fn):
    """Creates a `tff.templates.IterativeProcess`.

    Args:
      initialize_fn: A no-arg `tff.Computation` that creates the initial state
        of the chained computation.
      next_fn: A `tff.Computation` that defines an iterated function. If
        `initialize_fn` returns a type `T`, then `next_fn` must return a type
        `U` which is compatible with `T` or multiple values where the first type
        is `U`, and accept either a single argument of type `U` or multiple
        arguments where the first argument must be of type `U`.

    Raises:
      TypeError: `initialize_fn` and `next_fn` are not compatible function
        types.
    """
    py_typecheck.check_type(initialize_fn, computation_base.Computation)
    if initialize_fn.type_signature.parameter is not None:
      raise InitializeFnHasArgsError(initialize_fn)
    initialize_result_type = initialize_fn.type_signature.result

    py_typecheck.check_type(next_fn, computation_base.Computation)
    next_parameter_type = next_fn.type_signature.parameter
    # `next_first_parameter_type` may be `next_parameter_type` or
    # `next_parameter_type[0]`, depending on which one was assignable from
    # `initialize_result_type`.
    if next_parameter_type.is_assignable_from(initialize_result_type):
      # The only argument is the state type
      state_type = next_parameter_type
    elif (next_parameter_type.is_struct() and next_parameter_type and
          next_parameter_type[0].is_assignable_from(initialize_result_type)):
      # The first argument is the state type
      state_type = next_parameter_type[0]
    else:
      raise NextMustAcceptStateFromInitializeError(initialize_result_type,
                                                   next_parameter_type)

    next_result_type = next_fn.type_signature.result
    if state_type.is_assignable_from(next_result_type):
      # The whole return value is the state type
      pass
    elif (next_result_type.is_struct() and next_result_type and
          state_type.is_assignable_from(next_result_type[0])):
      # The first return value is state type
      pass
    else:
      raise NextMustReturnStateError(next_result_type, state_type)
    self._initialize_fn = initialize_fn
    self._next_fn = next_fn

  @property
  def initialize(self):
    """A no-arg `tff.Computation` that returns the initial state."""
    return self._initialize_fn

  @property
  def next(self):
    """A `tff.Computation` that produces the next state.

    Its first argument should always be the current state (originally produced
    by `tff.templates.IterativeProcess.initialize`), and the first (or only)
    returned value is the updated state.

    Returns:
      A `tff.Computation`.
    """
    return self._next_fn
