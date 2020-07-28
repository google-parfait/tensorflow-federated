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
      initialize_fn: a no-arg `tff.Computation` that creates the initial state
        of the chained computation.
      next_fn: a `tff.Computation` that defines an iterated function. If
        `initialize_fn` returns a type _T_, then `next_fn` must return a type
        _U_ which is compatible with _T_ or multiple values where the first type
        is _U_, and accept either a single argument of type _U_ or multiple
        arguments where the first argument must be of type _U_.

    Raises:
      TypeError: `initialize_fn` and `next_fn` are not compatible function
        types.
    """
    py_typecheck.check_type(initialize_fn, computation_base.Computation)
    if initialize_fn.type_signature.parameter is not None:
      raise TypeError(
          'initialize_fn must be a no-arg tff.Computation, but found parameter '
          '{}'.format(initialize_fn.type_signature))
    initialize_result_type = initialize_fn.type_signature.result

    py_typecheck.check_type(next_fn, computation_base.Computation)
    if next_fn.type_signature.parameter.is_struct(
    ) and next_fn.type_signature.parameter:
      next_first_param_type = next_fn.type_signature.parameter[0]
    else:
      next_first_param_type = next_fn.type_signature.parameter
    if not next_first_param_type.is_assignable_from(initialize_result_type):
      raise TypeError('The return type of initialize_fn must be assignable '
                      'to the first parameter of next_fn, but found\n'
                      'initialize_fn.type_signature.result=\n{}\n'
                      'next_fn.type_signature.parameter[0]=\n{}'.format(
                          initialize_result_type, next_first_param_type))

    next_result_type = next_fn.type_signature.result
    if not next_first_param_type.is_assignable_from(next_result_type):
      # This might be multiple output next_fn, check if the first argument might
      # be the state. If still not the right type, raise an error.
      if next_result_type.is_struct():
        next_result_type = next_result_type[0]
      if not next_first_param_type.is_assignable_from(next_result_type):
        raise TypeError('The return type of next_fn must be assignable to the '
                        'first parameter, but found\n'
                        'next_fn.type_signature.parameter[0]=\n{}\n'
                        'actual next_result_type=\n{}'.format(
                            next_first_param_type, next_result_type))
    self._initialize_fn = initialize_fn
    self._next_fn = next_fn

  @property
  def initialize(self):
    """A no-arg `tff.Computation` that returns the initial state."""
    return self._initialize_fn

  @property
  def next(self):
    """A `tff.Computation` that produces the next state.

    The first argument of should always be the current state (originally
    produced by `tff.templates.IterativeProcess.initialize`), and the first (or
    only) returned value is the updated state.

    Returns:
      A `tff.Computation`.
    """
    return self._next_fn
