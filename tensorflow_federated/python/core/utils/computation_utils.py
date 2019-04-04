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
"""Defines utility functions for constructing TFF computations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core import api as tff


def update_state(state, **kwargs):
  """Returns a new `state` (a namedtuple) with updated with kwargs."""
  # TODO(b/129569441): Support AnonymousTuple as well.
  if not py_typecheck.is_named_tuple(state):
    raise TypeError('state must be a namedtuple, but found {}'.format(
        type(state)))
  d = state._asdict()
  d.update(kwargs)
  return type(state)(**d)


class StatefulFn(object):
  """A base class for stateful functions."""

  def __init__(self, initialize_fn, next_fn):
    """Creates the StatefulFn.

    Args:
      initialize_fn: A no-arg function that returns a Python container which can
        be converted to a `tff.Value`, placed on the `tff.SERVER`, and
        passed as the first argument of `__call__`. This may be
        called in vanilla TensorFlow code, typically wrapped as a
        `tff.tf_compuatation`, as part of the initialization of a larger
        state object.
      next_fn: A function matching the signature of `__call__`, see below.
    """
    py_typecheck.check_callable(initialize_fn)
    py_typecheck.check_callable(next_fn)
    self._initialize_fn = initialize_fn
    self._next_fn = next_fn

  def initialize(self):
    """Returns the initial state."""
    return self._initialize_fn()

  def __call__(self, state, *args, **kwargs):
    """Performs the stateful function call.

    Args:
      state: A `tff.Value` placed on the `tff.SERVER`.
      *args: Arguments to the function.
      **kwargs: Arguments to the function.

    Returns:
       A tuple of `tff.Value`s (state@SERVER, ...) where
         * state: The updated state, to be passed to the next invocation
           of call.
         * ...: The result of the aggregation.
    """
    return self._next_fn(tff.to_value(state), *args, **kwargs)


class StatefulAggregateFn(StatefulFn):
  """A simple container for a stateful aggregation function.

  A typical (though trivial) example would be:

  ```
  stateless_federated_mean = tff.utils.StatefulAggregateFn(
      initialize_fn=lambda: (),  # The state is an empty tuple.
      next_fn=lambda state, value, weight=None: (
          state, tff.federated_mean(value, weight=weight)))
  ```
  """

  def __call__(self, state, value, weight=None):
    """Performs an aggregate of value@CLIENTS, with optional weight@CLIENTS.

    This is a function intended to (only) be invoked in the context
    of a `tff.federated_computation`. It shold be compatible with the
    TFF type signature

    ```
    (state@SERVER, value@CLIENTS, weight@CLIENTS) ->
         (state@SERVER, aggregate@SERVER).
    ```

    Args:
      state: A `tff.Value` placed on the `tff.SERVER`.
      value: A `tff.Value` to be aggregated, placed on the `tff.CLIENTS`.
      weight: An optional `tff.Value` for weighting values,
        placed on the `tff.CLIENTS`.

    Returns:
       A tuple of `tff.Value`s (state@SERVER, aggregate@SERVER) where
         * state: The updated state.
         * aggregate: The result of the aggregation of `value` weighted by
             `weight.
    """
    return self._next_fn(tff.to_value(state), tff.to_value(value),
                         tff.to_value(weight))


class StatefulBroadcastFn(StatefulFn):
  """A simple container for a stateful broadcast function.

   A typical (though trivial) example would be:

   ```
   stateless_federated_broadcast = tff.utils.StatefulBroadcastFn(
     initialize_fn=lambda: (),
     next_fn=lambda state, value: (
         state, tff.federated_broadcast(value)))
   ```
  """

  def __call__(self, state, value):
    """Performs a broadcast of value@SERVER, producing value@CLIENTS.

    This is a function intended to (only) be invoked in the context
    of a `tff.federated_computation`. It shold be compatible with the
    TFF type signature
    `(state@SERVER, value@SERVER) -> (state@SERVER, value@CLIENTS)`.

    Args:
      state: A `tff.Value` placed on the `tff.SERVER`.
      value: A `tff.Value` to be broadcast to the `tff.CLIENTS`.


    Returns:
       A tuple of `tff.Value`s (state@SERVER, value@CLIENTS) where
         * state: The updated state.
         * aggregate: The `value` now placed (communicated) to the
           `tff.CLIENTS`.
    """
    return self._next_fn(tff.to_value(state), tff.to_value(value))


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
    """Creates a `tff.IterativeProcess`.

    Args:
      initialize_fn: a no-arg `tff.Computation` that creates the initial state
        of the chained computation.
      next_fn: a `tff.Computation` that defines an iterated function. If
        `initialize_fn` returns a type _T_, then `next_fn` must also return type
        _T_  or multiple values where the first is of type _T_, and accept
        either a single argument of type _T_ or multiple arguments where the
        first argument must be of type _T_.

    Raises:
      TypeError: `initialize_fn` and `next_fn` are not compatible function
        types.
    """
    py_typecheck.check_type(initialize_fn, tff.Computation)
    if initialize_fn.type_signature.parameter is not None:
      raise TypeError('initialize_fn must be a no-arg tff.Computation, '
                      'but found parameter ' +
                      str(initialize_fn.type_signature))
    initialize_result_type = initialize_fn.type_signature.result

    py_typecheck.check_type(next_fn, tff.Computation)
    if isinstance(next_fn.type_signature.parameter, tff.NamedTupleType):
      next_first_param_type = next_fn.type_signature.parameter[0]
    else:
      next_first_param_type = next_fn.type_signature.parameter
    if initialize_result_type != next_first_param_type:
      raise TypeError('The return type of initialize_fn should match the '
                      'first parameter of next_fn, but found\n'
                      'initialize_fn.type_signature.result={}\n'
                      'next_fn.type_signature.parameter[0]={}'.format(
                          initialize_result_type, next_first_param_type))

    next_result_type = next_fn.type_signature.result
    if next_first_param_type != next_result_type:
      # This might be multiple output next_fn, check if the first argument might
      # be the state. If still not the right type, raise and error.
      if isinstance(next_result_type, tff.NamedTupleType):
        next_result_type = next_result_type[0]
      if next_first_param_type != next_result_type:
        raise TypeError('The return type of next_fn should match the '
                        'first parameter, but found\n'
                        'next_fn.type_signature.parameter[0]={}\n'
                        'next_fn.type_signature.result={}'.format(
                            next_first_param_type,
                            next_fn.type_signature.result))
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
    produced by `tff.IterativeProcess.initialize`), and the first (or only)
    returned value is the updated state.

    Returns:
      A `tff.Computation`.
    """
    return self._next_fn
