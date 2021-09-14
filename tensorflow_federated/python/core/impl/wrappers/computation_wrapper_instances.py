# Copyright 2018, The TensorFlow Federated Authors.
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
"""Definitions of specific computation wrapper instances."""

import functools

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.federated_context import federated_computation_utils
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_serialization
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper

# The documentation of the arguments and return values from the wrapper_fns
# is quite detailed and can be found in `computation_wrapper.py` along with
# the definitions of `_wrap` and `ComputationWrapper`. In order to avoid having
# to repeat those descriptions (and make any relevant changes in four separate
# places) the documentation here simply forwards readers over.
#
# pylint:disable=g-doc-args,g-doc-return-or-yield


def _tf_wrapper_fn(parameter_type, name):
  """Wrapper function to plug Tensorflow logic into the TFF framework.

  This function is passed through `computation_wrapper.ComputationWrapper`.
  Documentation its arguments can be found inside the definition of that class.
  """
  del name  # Unused.
  if not type_analysis.is_tensorflow_compatible_type(parameter_type):
    raise TypeError('`tf_computation`s can accept only parameter types with '
                    'constituents `SequenceType`, `StructType` '
                    'and `TensorType`; you have attempted to create one '
                    'with the type {}.'.format(parameter_type))
  ctx_stack = context_stack_impl.context_stack
  tf_serializer = tensorflow_serialization.tf_computation_serializer(
      parameter_type, ctx_stack)
  arg = next(tf_serializer)
  try:
    result = yield arg
  except Exception as e:  # pylint: disable=broad-except
    tf_serializer.throw(e)
  comp_pb, extra_type_spec = tf_serializer.send(result)
  tf_serializer.close()
  yield computation_impl.ConcreteComputation(comp_pb, ctx_stack,
                                             extra_type_spec)


tensorflow_wrapper = computation_wrapper.ComputationWrapper(
    computation_wrapper.PythonTracingStrategy(_tf_wrapper_fn))


def _federated_computation_wrapper_fn(parameter_type, name):
  """Wrapper function to plug orchestration logic into the TFF framework.

  This function is passed through `computation_wrapper.ComputationWrapper`.
  Documentation its arguments can be found inside the definition of that class.
  """
  ctx_stack = context_stack_impl.context_stack
  if parameter_type is None:
    parameter_name = None
  else:
    parameter_name = 'arg'
  fn_generator = federated_computation_utils.federated_computation_serializer(
      parameter_name=parameter_name,
      parameter_type=parameter_type,
      context_stack=ctx_stack,
      suggested_name=name)
  arg = next(fn_generator)
  try:
    result = yield arg
  except Exception as e:  # pylint: disable=broad-except
    fn_generator.throw(e)
  target_lambda, extra_type_spec = fn_generator.send(result)
  fn_generator.close()
  yield computation_impl.ConcreteComputation(target_lambda.proto, ctx_stack,
                                             extra_type_spec)


federated_computation_wrapper = computation_wrapper.ComputationWrapper(
    computation_wrapper.PythonTracingStrategy(
        _federated_computation_wrapper_fn))

# pylint:enable=g-doc-args,g-doc-return-or-yield


def building_block_to_computation(building_block):
  """Converts a computation building block to a computation impl."""
  py_typecheck.check_type(building_block,
                          building_blocks.ComputationBuildingBlock)
  return computation_impl.ConcreteComputation(building_block.proto,
                                              context_stack_impl.context_stack)


def _check_returns_type_helper(fn, expected_return_type):
  """Helper for `check_returns_type`."""
  if not computation_wrapper.is_function(fn):
    raise ValueError(f'`assert_raises` expected a function, but found {fn}.')

  @functools.wraps(fn)
  def wrapped_func(*args, **kwargs):
    result = fn(*args, **kwargs)
    if result is None:
      raise ValueError('TFF computations may not return `None`. '
                       'Consider instead returning `()`.')
    result_type = type_conversions.infer_type(result)
    if not result_type.is_identical_to(expected_return_type):
      raise TypeError(
          f'Value returned from `{fn.__name__}` did not match asserted type.\n'
          + computation_types.type_mismatch_error_message(
              result_type,
              expected_return_type,
              computation_types.TypeRelation.IDENTICAL,
              second_is_expected=True))
    return result

  return wrapped_func


def check_returns_type(*args):
  """Checks that the decorated function returns values of the provided type.

  This decorator can be used to ensure that a TFF computation returns a value
  of the expected type. For example:

  ```
  @tff.tf_computation(tf.int32, tf.int32)
  @tff.check_returns_type(tf.int32)
  def add(a, b):
    return a + b
  ```

  It can also be applied to non-TFF (Python) functions to ensure that the values
  they return conform to the expected type.

  Note that this assertion is run whenever the function is called. In the case
  of `@tff.tf_computation` and `@tff.federated_computation`s, this means that
  the assertion will run when the computation is traced. To enable this,
  `@tff.check_returns_type` should be applied *inside* the `tff.tf_computation`:

  ```
  # YES:
  @tff.tf_computation(...)
  @tff.check_returns_type(...)
  ...

  # NO:
  @tff.check_returns_type(...) # Don't put this before the line below
  @tff.tf_computation(...)
  ...
  ```

  Args:
    *args: Either a Python function, or TFF type spec, or both (function first).

  Returns:
    If invoked with a function as an argument, returns an instance of a TFF
    computation constructed based on this function. If called without one, as
    in the typical decorator style of usage, returns a callable that expects
    to be called with the function definition supplied as a parameter. See
    also `tff.tf_computation` for an extended documentation.
  """
  if not args:
    raise ValueError('`assert_return`s called without a return type')
  if computation_wrapper.is_function(args[0]):
    # If the first argument on the list is a Python function or a
    # tf.function, this is the one that's being wrapped. This is the case of
    # either a decorator invocation without arguments as "@xyz" applied to a
    # function definition, of an inline invocation as "... = xyz(lambda....).
    if len(args) != 2:
      raise ValueError(
          f'`check_returns_type` expected two arguments: a function to decorate '
          f'and an expected return type. Found {len(args)} arguments: {args}')
    return _check_returns_type_helper(args[0],
                                      computation_types.to_type(args[1]))
  else:
    # The function is being invoked as a decorator with arguments.
    # The arguments come first, then the returned value is applied to
    # the function to be wrapped.
    if len(args) != 1:
      raise ValueError(
          f'`check_returns_type` expected a single argument specifying the '
          f'return type. Found {len(args)} arguments: {args}')
    return_type = computation_types.to_type(args[0])
    if return_type is None:
      raise ValueError('Asserted return type may not be `None`. '
                       'Consider instead a return type of `()`')
    return lambda fn: _check_returns_type_helper(fn, return_type)
