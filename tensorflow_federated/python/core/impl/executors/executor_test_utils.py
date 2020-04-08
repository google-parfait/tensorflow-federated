# Lint as: python3
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
"""Utils for testing executors."""

import asyncio

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import reference_executor
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import placement_literals
from tensorflow_federated.python.core.impl.compiler import type_factory
from tensorflow_federated.python.core.impl.compiler import type_serialization
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.executors import execution_context
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


def install_executor(executor_factory_instance):
  context = execution_context.ExecutionContext(executor_factory_instance)
  return context_stack_impl.context_stack.install(context)


def executors(*args):
  """A decorator for creating tests parameterized by executors.

  Note: To use this decorator your test is required to inherit from
  `parameterized.TestCase`.

  1.  The decorator can be specified without arguments:

      ```
      @executors
      def foo(self):
        ...
      ```

  2.  The decorator can be called with arguments:

      ```
      @executors(
          ('label', executor),
          ...
      )
      def foo(self):
        ...
      ```

  If the decorator is specified without arguments or is called with no
  arguments, the default this decorator with parameterize the test by the
  following executors:

  *   reference executor
  *   local executor

  If the decorator is called with arguments the arguments must be in a form that
  is accpeted by `parameterized.named_parameters`.

  Args:
    *args: Either a test function to be decorated or named executors for the
      decorated method, either a single iterable, or a list of tuples or dicts.

  Returns:
     A test generator to be handled by `parameterized.TestGeneratorMetaclass`.
  """

  def executor_decorator(fn):
    """Create a wrapped function with custom execution contexts."""

    def wrapped_fn(self, executor):
      """Install a particular execution context before running `fn`."""
      # Executors inheriting from `executor_base.Executor` will need to be
      # wrapped in an execution context. The `ReferenceExecutor` is special and
      # inherits from `context_base.Context`, so we don't wrap.
      if not isinstance(executor, context_base.Context):
        context = execution_context.ExecutionContext(executor)
      else:
        context = executor
      with context_stack_impl.context_stack.install(context):
        fn(self)

    return wrapped_fn

  def decorator(fn, *named_executors):
    """Construct a custom `parameterized.named_parameter` decorator for `fn`."""
    if not named_executors:
      named_executors = [
          ('reference', reference_executor.ReferenceExecutor(compiler=None)),
          ('local', executor_stacks.local_executor_factory()),
      ]
    named_parameters_decorator = parameterized.named_parameters(
        *named_executors)
    fn = executor_decorator(fn)
    fn = named_parameters_decorator(fn)
    return fn

  if len(args) == 1 and callable(args[0]):
    return decorator(args[0])
  else:
    return lambda x: decorator(x, *args)


class AsyncTestCase(absltest.TestCase):
  """A test case that manages a new event loop for each test.

  Each test will have a new event loop instead of using the current event loop.
  This ensures that tests are isolated from each other and avoid unexpected side
  effects.

  Attributes:
    loop: An `asyncio` event loop.
  """

  def setUp(self):
    super().setUp()
    self.loop = asyncio.new_event_loop()

    # If `setUp()` fails, then `tearDown()` is not called; however cleanup
    # functions will be called. Register the newly created loop `close()`
    # function here to ensure it is closed after each test.
    self.addCleanup(self.loop.close)

  def run_sync(self, coro):
    return self.loop.run_until_complete(coro)


class TracingExecutor(executor_base.Executor):
  """Tracing executor keeps a log of all calls for use in testing."""

  def __init__(self, target):
    """Creates a new instance of a tracing executor.

    The tracing executor keeps the trace of all calls. Entries in the trace
    consist of the method name followed by arguments and the returned result,
    with the executor values represented as integer indexes starting from 1.

    Args:
      target: An instance of `executor_base.Executor`.
    """
    py_typecheck.check_type(target, executor_base.Executor)
    self._target = target
    self._last_used_index = 0
    self._trace = []

  @property
  def trace(self):
    return self._trace

  def _get_new_value_index(self):
    val_index = self._last_used_index + 1
    self._last_used_index = val_index
    return val_index

  async def create_value(self, value, type_spec=None):
    target_val = await self._target.create_value(value, type_spec)
    wrapped_val = TracingExecutorValue(self, self._get_new_value_index(),
                                       target_val)
    if type_spec is not None:
      self._trace.append(('create_value', value, type_spec, wrapped_val.index))
    else:
      self._trace.append(('create_value', value, wrapped_val.index))
    return wrapped_val

  async def create_call(self, comp, arg=None):
    if arg is not None:
      target_val = await self._target.create_call(comp.value, arg.value)
      wrapped_val = TracingExecutorValue(self, self._get_new_value_index(),
                                         target_val)
      self._trace.append(
          ('create_call', comp.index, arg.index, wrapped_val.index))
      return wrapped_val
    else:
      target_val = await self._target.create_call(comp.value)
      wrapped_val = TracingExecutorValue(self, self._get_new_value_index(),
                                         target_val)
      self._trace.append(('create_call', comp.index, wrapped_val.index))
      return wrapped_val

  async def create_tuple(self, elements):
    target_val = await self._target.create_tuple(
        anonymous_tuple.map_structure(lambda x: x.value, elements))
    wrapped_val = TracingExecutorValue(self, self._get_new_value_index(),
                                       target_val)
    self._trace.append(
        ('create_tuple',
         anonymous_tuple.map_structure(lambda x: x.index,
                                       elements), wrapped_val.index))
    return wrapped_val

  def close(self):
    self._target.close()

  async def create_selection(self, source, index=None, name=None):
    target_val = await self._target.create_selection(
        source.value, index=index, name=name)
    wrapped_val = TracingExecutorValue(self, self._get_new_value_index(),
                                       target_val)
    self._trace.append(
        ('create_selection', source.index, index if index is not None else name,
         wrapped_val.index))
    return wrapped_val


class TracingExecutorValue(executor_value_base.ExecutorValue):
  """A value managed by `TracingExecutor`."""

  def __init__(self, owner, index, value):
    """Creates an instance of a value in the tracing executor.

    Args:
      owner: An instance of `TracingExecutor`.
      index: An integer identifying the value.
      value: An embedded value from the target executor.
    """
    py_typecheck.check_type(owner, TracingExecutor)
    py_typecheck.check_type(index, int)
    py_typecheck.check_type(value, executor_value_base.ExecutorValue)
    self._owner = owner
    self._index = index
    self._value = value

  @property
  def index(self):
    return self._index

  @property
  def value(self):
    return self._value

  @property
  def type_signature(self):
    return self._value.type_signature

  async def compute(self):
    result = await self._value.compute()
    self._owner.trace.append(('compute', self._index, result))
    return result


def create_dummy_intrinsic_def():
  """Returns a `intrinsic_defs.IntrinsicDef` and type."""
  value = intrinsic_defs.FEDERATED_EVAL_AT_SERVER
  type_signature = computation_types.FunctionType(
      computation_types.FunctionType(None, tf.int32),
      type_factory.at_server(tf.int32))
  return value, type_signature


def create_dummy_placement_literal():
  """Returns a `placement_literals.PlacementLiteral` and type."""
  value = placement_literals.SERVER
  type_signature = computation_types.PlacementType()
  return value, type_signature


def create_dummy_computation_call():
  """Returns a call computation and type."""
  function, function_type = create_dummy_computation_tensorflow_constant()
  type_signature = function_type.result
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      call=pb.Call(function=function))
  return value, type_signature


def create_dummy_computation_intrinsic():
  """Returns a intrinsic computation and type."""
  intrinsic_def, type_signature = create_dummy_intrinsic_def()
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      intrinsic=pb.Intrinsic(uri=intrinsic_def.uri))
  return value, type_signature


def create_dummy_computation_lambda_empty():
  """Returns a lambda computation and type `( -> <>)`."""
  result_type = computation_types.NamedTupleType([])
  type_signature = computation_types.FunctionType(None, result_type)
  result = pb.Computation(
      type=type_serialization.serialize_type(result_type),
      tuple=pb.Tuple(element=[]))
  fn = pb.Lambda(parameter_name=None, result=result)
  # We are unpacking the lambda argument here because `lambda` is a reserved
  # keyword in Python, but it is also the name of the parameter for a
  # `pb.Computation`.
  # https://developers.google.com/protocol-buffers/docs/reference/python-generated#keyword-conflicts
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature), **{'lambda': fn})  # pytype: disable=wrong-keyword-args
  return value, type_signature


def create_dummy_computation_lambda_identity(type_spec=tf.int32):
  """Returns a lambda computation and type `(T -> T)`."""
  type_signature = type_factory.unary_op(type_spec)
  result = pb.Computation(
      type=type_serialization.serialize_type(type_spec),
      reference=pb.Reference(name='a'))
  fn = pb.Lambda(parameter_name='a', result=result)
  # We are unpacking the lambda argument here because `lambda` is a reserved
  # keyword in Python, but it is also the name of the parameter for a
  # `pb.Computation`.
  # https://developers.google.com/protocol-buffers/docs/reference/python-generated#keyword-conflicts
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature), **{'lambda': fn})  # pytype: disable=wrong-keyword-args
  return value, type_signature


def create_dummy_computation_placement():
  """Returns a placement computation and type."""
  placement_literal, type_signature = create_dummy_placement_literal()
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      placement=pb.Placement(uri=placement_literal.uri))
  return value, type_signature


def create_dummy_computation_reference():
  """Returns a reference computation and type."""
  type_signature = computation_types.TensorType(tf.int32)
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      reference=pb.Reference(name='a'))
  return value, type_signature


def create_dummy_computation_selection():
  """Returns a selection computation and type."""
  source, source_type = create_dummy_computation_tuple()
  type_signature = source_type[0]
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      selection=pb.Selection(source=source, index=0))
  return value, type_signature


def create_dummy_computation_tensorflow_constant(value=10):
  """Returns a tensorflow computation and type `( -> T)`."""

  with tf.Graph().as_default() as graph:
    result = tf.constant(value)
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result, graph)

  type_signature = computation_types.FunctionType(None, result_type)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=None,
      result=result_binding)
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow)
  return value, type_signature


def create_dummy_computation_tensorflow_empty():
  """Returns a tensorflow computation and type `( -> <>)`."""
  return create_dummy_computation_tensorflow_constant(value=[])


def create_dummy_computation_tensorflow_identity(type_spec=tf.int32):
  """Returns a tensorflow computation and type `(T -> T)`."""

  with tf.Graph().as_default() as graph:
    parameter_value, parameter_binding = tensorflow_utils.stamp_parameter_in_graph(
        'a', type_spec, graph)
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        parameter_value, graph)

  type_signature = computation_types.FunctionType(type_spec, result_type)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding)
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow)
  return value, type_signature


def create_dummy_computation_tensorflow_tuple(value=10):
  """Returns a tensorflow computation and type.

  `( -> <('a', T), ('b', T), ('c', T)>)`

  Args:
    value: An optional integer value.
  """

  with tf.Graph().as_default() as graph:
    names = ['a', 'b', 'c']
    result = anonymous_tuple.AnonymousTuple(
        (n, tf.constant(value)) for n in names)
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result, graph)

  type_signature = computation_types.FunctionType(None, result_type)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=None,
      result=result_binding)
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow)
  return value, type_signature


def create_dummy_computation_tuple():
  """Returns a tuple computation and type."""
  names = ['a', 'b', 'c']
  element_value, element_type = create_dummy_computation_tensorflow_constant()
  elements = [pb.Tuple.Element(name=n, value=element_value) for n in names]
  type_signature = computation_types.NamedTupleType(
      (n, element_type) for n in names)
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tuple=pb.Tuple(element=elements))
  return value, type_signature


def create_dummy_value_clients():
  """Returns a Python value and type at clients."""
  value = [10] * 3
  type_signature = type_factory.at_clients(tf.int32)
  return value, type_signature


def create_dummy_value_clients_all_equal():
  """Returns a Python value and type at clients and all equal."""
  value = 10
  type_signature = type_factory.at_clients(tf.int32, all_equal=True)
  return value, type_signature


def create_dummy_value_server():
  """Returns a Python value and type at server."""
  value = 10
  type_signature = type_factory.at_server(tf.int32)
  return value, type_signature


def create_dummy_value_unplaced():
  """Returns a Python value and type unplaced."""
  value = 10
  type_signature = computation_types.TensorType(tf.int32)
  return value, type_signature
