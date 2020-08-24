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
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import computation_factory
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.executors import execution_context
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.types import placement_literals
from tensorflow_federated.python.core.impl.types import type_factory
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


def install_executor(executor_factory_instance):
  context = execution_context.ExecutionContext(executor_factory_instance)
  return context_stack_impl.context_stack.install(context)


def executors(*args):
  """A decorator for creating tests parameterized by executors.

  Note: To use this decorator your test is required to inherit from
  `parameterized.TestCase`.

  The decorator can be called without arguments:

  ```
  @executors
  def foo(self):
    ...
  ```

  or with arguments:

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

  def decorator(fn, *named_executors):
    if not named_executors:
      named_executors = [
          ('local', executor_stacks.local_executor_factory()),
          ('sizing', executor_stacks.sizing_executor_factory()),
      ]

    @parameterized.named_parameters(*named_executors)
    def wrapped_fn(self, executor):
      """Install a particular execution context before running `fn`."""
      context = execution_context.ExecutionContext(executor)
      with context_stack_impl.context_stack.install(context):
        fn(self)

    return wrapped_fn

  if len(args) == 1 and callable(args[0]):
    return decorator(args[0])
  else:
    return lambda fn: decorator(fn, *args)


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

  async def create_struct(self, elements):
    target_val = await self._target.create_struct(
        structure.map_structure(lambda x: x.value, elements))
    wrapped_val = TracingExecutorValue(self, self._get_new_value_index(),
                                       target_val)
    self._trace.append(
        ('create_struct', structure.map_structure(lambda x: x.index,
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


def create_dummy_intrinsic_def_federated_aggregate():
  value = intrinsic_defs.FEDERATED_AGGREGATE
  type_signature = computation_types.FunctionType([
      type_factory.at_clients(tf.float32),
      tf.float32,
      type_factory.reduction_op(tf.float32, tf.float32),
      type_factory.binary_op(tf.float32),
      computation_types.FunctionType(tf.float32, tf.float32),
  ], type_factory.at_server(tf.float32))
  return value, type_signature


def create_dummy_intrinsic_def_federated_apply():
  value = intrinsic_defs.FEDERATED_APPLY
  type_signature = computation_types.FunctionType([
      type_factory.unary_op(tf.float32),
      type_factory.at_server(tf.float32),
  ], type_factory.at_server(tf.float32))
  return value, type_signature


def create_dummy_intrinsic_def_federated_broadcast():
  value = intrinsic_defs.FEDERATED_BROADCAST
  type_signature = computation_types.FunctionType(
      type_factory.at_server(tf.float32),
      type_factory.at_clients(tf.float32, all_equal=True))
  return value, type_signature


def create_dummy_intrinsic_def_federated_collect():
  value = intrinsic_defs.FEDERATED_COLLECT
  type_signature = computation_types.FunctionType(
      type_factory.at_clients(tf.float32),
      type_factory.at_server(computation_types.SequenceType(tf.float32)))
  return value, type_signature


def create_dummy_intrinsic_def_federated_eval_at_clients():
  value = intrinsic_defs.FEDERATED_EVAL_AT_CLIENTS
  type_signature = computation_types.FunctionType(
      computation_types.FunctionType(None, tf.float32),
      type_factory.at_clients(tf.float32))
  return value, type_signature


def create_dummy_intrinsic_def_federated_eval_at_server():
  value = intrinsic_defs.FEDERATED_EVAL_AT_SERVER
  type_signature = computation_types.FunctionType(
      computation_types.FunctionType(None, tf.float32),
      type_factory.at_server(tf.float32))
  return value, type_signature


def create_dummy_intrinsic_def_federated_map():
  value = intrinsic_defs.FEDERATED_MAP
  type_signature = computation_types.FunctionType([
      type_factory.unary_op(tf.float32),
      type_factory.at_clients(tf.float32),
  ], type_factory.at_clients(tf.float32))
  return value, type_signature


def create_dummy_intrinsic_def_federated_map_all_equal():
  value = intrinsic_defs.FEDERATED_MAP_ALL_EQUAL
  type_signature = computation_types.FunctionType([
      type_factory.unary_op(tf.float32),
      type_factory.at_clients(tf.float32, all_equal=True),
  ], type_factory.at_clients(tf.float32, all_equal=True))
  return value, type_signature


def create_dummy_intrinsic_def_federated_mean():
  value = intrinsic_defs.FEDERATED_MEAN
  type_signature = computation_types.FunctionType(
      type_factory.at_clients(tf.float32), type_factory.at_server(tf.float32))
  return value, type_signature


def create_dummy_intrinsic_def_federated_reduce():
  value = intrinsic_defs.FEDERATED_REDUCE
  type_signature = computation_types.FunctionType([
      type_factory.at_clients(tf.float32),
      tf.float32,
      type_factory.reduction_op(tf.float32, tf.float32),
  ], type_factory.at_server(tf.float32))
  return value, type_signature


def create_dummy_intrinsic_def_federated_secure_sum():
  value = intrinsic_defs.FEDERATED_SECURE_SUM
  type_signature = computation_types.FunctionType([
      type_factory.at_clients(tf.float32),
      tf.float32,
  ], type_factory.at_server(tf.float32))
  return value, type_signature


def create_dummy_intrinsic_def_federated_sum():
  value = intrinsic_defs.FEDERATED_SUM
  type_signature = computation_types.FunctionType(
      type_factory.at_clients(tf.float32), type_factory.at_server(tf.float32))
  return value, type_signature


def create_dummy_intrinsic_def_federated_value_at_clients():
  value = intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS
  type_signature = computation_types.FunctionType(
      tf.float32, type_factory.at_clients(tf.float32, all_equal=True))
  return value, type_signature


def create_dummy_intrinsic_def_federated_value_at_server():
  value = intrinsic_defs.FEDERATED_VALUE_AT_SERVER
  type_signature = computation_types.FunctionType(
      tf.float32, type_factory.at_server(tf.float32))
  return value, type_signature


def create_dummy_intrinsic_def_federated_weighted_mean():
  value = intrinsic_defs.FEDERATED_WEIGHTED_MEAN
  type_signature = computation_types.FunctionType([
      type_factory.at_clients(tf.float32),
      type_factory.at_clients(tf.float32),
  ], type_factory.at_server(tf.float32))
  return value, type_signature


def create_dummy_intrinsic_def_federated_zip_at_clients():
  value = intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS
  type_signature = computation_types.FunctionType([
      type_factory.at_clients(tf.float32),
      type_factory.at_clients(tf.float32)
  ], type_factory.at_clients([tf.float32, tf.float32]))
  return value, type_signature


def create_dummy_intrinsic_def_federated_zip_at_server():
  value = intrinsic_defs.FEDERATED_ZIP_AT_SERVER
  type_signature = computation_types.FunctionType(
      [type_factory.at_server(tf.float32),
       type_factory.at_server(tf.float32)],
      type_factory.at_server([tf.float32, tf.float32]))
  return value, type_signature


def create_dummy_placement_literal():
  """Returns a `placement_literals.PlacementLiteral` and type."""
  value = placement_literals.SERVER
  type_signature = computation_types.PlacementType()
  return value, type_signature


def create_dummy_computation_call():
  """Returns a call computation and type."""
  fn, fn_type = create_dummy_computation_tensorflow_constant()
  type_signature = fn_type.result
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      call=pb.Call(function=fn))
  return value, type_signature


def create_dummy_computation_intrinsic():
  """Returns a intrinsic computation and type."""
  intrinsic_def, type_signature = create_dummy_intrinsic_def_federated_eval_at_server(
  )
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      intrinsic=pb.Intrinsic(uri=intrinsic_def.uri))
  return value, type_signature


def create_dummy_computation_lambda_empty():
  """Returns a lambda computation and type `( -> <>)`."""
  value = computation_factory.create_lambda_empty_struct()
  type_signature = computation_types.FunctionType(None, [])
  return value, type_signature


def create_dummy_computation_lambda_identity():
  """Returns a lambda computation and type `(float32 -> float32)`."""
  tensor_type = computation_types.TensorType(tf.float32)
  value = computation_factory.create_lambda_identity(tensor_type)
  type_signature = computation_types.FunctionType(tensor_type, tensor_type)
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
  type_signature = computation_types.TensorType(tf.float32)
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


def create_dummy_computation_tensorflow_add():
  """Returns a tensorflow computation and type.

  `(<float32,float32> -> float32)`
  """
  type_spec = tf.float32

  with tf.Graph().as_default() as graph:
    parameter_1_value, parameter_1_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', type_spec, graph)
    parameter_2_value, parameter_2_binding = tensorflow_utils.stamp_parameter_in_graph(
        'y', type_spec, graph)
    result_value = tf.add(parameter_1_value, parameter_2_value)
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result_value, graph)

  parameter_type = computation_types.StructType([type_spec, type_spec])
  type_signature = computation_types.FunctionType(parameter_type, result_type)
  struct_binding = pb.TensorFlow.StructBinding(
      element=[parameter_1_binding, parameter_2_binding])
  parameter_binding = pb.TensorFlow.Binding(struct=struct_binding)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding)
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow)
  return value, type_signature


def create_dummy_computation_tensorflow_constant():
  """Returns a tensorflow computation and type `( -> float32)`."""
  value = 10.0
  tensor_type = computation_types.TensorType(tf.float32)
  value, type_signature = tensorflow_computation_factory.create_constant(
      value, tensor_type)
  return value, type_signature


def create_dummy_computation_tensorflow_empty():
  """Returns a tensorflow computation and type `( -> <>)`."""
  value, type_signature = tensorflow_computation_factory.create_empty_tuple()
  return value, type_signature


def create_dummy_computation_tensorflow_identity():
  """Returns a tensorflow computation and type `(float32 -> float32)`."""
  tensor_type = computation_types.TensorType(tf.float32)
  value, type_signature = tensorflow_computation_factory.create_identity(
      tensor_type)
  return value, type_signature


def create_dummy_computation_tensorflow_random():
  """Returns a tensorflow computation and type `( -> float32)`."""

  with tf.Graph().as_default() as graph:
    result = tf.random.normal([])
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


def create_dummy_computation_tensorflow_tuple():
  """Returns a tensorflow computation and type.

  `( -> <('a', float32), ('b', float32), ('c', float32)>)`
  """
  value = 10.0

  with tf.Graph().as_default() as graph:
    names = ['a', 'b', 'c']
    result = structure.Struct((n, tf.constant(value)) for n in names)
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
  fn, fn_type = create_dummy_computation_tensorflow_constant()
  element_value = pb.Computation(
      type=type_serialization.serialize_type(fn_type),
      call=pb.Call(function=fn))
  element_type = fn_type.result
  elements = [pb.Struct.Element(name=n, value=element_value) for n in names]
  type_signature = computation_types.StructType(
      (n, element_type) for n in names)
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      struct=pb.Struct(element=elements))
  return value, type_signature


def create_dummy_value_at_clients(number_of_clients: int = 3):
  """Returns a Python value and federated type at clients."""
  value = [float(x) for x in range(10, number_of_clients + 10)]
  type_signature = type_factory.at_clients(tf.float32)
  return value, type_signature


def create_dummy_value_at_clients_all_equal():
  """Returns a Python value and federated type at clients and all equal."""
  value = 10.0
  type_signature = type_factory.at_clients(tf.float32, all_equal=True)
  return value, type_signature


def create_dummy_value_at_server():
  """Returns a Python value and federated type at server."""
  value = 10.0
  type_signature = type_factory.at_server(tf.float32)
  return value, type_signature


def create_dummy_value_unplaced():
  """Returns a Python value and unplaced type."""
  value = 10.0
  type_signature = computation_types.TensorType(tf.float32)
  return value, type_signature
