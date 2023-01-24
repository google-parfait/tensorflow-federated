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
"""Utilities for testing executors."""

import asyncio

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.compiler import computation_factory
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_factory
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.executors import cardinalities_utils
from tensorflow_federated.python.core.impl.executors import eager_tf_executor
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_value_base
from tensorflow_federated.python.core.impl.executors import federated_resolving_strategy
from tensorflow_federated.python.core.impl.executors import federating_executor
from tensorflow_federated.python.core.impl.executors import ingestable_base
from tensorflow_federated.python.core.impl.executors import reference_resolving_executor
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_factory
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


class LocalTestExecutorFactory(executor_factory.ExecutorFactory):
  """Minimal implementation of a full local executor factory."""

  def __init__(
      self,
      default_num_clients: int = 0,
      leaf_executor_fn=eager_tf_executor.EagerTFExecutor,
  ):
    self._default_num_clients = default_num_clients
    self._leaf_executor_fn = leaf_executor_fn

  def create_executor(self, cardinalities):
    device = tf.config.list_logical_devices(device_type='CPU')[0]
    num_requested_clients = cardinalities.get(placements.CLIENTS)
    if num_requested_clients is None:
      num_clients = self._default_num_clients
    else:
      num_clients = num_requested_clients
    client_stacks = [
        self._leaf_executor_fn(device=device) for _ in range(num_clients)
    ]

    federating_strategy_factory = federated_resolving_strategy.FederatedResolvingStrategy.factory(
        {
            placements.CLIENTS: client_stacks,
            placements.SERVER: self._leaf_executor_fn(device=device),
        },
        local_computation_factory=tensorflow_computation_factory.TensorFlowComputationFactory(),
    )
    executor = federating_executor.FederatingExecutor(
        federating_strategy_factory, self._leaf_executor_fn(device=device)
    )
    return reference_resolving_executor.ReferenceResolvingExecutor(executor)

  def clean_up_executor(self, cardinalities):
    pass


class BasicTestExFactory(executor_factory.ExecutorFactory):
  """Minimal implementation of a parameterized executor factory."""

  def __init__(self, executor):
    if isinstance(executor, executor_base.Executor):
      self._executor = executor
    else:
      self._executor = None
      self._executor_fn = executor

  def create_executor(self, cardinalities):
    if self._executor is not None:
      return self._executor
    else:
      return self._executor_fn(cardinalities)

  def clean_up_executor(self, cardinalities):
    pass


class TestExecutionContext(context_base.SyncContext):
  """Minimal execution context for testing executors."""

  def __init__(self, ex_factory):
    self._executor_factory = ex_factory
    self._loop = asyncio.new_event_loop()

  def _unwrap_tensors(self, value):
    if tf.is_tensor(value):
      return value.numpy()
    elif isinstance(value, structure.Struct):
      return structure.Struct(
          (k, self._unwrap_tensors(v))
          for k, v in structure.iter_elements(value)
      )
    else:
      return value

  def invoke(self, comp, arg):
    cardinalities = cardinalities_utils.infer_cardinalities(
        arg, comp.type_signature.parameter
    )
    executor = self._executor_factory.create_executor(cardinalities)
    embedded_comp = self._loop.run_until_complete(
        executor.create_value(comp, comp.type_signature)
    )
    if arg is not None:
      if isinstance(arg, ingestable_base.Ingestable):
        arg_coro = self._loop.run_until_complete(arg.ingest(executor))
        arg = self._loop.run_until_complete(arg_coro.compute())
      embedded_arg = self._loop.run_until_complete(
          executor.create_value(arg, comp.type_signature.parameter)
      )
    else:
      embedded_arg = None
    embedded_call = self._loop.run_until_complete(
        executor.create_call(embedded_comp, embedded_arg)
    )
    return type_conversions.type_to_py_container(
        self._unwrap_tensors(
            self._loop.run_until_complete(embedded_call.compute())
        ),
        comp.type_signature.result,
    )


def install_executor(executor_factory_instance):
  context = TestExecutionContext(executor_factory_instance)
  return context_stack_impl.context_stack.install(context)


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
    wrapped_val = TracingExecutorValue(
        self, self._get_new_value_index(), target_val
    )
    if type_spec is not None:
      self._trace.append(('create_value', value, type_spec, wrapped_val.index))
    else:
      self._trace.append(('create_value', value, wrapped_val.index))
    return wrapped_val

  async def create_call(self, comp, arg=None):
    if arg is not None:
      target_val = await self._target.create_call(comp.reference, arg.reference)
      wrapped_val = TracingExecutorValue(
          self, self._get_new_value_index(), target_val
      )
      self._trace.append(
          ('create_call', comp.index, arg.index, wrapped_val.index)
      )
      return wrapped_val
    else:
      target_val = await self._target.create_call(comp.reference)
      wrapped_val = TracingExecutorValue(
          self, self._get_new_value_index(), target_val
      )
      self._trace.append(('create_call', comp.index, wrapped_val.index))
      return wrapped_val

  async def create_struct(self, elements):
    target_val = await self._target.create_struct(
        structure.map_structure(lambda x: x.reference, elements)
    )
    wrapped_val = TracingExecutorValue(
        self, self._get_new_value_index(), target_val
    )
    self._trace.append((
        'create_struct',
        structure.map_structure(lambda x: x.index, elements),
        wrapped_val.index,
    ))
    return wrapped_val

  def close(self):
    self._target.close()

  async def create_selection(self, source, index):
    target_val = await self._target.create_selection(source.reference, index)
    wrapped_val = TracingExecutorValue(
        self, self._get_new_value_index(), target_val
    )
    self._trace.append(
        ('create_selection', source.index, index, wrapped_val.index)
    )
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
  def reference(self):
    return self._value

  @property
  def type_signature(self):
    return self._value.type_signature

  async def compute(self):
    result = await self._value.compute()
    self._owner.trace.append(('compute', self._index, result))
    return result


def create_whimsy_intrinsic_def_federated_aggregate():  # pylint: disable=missing-function-docstring
  value = intrinsic_defs.FEDERATED_AGGREGATE
  type_signature = computation_types.FunctionType(
      [
          computation_types.at_clients(tf.float32),
          tf.float32,
          type_factory.reduction_op(tf.float32, tf.float32),
          type_factory.binary_op(tf.float32),
          computation_types.FunctionType(tf.float32, tf.float32),
      ],
      computation_types.at_server(tf.float32),
  )
  return value, type_signature


def create_whimsy_intrinsic_def_federated_apply():
  value = intrinsic_defs.FEDERATED_APPLY
  type_signature = computation_types.FunctionType(
      [
          type_factory.unary_op(tf.float32),
          computation_types.at_server(tf.float32),
      ],
      computation_types.at_server(tf.float32),
  )
  return value, type_signature


def create_whimsy_intrinsic_def_federated_broadcast():
  value = intrinsic_defs.FEDERATED_BROADCAST
  type_signature = computation_types.FunctionType(
      computation_types.at_server(tf.float32),
      computation_types.at_clients(tf.float32, all_equal=True),
  )
  return value, type_signature


def create_whimsy_intrinsic_def_federated_eval_at_clients():
  value = intrinsic_defs.FEDERATED_EVAL_AT_CLIENTS
  type_signature = computation_types.FunctionType(
      computation_types.FunctionType(None, tf.float32),
      computation_types.at_clients(tf.float32),
  )
  return value, type_signature


def create_whimsy_intrinsic_def_federated_eval_at_server():
  value = intrinsic_defs.FEDERATED_EVAL_AT_SERVER
  type_signature = computation_types.FunctionType(
      computation_types.FunctionType(None, tf.float32),
      computation_types.at_server(tf.float32),
  )
  return value, type_signature


def create_whimsy_intrinsic_def_federated_map():
  value = intrinsic_defs.FEDERATED_MAP
  type_signature = computation_types.FunctionType(
      [
          type_factory.unary_op(tf.float32),
          computation_types.at_clients(tf.float32),
      ],
      computation_types.at_clients(tf.float32),
  )
  return value, type_signature


def create_whimsy_intrinsic_def_federated_map_all_equal():
  value = intrinsic_defs.FEDERATED_MAP_ALL_EQUAL
  type_signature = computation_types.FunctionType(
      [
          type_factory.unary_op(tf.float32),
          computation_types.at_clients(tf.float32, all_equal=True),
      ],
      computation_types.at_clients(tf.float32, all_equal=True),
  )
  return value, type_signature


def create_whimsy_intrinsic_def_federated_mean():
  value = intrinsic_defs.FEDERATED_MEAN
  type_signature = computation_types.FunctionType(
      computation_types.at_clients(tf.float32),
      computation_types.at_server(tf.float32),
  )
  return value, type_signature


def create_whimsy_intrinsic_def_federated_secure_sum_bitwidth():
  value = intrinsic_defs.FEDERATED_SECURE_SUM_BITWIDTH
  type_signature = computation_types.FunctionType(
      [
          computation_types.at_clients(tf.int32),
          tf.int32,
      ],
      computation_types.at_server(tf.int32),
  )
  return value, type_signature


_WHIMSY_SELECT_CLIENT_KEYS_TYPE = computation_types.at_clients(
    computation_types.TensorType(tf.int32, [3])
)
_WHIMSY_SELECT_MAX_KEY_TYPE = computation_types.at_server(tf.int32)
_WHIMSY_SELECT_SERVER_STATE_TYPE = computation_types.at_server(tf.string)
_WHIMSY_SELECTED_TYPE = computation_types.to_type((tf.string, tf.int32))
_WHIMSY_SELECT_SELECT_FN_TYPE = computation_types.FunctionType(
    (tf.string, tf.int32), _WHIMSY_SELECTED_TYPE
)
_WHIMSY_SELECT_RESULT_TYPE = computation_types.at_clients(
    computation_types.SequenceType(_WHIMSY_SELECTED_TYPE)
)
_WHIMSY_SELECT_TYPE = computation_types.FunctionType(
    [
        _WHIMSY_SELECT_CLIENT_KEYS_TYPE,
        _WHIMSY_SELECT_MAX_KEY_TYPE,
        _WHIMSY_SELECT_SERVER_STATE_TYPE,
        _WHIMSY_SELECT_SELECT_FN_TYPE,
    ],
    _WHIMSY_SELECT_RESULT_TYPE,
)
_WHIMSY_SELECT_NUM_CLIENTS = 3


def create_whimsy_intrinsic_def_federated_secure_select():
  return intrinsic_defs.FEDERATED_SECURE_SELECT, _WHIMSY_SELECT_TYPE


def create_whimsy_intrinsic_def_federated_select():
  return intrinsic_defs.FEDERATED_SELECT, _WHIMSY_SELECT_TYPE


def create_whimsy_federated_select_args():
  client_keys = [[0, 1, 2]] * _WHIMSY_SELECT_NUM_CLIENTS
  max_key = 2
  server_state = 'abc'
  select_fn = create_whimsy_computation_tensorflow_identity(
      _WHIMSY_SELECTED_TYPE
  )
  return [
      (client_keys, _WHIMSY_SELECT_CLIENT_KEYS_TYPE),
      (max_key, _WHIMSY_SELECT_MAX_KEY_TYPE),
      (server_state, _WHIMSY_SELECT_SERVER_STATE_TYPE),
      select_fn,
  ]


def create_whimsy_federated_select_expected_result():
  """Constructs the expected result of the `whimsy` `federated_select`."""
  results = []
  for _ in range(_WHIMSY_SELECT_NUM_CLIENTS):
    result = [('abc', 0), ('abc', 1), ('abc', 2)]
    element_spec = computation_types.StructType(
        [(None, tf.string), (None, tf.int32)]
    )
    results.append(
        tensorflow_utils.make_data_set_from_elements(None, result, element_spec)
    )
  return results


def create_whimsy_intrinsic_def_federated_sum():
  value = intrinsic_defs.FEDERATED_SUM
  type_signature = computation_types.FunctionType(
      computation_types.at_clients(tf.float32),
      computation_types.at_server(tf.float32),
  )
  return value, type_signature


def create_whimsy_intrinsic_def_federated_value_at_clients():
  value = intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS
  type_signature = computation_types.FunctionType(
      tf.float32, computation_types.at_clients(tf.float32, all_equal=True)
  )
  return value, type_signature


def create_whimsy_intrinsic_def_federated_value_at_server():
  value = intrinsic_defs.FEDERATED_VALUE_AT_SERVER
  type_signature = computation_types.FunctionType(
      tf.float32, computation_types.at_server(tf.float32)
  )
  return value, type_signature


def create_whimsy_intrinsic_def_federated_weighted_mean():
  value = intrinsic_defs.FEDERATED_WEIGHTED_MEAN
  type_signature = computation_types.FunctionType(
      [
          computation_types.at_clients(tf.float32),
          computation_types.at_clients(tf.float32),
      ],
      computation_types.at_server(tf.float32),
  )
  return value, type_signature


def create_whimsy_intrinsic_def_federated_zip_at_clients():
  value = intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS
  type_signature = computation_types.FunctionType(
      [
          computation_types.at_clients(tf.float32),
          computation_types.at_clients(tf.float32),
      ],
      computation_types.at_clients([tf.float32, tf.float32]),
  )
  return value, type_signature


def create_whimsy_intrinsic_def_federated_zip_at_server():
  value = intrinsic_defs.FEDERATED_ZIP_AT_SERVER
  type_signature = computation_types.FunctionType(
      [
          computation_types.at_server(tf.float32),
          computation_types.at_server(tf.float32),
      ],
      computation_types.at_server([tf.float32, tf.float32]),
  )
  return value, type_signature


def create_whimsy_placement_literal():
  """Returns a `placements.PlacementLiteral` and type."""
  value = placements.SERVER
  type_signature = computation_types.PlacementType()
  return value, type_signature


def create_whimsy_computation_call():
  """Returns a call computation and type."""
  fn, fn_type = create_whimsy_computation_tensorflow_constant()
  type_signature = fn_type.result
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      call=pb.Call(function=fn),
  )
  return value, type_signature


def create_whimsy_computation_intrinsic():
  """Returns a intrinsic computation and type."""
  intrinsic_def, type_signature = (
      create_whimsy_intrinsic_def_federated_eval_at_server()
  )
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      intrinsic=pb.Intrinsic(uri=intrinsic_def.uri),
  )
  return value, type_signature


def create_whimsy_computation_lambda_empty():
  """Returns a lambda computation and type `( -> <>)`."""
  value = computation_factory.create_lambda_empty_struct()
  type_signature = computation_types.FunctionType(None, [])
  return value, type_signature


def create_whimsy_computation_lambda_identity():
  """Returns a lambda computation and type `(float32 -> float32)`."""
  tensor_type = computation_types.TensorType(tf.float32)
  value = computation_factory.create_lambda_identity(tensor_type)
  type_signature = computation_types.FunctionType(tensor_type, tensor_type)
  return value, type_signature


def create_whimsy_computation_placement():
  """Returns a placement computation and type."""
  placement_literal, type_signature = create_whimsy_placement_literal()
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      placement=pb.Placement(uri=placement_literal.uri),
  )
  return value, type_signature


def create_whimsy_computation_reference():
  """Returns a reference computation and type."""
  type_signature = computation_types.TensorType(tf.float32)
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      reference=pb.Reference(name='a'),
  )
  return value, type_signature


def create_whimsy_computation_selection():
  """Returns a selection computation and type."""
  source, source_type = create_whimsy_computation_tuple()
  type_signature = source_type[0]
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      selection=pb.Selection(source=source, index=0),
  )
  return value, type_signature


def create_whimsy_computation_tensorflow_add():
  """Returns a tensorflow computation and type.

  `(<float32,float32> -> float32)`
  """
  type_spec = tf.float32

  with tf.Graph().as_default() as graph:
    parameter_1_value, parameter_1_binding = (
        tensorflow_utils.stamp_parameter_in_graph('x', type_spec, graph)
    )
    parameter_2_value, parameter_2_binding = (
        tensorflow_utils.stamp_parameter_in_graph('y', type_spec, graph)
    )
    result_value = tf.add(parameter_1_value, parameter_2_value)
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result_value, graph
    )

  parameter_type = computation_types.StructType([type_spec, type_spec])
  type_signature = computation_types.FunctionType(parameter_type, result_type)
  struct_binding = pb.TensorFlow.StructBinding(
      element=[parameter_1_binding, parameter_2_binding]
  )
  parameter_binding = pb.TensorFlow.Binding(struct=struct_binding)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=parameter_binding,
      result=result_binding,
  )
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow,
  )
  return value, type_signature


def create_whimsy_computation_tensorflow_constant():
  """Returns a tensorflow computation and type `( -> float32)`."""
  value = 10.0
  tensor_type = computation_types.TensorType(tf.float32)
  value, type_signature = tensorflow_computation_factory.create_constant(
      value, tensor_type
  )
  return value, type_signature


def create_whimsy_computation_tensorflow_empty():
  """Returns a tensorflow computation and type `( -> <>)`."""
  value, type_signature = tensorflow_computation_factory.create_empty_tuple()
  return value, type_signature


def create_whimsy_computation_tensorflow_identity(arg_type=tf.float32):
  """Returns a tensorflow computation and type `(float32 -> float32)`."""
  value, type_signature = tensorflow_computation_factory.create_identity(
      computation_types.to_type(arg_type)
  )
  return value, type_signature


def create_whimsy_computation_tensorflow_random():
  """Returns a tensorflow computation and type `( -> float32)`."""

  with tf.Graph().as_default() as graph:
    result = tf.random.normal([])
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result, graph
    )

  type_signature = computation_types.FunctionType(None, result_type)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=None,
      result=result_binding,
  )
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow,
  )
  return value, type_signature


def create_whimsy_computation_tensorflow_tuple():
  """Returns a tensorflow computation and type.

  `( -> <('a', float32), ('b', float32), ('c', float32)>)`
  """
  value = 10.0

  with tf.Graph().as_default() as graph:
    names = ['a', 'b', 'c']
    result = structure.Struct((n, tf.constant(value)) for n in names)
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        result, graph
    )

  type_signature = computation_types.FunctionType(None, result_type)
  tensorflow = pb.TensorFlow(
      graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
      parameter=None,
      result=result_binding,
  )
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      tensorflow=tensorflow,
  )
  return value, type_signature


def create_whimsy_computation_tuple():
  """Returns a tuple computation and type."""
  names = ['a', 'b', 'c']
  fn, fn_type = create_whimsy_computation_tensorflow_constant()
  element_value = pb.Computation(
      type=type_serialization.serialize_type(fn_type), call=pb.Call(function=fn)
  )
  element_type = fn_type.result
  elements = [pb.Struct.Element(name=n, value=element_value) for n in names]
  type_signature = computation_types.StructType(
      (n, element_type) for n in names
  )
  value = pb.Computation(
      type=type_serialization.serialize_type(type_signature),
      struct=pb.Struct(element=elements),
  )
  return value, type_signature


def create_whimsy_value_at_clients(number_of_clients: int = 3):
  """Returns a Python value and federated type at clients."""
  value = [float(x) for x in range(10, number_of_clients + 10)]
  type_signature = computation_types.at_clients(tf.float32)
  return value, type_signature


def create_whimsy_value_at_clients_all_equal():
  """Returns a Python value and federated type at clients and all equal."""
  value = 10.0
  type_signature = computation_types.at_clients(tf.float32, all_equal=True)
  return value, type_signature


def create_whimsy_value_at_server():
  """Returns a Python value and federated type at server."""
  value = 10.0
  type_signature = computation_types.at_server(tf.float32)
  return value, type_signature


def create_whimsy_value_unplaced():
  """Returns a Python value and unplaced type."""
  value = 10.0
  type_signature = computation_types.TensorType(tf.float32)
  return value, type_signature


def _return_assertion_error():
  return AssertionError


class RaisingExecutor(eager_tf_executor.EagerTFExecutor):
  """An executor which can be configured to raise on `create_value`."""

  def __init__(self, error_fn=_return_assertion_error):
    self._should_raise = True
    self._error_fn = error_fn
    super().__init__()

  def stop_raising(self):
    self._should_raise = False

  async def create_value(self, *args, **kwargs):
    if self._should_raise:
      raise self._error_fn()
    return await super().create_value(*args, **kwargs)
