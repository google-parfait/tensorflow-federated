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
"""Execution contexts for the test backend."""

from typing import Optional

import attr
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import compiler as native_compiler
from tensorflow_federated.python.core.backends.test import compiler as test_compiler
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.execution_contexts import async_execution_context
from tensorflow_federated.python.core.impl.execution_contexts import sync_execution_context
from tensorflow_federated.python.core.impl.executor_stacks import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_factory as executor_factory_interface


def create_async_test_cpp_execution_context(
    *,
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
) -> async_execution_context.AsyncExecutionContext:
  """Creates an execution context that executes computations locally."""
  factory = executor_factory.local_cpp_executor_factory(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      stream_structs=stream_structs,
  )

  def _compile(comp):
    # Compile secure_sum and secure_sum_bitwidth intrinsics to insecure
    # TensorFlow computations for testing purposes.
    comp = test_compiler.replace_secure_intrinsics_with_bodies(comp)
    comp = native_compiler.desugar_and_transform_to_native(comp)
    return comp

  return async_execution_context.AsyncExecutionContext(
      executor_fn=factory,
      compiler_fn=_compile,
  )


def set_async_test_cpp_execution_context(
    *,
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
) -> None:
  """Sets an execution context that executes computations locally."""
  context = create_async_test_cpp_execution_context(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      stream_structs=stream_structs,
  )
  context_stack_impl.context_stack.set_default_context(context)


def create_sync_test_cpp_execution_context(
    *,
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
) -> sync_execution_context.SyncExecutionContext:
  """Creates an execution context that executes computations locally."""
  factory = executor_factory.local_cpp_executor_factory(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      stream_structs=stream_structs,
  )

  def _compile(comp):
    # Compile secure_sum and secure_sum_bitwidth intrinsics to insecure
    # TensorFlow computations for testing purposes.
    comp = test_compiler.replace_secure_intrinsics_with_bodies(comp)
    comp = native_compiler.desugar_and_transform_to_native(comp)
    return comp

  return sync_execution_context.SyncExecutionContext(
      executor_fn=factory,
      compiler_fn=_compile,
  )


def set_sync_test_cpp_execution_context(
    *,
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
) -> None:
  """Sets an execution context that executes computations locally."""
  context = create_sync_test_cpp_execution_context(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      stream_structs=stream_structs,
  )
  context_stack_impl.context_stack.set_default_context(context)


@attr.s
class DistributedConfiguration:
  """Class for distributed runtime configuration."""

  server_mesh: Optional[tf.experimental.dtensor.Mesh] = attr.ib(default=None)
  client_mesh: Optional[tf.experimental.dtensor.Mesh] = attr.ib(default=None)

  def __attrs_post_init__(self):
    if self.server_mesh is None and self.client_mesh is None:
      raise ValueError(
          "Both server side and client side mesh are unspecified"
          " in distributed configuration."
      )


def _get_distributed_executor_factory(
    distributed_config: DistributedConfiguration,
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
) -> executor_factory_interface.ExecutorFactory:
  """Return an execution factory which constructs DTensor based executor."""
  return executor_factory.local_cpp_executor_factory(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      stream_structs=stream_structs,
      server_mesh=distributed_config.server_mesh,
      client_mesh=distributed_config.client_mesh,
  )


def create_async_experimental_distributed_cpp_execution_context(
    distributed_config: DistributedConfiguration,
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
) -> async_execution_context.AsyncExecutionContext:
  """Creates a local async execution context backed by TFF-C++ runtime.

  When using this context, the local sequence reductions assumed to expressed
  using tff.sequence_reduce. Iterating over dataset or dataset.reduce inside
  TF graph are currently *not* supported.

  Args:
    distributed_config: A runtime configuration for running TF computation in a
      distributed manner. A server side and/or client side mesh can be supplied
      in the configuration if TF computation should be executed with DTensor
      executor.
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the C++ runtime. If nonpositive, there is no
      limit.
    stream_structs: The flag to enable decomposing and streaming struct values.

  Returns:
    An instance of `context_base.AsyncContext` representing the TFF-C++ runtime.
  """

  def _compile(comp):
    # Compile secure_sum and secure_sum_bitwidth intrinsics to insecure
    # TensorFlow computations for testing purposes.
    comp = test_compiler.replace_secure_intrinsics_with_bodies(comp)
    comp = native_compiler.desugar_and_transform_to_native(comp)
    return comp

  return async_execution_context.AsyncExecutionContext(
      executor_fn=_get_distributed_executor_factory(
          distributed_config,
          default_num_clients,
          max_concurrent_computation_calls,
          stream_structs,
      ),
      compiler_fn=_compile,
  )


def set_async_experimental_distributed_cpp_execution_context(
    distributed_config: DistributedConfiguration,
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
) -> None:
  """Sets a local execution context backed by TFF-C++ runtime.

  Args:
    distributed_config: A runtime configuration for running TF computation in a
      distributed manner. A server side and/or client side mesh can be supplied
      in the configuration if TF computation should be executed with DTensor
      executor.
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the CPP runtime. If nonpositive, there is no
      limit.
    stream_structs: The flag to enable decomposing and streaming struct values.
  """
  context = create_async_experimental_distributed_cpp_execution_context(
      distributed_config=distributed_config,
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      stream_structs=stream_structs,
  )
  context_stack_impl.context_stack.set_default_context(context)


def create_sync_experimental_distributed_cpp_execution_context(
    distributed_config: DistributedConfiguration,
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
) -> sync_execution_context.SyncExecutionContext:
  """Creates a local execution context backed by TFF-C++ runtime.

  Args:
    distributed_config: A runtime configuration for running TF computation in a
      distributed manner. A server side and/or client side mesh can be supplied
      in the configuration if TF computation should be executed with DTensor
      executor.
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the CPP runtime. If nonpositive, there is no
      limit.
    stream_structs: The flag to enable decomposing and streaming struct values.

  Returns:
    An instance of `tff.framework.SyncContext` representing the TFF-C++ runtime.
  """

  def _compile(comp):
    # Compile secure_sum and secure_sum_bitwidth intrinsics to insecure
    # TensorFlow computations for testing purposes.
    comp = test_compiler.replace_secure_intrinsics_with_bodies(comp)
    comp = native_compiler.desugar_and_transform_to_native(comp)
    return comp

  return sync_execution_context.SyncExecutionContext(
      executor_fn=_get_distributed_executor_factory(
          distributed_config,
          default_num_clients,
          max_concurrent_computation_calls,
          stream_structs,
      ),
      compiler_fn=_compile,
  )


def set_sync_experimental_distributed_cpp_execution_context(
    distributed_config: DistributedConfiguration,
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
) -> None:
  """Sets a default local sync execution context backed by TFF-C++ runtime.

  Args:
    distributed_config: A runtime configuration for running TF computation in a
      distributed manner. A server side and/or client side mesh can be supplied
      in the configuration if TF computation should be executed with DTensor
      executor.
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the C++ runtime. If nonpositive, there is no
      limit.
    stream_structs: The flag to enable decomposing and streaming struct values.
  """
  context = create_sync_experimental_distributed_cpp_execution_context(
      distributed_config=distributed_config,
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      stream_structs=stream_structs,
  )
  context_stack_impl.context_stack.set_default_context(context)
