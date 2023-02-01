# Copyright 2022, The TensorFlow Federated Authors.
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
"""Execution contexts for the native backend."""

from collections.abc import Sequence

from tensorflow_federated.python.core.backends.native import compiler
from tensorflow_federated.python.core.impl.context_stack import set_default_context
from tensorflow_federated.python.core.impl.execution_contexts import async_execution_context
from tensorflow_federated.python.core.impl.execution_contexts import sync_execution_context
from tensorflow_federated.python.core.impl.executor_stacks import cpp_executor_factory
from tensorflow_federated.python.core.impl.executors import executor_bindings


def create_sync_local_cpp_execution_context(
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
) -> sync_execution_context.SyncExecutionContext:
  """Creates a local execution context backed by TFF-C++ runtime.

  Args:
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
  del stream_structs  # Unused.
  factory = cpp_executor_factory.local_cpp_executor_factory(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
  )
  context = sync_execution_context.SyncExecutionContext(
      executor_fn=factory, compiler_fn=compiler.desugar_and_transform_to_native
  )
  return context


def set_sync_local_cpp_execution_context(
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
) -> None:
  context = create_sync_local_cpp_execution_context(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      stream_structs=stream_structs,
  )
  set_default_context.set_default_context(context)


def create_async_local_cpp_execution_context(
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
) -> async_execution_context.AsyncExecutionContext:
  """Creates a local async execution context backed by TFF-C++ runtime.

  Args:
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
  del stream_structs  # Unused.
  factory = cpp_executor_factory.local_cpp_executor_factory(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
  )
  context = async_execution_context.AsyncExecutionContext(
      executor_fn=factory, compiler_fn=compiler.desugar_and_transform_to_native
  )
  return context


def set_async_local_cpp_execution_context(
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
) -> None:
  """Sets a local execution context backed by TFF-C++ runtime.

  Args:
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the CPP runtime. If nonpositive, there is no
      limit.
    stream_structs: The flag to enable decomposing and streaming struct values.
  """
  context = create_async_local_cpp_execution_context(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      stream_structs=stream_structs,
  )
  set_default_context.set_default_context(context)


def set_remote_cpp_execution_context(
    channels: Sequence[executor_bindings.GRPCChannel],
    default_num_clients: int = 0,
):
  context = create_remote_cpp_execution_context(
      channels=channels, default_num_clients=default_num_clients
  )
  set_default_context.set_default_context(context)


def create_remote_cpp_execution_context(
    channels: Sequence[executor_bindings.GRPCChannel],
    default_num_clients: int = 0,
) -> sync_execution_context.SyncExecutionContext:
  """Creates a remote execution context backed by TFF-C++ runtime."""
  factory = cpp_executor_factory.remote_cpp_executor_factory(
      channels=channels, default_num_clients=default_num_clients
  )
  context = sync_execution_context.SyncExecutionContext(
      executor_fn=factory, compiler_fn=compiler.desugar_and_transform_to_native
  )
  return context


def create_remote_async_cpp_execution_context(
    channels: Sequence[executor_bindings.GRPCChannel],
    default_num_clients: int = 0,
) -> async_execution_context.AsyncExecutionContext:
  """Creates a remote execution context backed by TFF-C++ runtime."""
  factory = cpp_executor_factory.remote_cpp_executor_factory(
      channels=channels, default_num_clients=default_num_clients
  )
  context = async_execution_context.AsyncExecutionContext(
      executor_fn=factory, compiler_fn=compiler.desugar_and_transform_to_native
  )
  return context
