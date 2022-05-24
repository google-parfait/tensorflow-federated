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

from typing import Optional, Sequence

from tensorflow_federated.python.core.backends.native import compiler
from tensorflow_federated.python.core.impl.context_stack import set_default_context
from tensorflow_federated.python.core.impl.execution_contexts import cpp_async_execution_context
from tensorflow_federated.python.core.impl.execution_contexts import cpp_sync_execution_context
from tensorflow_federated.python.core.impl.executor_stacks import cpp_executor_factory
from tensorflow_federated.python.core.impl.executors import executor_bindings


def set_local_cpp_execution_context(
    default_num_clients: int = 0,
    max_concurrent_computation_calls: Optional[int] = None):
  context = create_local_cpp_execution_context(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls)
  set_default_context.set_default_context(context)


def create_local_cpp_execution_context(
    default_num_clients: int = 0,
    max_concurrent_computation_calls: Optional[int] = None):
  """Creates a local execution context backed by TFF-C++ runtime.

  Args:
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the CPP runtime. If `None`, there is no limit.

  Returns:
    An instance of `context_base.Context` representing the TFF-C++ runtime.
  """
  factory = cpp_executor_factory.local_cpp_executor_factory(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls)
  context = cpp_sync_execution_context.SyncSerializeAndExecuteCPPContext(
      factory=factory, compiler_fn=compiler.desugar_and_transform_to_native)
  return context


def create_local_async_cpp_execution_context(
    default_num_clients: int = 0,
    max_concurrent_computation_calls: Optional[int] = None):
  """Creates a local async execution context backed by TFF-C++ runtime.

  Args:
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the CPP runtime. If `None`, there is no limit.

  Returns:
    An instance of `context_base.Context` representing the TFF-C++ runtime.
  """
  factory = cpp_executor_factory.local_cpp_executor_factory(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls)
  context = cpp_async_execution_context.AsyncSerializeAndExecuteCPPContext(
      factory=factory, compiler_fn=compiler.desugar_and_transform_to_native)
  return context


def set_local_async_cpp_execution_context(
    default_num_clients: int = 0,
    max_concurrent_computation_calls: Optional[int] = None):
  """Sets a local execution context backed by TFF-C++ runtime.

  Args:
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the CPP runtime. If `None`, there is no limit.
  """
  context = create_local_async_cpp_execution_context(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls)
  set_default_context.set_default_context(context)


def set_remote_cpp_execution_context(
    channels: Sequence[executor_bindings.GRPCChannel],
    default_num_clients: int = 0):
  context = create_remote_cpp_execution_context(
      channels=channels, default_num_clients=default_num_clients)
  set_default_context.set_default_context(context)


def create_remote_cpp_execution_context(
    channels: Sequence[executor_bindings.GRPCChannel],
    default_num_clients: int = 0
) -> cpp_sync_execution_context.SyncSerializeAndExecuteCPPContext:
  """Creates a remote execution context backed by TFF-C++ runtime."""
  factory = cpp_executor_factory.remote_cpp_executor_factory(
      channels=channels, default_num_clients=default_num_clients)
  context = cpp_sync_execution_context.SyncSerializeAndExecuteCPPContext(
      factory=factory, compiler_fn=compiler.desugar_and_transform_to_native)
  return context


def create_remote_async_cpp_execution_context(
    channels: Sequence[executor_bindings.GRPCChannel],
    default_num_clients: int = 0
) -> cpp_async_execution_context.AsyncSerializeAndExecuteCPPContext:
  """Creates a remote execution context backed by TFF-C++ runtime."""
  factory = cpp_executor_factory.remote_cpp_executor_factory(
      channels=channels, default_num_clients=default_num_clients)
  context = cpp_async_execution_context.AsyncSerializeAndExecuteCPPContext(
      factory=factory, compiler_fn=compiler.desugar_and_transform_to_native)
  return context
