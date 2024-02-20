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
import dataclasses
from typing import Optional

import tensorflow as tf

from tensorflow_federated.python.core.backends.native import compiler
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_executor_bindings
from tensorflow_federated.python.core.impl.context_stack import set_default_context
from tensorflow_federated.python.core.impl.execution_contexts import async_execution_context
from tensorflow_federated.python.core.impl.execution_contexts import sync_execution_context
from tensorflow_federated.python.core.impl.executor_stacks import cpp_executor_factory
from tensorflow_federated.python.core.impl.executors import executor_bindings
from tensorflow_federated.python.core.impl.executors import executor_factory


@dataclasses.dataclass(frozen=True)
class DistributedConfiguration:
  """Class for distributed runtime configuration."""

  server_mesh: tf.experimental.dtensor.Mesh = None
  client_mesh: tf.experimental.dtensor.Mesh = None

  def __post_init__(self):
    if self.server_mesh is None and self.client_mesh is None:
      raise ValueError(
          "Both server side and client side mesh are unspecified"
          " in distributed configuration."
      )


def _create_tensorflow_backend_execution_stack(
    max_concurrent_computation_calls: int,
) -> executor_bindings.Executor:
  """Returns a leaf executor for Tensorflow based executor."""
  tensorflow_executor = tensorflow_executor_bindings.create_tensorflow_executor(
      max_concurrent_computation_calls
  )
  reference_resolving_executor = (
      executor_bindings.create_reference_resolving_executor(tensorflow_executor)
  )
  return executor_bindings.create_sequence_executor(
      reference_resolving_executor
  )


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
      leaf_executor_fn=_create_tensorflow_backend_execution_stack,
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
  """Sets a default local sync execution context backed by TFF-C++ runtime.

  Args:
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the C++ runtime. If nonpositive, there is no
      limit.
    stream_structs: The flag to enable decomposing and streaming struct values.
  """
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
      leaf_executor_fn=_create_tensorflow_backend_execution_stack,
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


def _create_dtensor_executor(mesh: tf.experimental.dtensor.Mesh):
  """Return leaf_executor_fn for DTensor based executor."""

  def _dtensor_executor_fn(max_concurrent_computation_calls: int):
    with tf.experimental.dtensor.run_on(mesh):
      return executor_bindings.create_sequence_executor(
          executor_bindings.create_reference_resolving_executor(
              tensorflow_executor_bindings.create_dtensor_executor(
                  tf.experimental.dtensor.device_name(),
                  mesh.to_string(),
                  max_concurrent_computation_calls,
              )
          )
      )

  return _dtensor_executor_fn


def _get_distributed_executor_factory(
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    distributed_config: Optional[DistributedConfiguration] = None,
) -> executor_factory.ExecutorFactory:
  """Return an execution factory which constructs DTensor based executor."""
  server_leaf_executor_fn = _create_tensorflow_backend_execution_stack
  client_leaf_executor_fn = _create_tensorflow_backend_execution_stack
  if distributed_config is None:
    raise ValueError("Distributed configuration is unspecified.")

  if distributed_config.server_mesh is not None:
    server_leaf_executor_fn = _create_dtensor_executor(
        distributed_config.server_mesh
    )
  if distributed_config.client_mesh is not None:
    client_leaf_executor_fn = _create_dtensor_executor(
        distributed_config.client_mesh
    )
  return cpp_executor_factory.local_cpp_executor_factory(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      leaf_executor_fn=server_leaf_executor_fn,
      client_leaf_executor_fn=client_leaf_executor_fn,
  )


def create_sync_experimental_distributed_cpp_execution_context(
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
    distributed_config: Optional[DistributedConfiguration] = None,
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
    distributed_config: A runtime configuration for running TF computation in a
      distributed manner. A server side and/or client side mesh can be supplied
      in the configuration if TF computation should be executed with DTensor
      executor.

  Returns:
    An instance of `tff.framework.SyncContext` representing the TFF-C++ runtime.
  """
  del stream_structs  # Unused.

  context = sync_execution_context.SyncExecutionContext(
      executor_fn=_get_distributed_executor_factory(
          default_num_clients,
          max_concurrent_computation_calls,
          distributed_config,
      ),
      compiler_fn=compiler.desugar_and_transform_to_native,
  )
  return context


def set_sync_experimental_distributed_cpp_execution_context(
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
    distributed_config: Optional[DistributedConfiguration] = None,
) -> None:
  """Sets a default local sync execution context backed by TFF-C++ runtime.

  Args:
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the C++ runtime. If nonpositive, there is no
      limit.
    stream_structs: The flag to enable decomposing and streaming struct values.
    distributed_config: A runtime configuration for running TF computation in a
      distributed manner. A server side and/or client side mesh can be supplied
      in the configuration if TF computation should be executed with DTensor
      executor.
  """
  context = create_sync_experimental_distributed_cpp_execution_context(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      stream_structs=stream_structs,
      distributed_config=distributed_config,
  )
  set_default_context.set_default_context(context)


def create_async_experimental_distributed_cpp_execution_context(
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
    distributed_config: Optional[DistributedConfiguration] = None,
) -> async_execution_context.AsyncExecutionContext:
  """Creates a local async execution context backed by TFF-C++ runtime.

  When using this context, the local sequence reductions assumed to expressed
  using tff.sequence_reduce. Iterating over dataset or dataset.reduce inside
  TF graph are currently *not* supported.

  Args:
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the C++ runtime. If nonpositive, there is no
      limit.
    stream_structs: The flag to enable decomposing and streaming struct values.
    distributed_config: A runtime configuration for running TF computation in a
      distributed manner. A server side and/or client side mesh can be supplied
      in the configuration if TF computation should be executed with DTensor
      executor.

  Returns:
    An instance of `context_base.AsyncContext` representing the TFF-C++ runtime.
  """
  del stream_structs  # Unused.

  context = async_execution_context.AsyncExecutionContext(
      executor_fn=_get_distributed_executor_factory(
          default_num_clients,
          max_concurrent_computation_calls,
          distributed_config,
      ),
      compiler_fn=compiler.desugar_and_transform_to_native,
  )
  return context


def set_async_experimental_distributed_cpp_execution_context(
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
    distributed_config: Optional[DistributedConfiguration] = None,
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
    distributed_config: A runtime configuration for running TF computation in a
      distributed manner. A server side and/or client side mesh can be supplied
      in the configuration if TF computation should be executed with DTensor
      executor.
  """
  context = create_async_experimental_distributed_cpp_execution_context(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      stream_structs=stream_structs,
      distributed_config=distributed_config,
  )
  set_default_context.set_default_context(context)


def create_sync_remote_cpp_execution_context(
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


def set_sync_remote_cpp_execution_context(
    channels: Sequence[executor_bindings.GRPCChannel],
    default_num_clients: int = 0,
):
  context = create_sync_remote_cpp_execution_context(
      channels=channels, default_num_clients=default_num_clients
  )
  set_default_context.set_default_context(context)


def create_async_remote_cpp_execution_context(
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
