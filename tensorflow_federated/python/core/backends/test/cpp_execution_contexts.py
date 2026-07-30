# Copyright 2021, The TensorFlow Federated Authors.
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


import federated_language

from tensorflow_federated.python.core.backends.native import compiler as native_compiler
from tensorflow_federated.python.core.backends.test import compiler as test_compiler
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_executor_bindings
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.executor_stacks import cpp_executor_factory
from tensorflow_federated.python.core.impl.executors import executor_bindings


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


def create_async_test_cpp_execution_context(
    *,
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
) -> federated_language.framework.AsyncExecutionContext:
  """Creates an async execution context for local testing of computations.

  Test execution contexts are useful for simulating the behavior of secure
  aggregation (e.g. `secure_sum`, `secure_modular_sum`) without actually
  performing secure aggregation.

  Args:
    default_num_clients: The number of clients to be used if the number of
      clients cannot be inferred from the arguments to a computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the C++ runtime. If nonpositive, there is no
      limit.
    stream_structs: The flag to enable decomposing and streaming struct values.

  Returns:
    An instance of `AsyncExecutionContext` for local testing of computations.

  Raises:
    ValueError: If invalid parameters are provided to either the C++ or Python
      runtimes, as detailed above.
  """
  del stream_structs  # Unused.

  def _compile(comp):
    # Compile secure_sum and secure_sum_bitwidth intrinsics to insecure
    # TensorFlow computations for testing purposes.
    comp = test_compiler.replace_secure_intrinsics_with_bodies(comp)
    comp = native_compiler.desugar_and_transform_to_native(comp)
    return comp

  factory = cpp_executor_factory.local_cpp_executor_factory(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      leaf_executor_fn=_create_tensorflow_backend_execution_stack,
  )
  context = federated_language.framework.AsyncExecutionContext(
      executor_fn=factory,
      compiler_fn=_compile,
      transform_args=tensorflow_computation.transform_args,
      transform_result=tensorflow_computation.transform_result,
  )
  return context


def set_async_test_cpp_execution_context(
    *,
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
) -> None:
  """Sets an async execution context for local testing of computations.

  Test execution contexts are useful for simulating the behavior of secure
  aggregation (e.g. `secure_sum`, `secure_modular_sum`) without actually
  performing secure aggregation.

  Args:
    default_num_clients: The number of clients to be used if the number of
      clients cannot be inferred from the arguments to a computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the C++ runtime. If nonpositive, there is no
      limit.
    stream_structs: The flag to enable decomposing and streaming struct values.

  Raises:
    ValueError: If invalid parameters are provided to either the C++ or Python
      runtimes, as detailed above.
  """
  context = create_async_test_cpp_execution_context(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      stream_structs=stream_structs,
  )
  federated_language.framework.get_context_stack().set_default_context(context)


def create_sync_test_cpp_execution_context(
    *,
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
) -> federated_language.framework.SyncExecutionContext:
  """Creates an execution context for local testing of computations.

  Test execution contexts are useful for simulating the behavior of secure
  aggregation (e.g. `secure_sum`, `secure_modular_sum`) without actually
  performing secure aggregation.

  Args:
    default_num_clients: The number of clients to be used if the number of
      clients cannot be inferred from the arguments to a computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the C++ runtime. If nonpositive, there is no
      limit.
    stream_structs: The flag to enable decomposing and streaming struct values.

  Returns:
    An execution context for local testing of computations.

  Raises:
    ValueError: If invalid parameters are provided to either the C++ or Python
      runtimes, as detailed above.
  """
  del stream_structs  # Unused.

  def _compile(comp):
    # Compile secure_sum and secure_sum_bitwidth intrinsics to insecure
    # TensorFlow computations for testing purposes.
    comp = test_compiler.replace_secure_intrinsics_with_bodies(comp)
    comp = native_compiler.desugar_and_transform_to_native(comp)
    return comp

  factory = cpp_executor_factory.local_cpp_executor_factory(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      leaf_executor_fn=_create_tensorflow_backend_execution_stack,
  )
  context = federated_language.framework.SyncExecutionContext(
      executor_fn=factory,
      compiler_fn=_compile,
      transform_args=tensorflow_computation.transform_args,
      transform_result=tensorflow_computation.transform_result,
  )
  return context


def set_sync_test_cpp_execution_context(
    *,
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
) -> None:
  """Sets an execution context for local testing of computations.

  Test execution contexts are useful for simulating the behavior of secure
  aggregation (e.g. `secure_sum`, `secure_modular_sum`) without actually
  performing secure aggregation.

  Args:
    default_num_clients: The number of clients to be used if the number of
      clients cannot be inferred from the arguments to a computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the C++ runtime. If nonpositive, there is no
      limit.
    stream_structs: The flag to enable decomposing and streaming struct values.

  Raises:
    ValueError: If invalid parameters are provided to either the C++ or Python
      runtimes, as detailed above.
  """
  context = create_sync_test_cpp_execution_context(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      stream_structs=stream_structs,
  )
  federated_language.framework.get_context_stack().set_default_context(context)
