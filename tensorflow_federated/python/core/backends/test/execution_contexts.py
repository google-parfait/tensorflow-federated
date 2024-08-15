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

from tensorflow_federated.python.core.backends.native import compiler as native_compiler
from tensorflow_federated.python.core.backends.test import compiler as test_compiler
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.execution_contexts import async_execution_context
from tensorflow_federated.python.core.impl.execution_contexts import sync_execution_context
from tensorflow_federated.python.core.impl.executor_stacks import executor_factory


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
      transform_args=tensorflow_computation.transform_args,
      transform_result=tensorflow_computation.transform_result,
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
      transform_args=tensorflow_computation.transform_args,
      transform_result=tensorflow_computation.transform_result,
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
