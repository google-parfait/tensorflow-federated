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

from typing import Optional

from tensorflow_federated.python.core.backends.native import compiler as native_compiler
from tensorflow_federated.python.core.backends.test import compiler as test_compiler
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.execution_contexts import cpp_sync_execution_context
from tensorflow_federated.python.core.impl.executor_stacks import cpp_executor_factory


def create_test_cpp_execution_context(
    *,
    default_num_clients: int = 0,
    max_concurrent_computation_calls: Optional[int] = None
) -> context_base.Context:
  """Creates an execution context for local testing of computations.

  Test execution contexts are useful for simulating the behavior of secure
  aggregation (e.g. `secure_sum`, `secure_modular_sum`) without actually
  performing secure aggregation.

  Args:
    default_num_clients: The number of clients to be used if the number of
      clients cannot be inferred from the arguments to a computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls
      to a single computation in the C++ runtime. If `None`, there is no limit.
      This argument must not be provided if `use_cpp=False`.

  Returns:
    An execution context for local testing of computations.

  Raises:
    ValueError: If invalid parameters are provided to either the C++ or Python
      runtimes, as detailed above.
  """
  def _compile(comp):
    # Compile secure_sum and secure_sum_bitwidth intrinsics to insecure
    # TensorFlow computations for testing purposes.
    comp = test_compiler.replace_secure_intrinsics_with_bodies(comp)
    comp = native_compiler.desugar_and_transform_to_native(comp)
    return comp

  factory = cpp_executor_factory.local_cpp_executor_factory(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls)
  context = cpp_sync_execution_context.SyncSerializeAndExecuteCPPContext(
      factory=factory, compiler_fn=_compile)
  return context


def set_test_cpp_execution_context(
    *,
    default_num_clients: int = 0,
    max_concurrent_computation_calls: Optional[int] = None):
  """Sets an execution context for local testing of computations.

  Test execution contexts are useful for simulating the behavior of secure
  aggregation (e.g. `secure_sum`, `secure_modular_sum`) without actually
  performing secure aggregation.

  Args:
    default_num_clients: The number of clients to be used if the number of
      clients cannot be inferred from the arguments to a computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls
      to a single computation in the C++ runtime. If `None`, there is no limit.
      This argument must not be provided if `use_cpp=False`.

  Raises:
    ValueError: If invalid parameters are provided to either the C++ or Python
      runtimes, as detailed above.
  """
  context = create_test_cpp_execution_context(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls)
  context_stack_impl.context_stack.set_default_context(context)
