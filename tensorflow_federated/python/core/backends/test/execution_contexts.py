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

from tensorflow_federated.python.core.impl.compiler import intrinsic_reductions
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.execution_contexts import synchronous_execution_context
from tensorflow_federated.python.core.impl.executors import executor_stacks
from tensorflow_federated.python.core.impl.wrappers import computation_wrapper_instances


def create_test_execution_context(default_num_clients=0, clients_per_thread=1):
  """Creates an execution context that executes computations locally."""
  factory = executor_stacks.local_executor_factory(
      default_num_clients=default_num_clients,
      clients_per_thread=clients_per_thread)

  def compiler(comp):
    # Compile secure_sum and secure_sum_bitwidth intrinsics to insecure
    # TensorFlow computations for testing purposes.
    replaced_intrinsic_bodies, _ = intrinsic_reductions.replace_secure_intrinsics_with_insecure_bodies(
        comp.to_building_block())
    return computation_wrapper_instances.building_block_to_computation(
        replaced_intrinsic_bodies)

  return synchronous_execution_context.ExecutionContext(
      executor_fn=factory, compiler_fn=compiler)


def set_test_execution_context(default_num_clients=0, clients_per_thread=1):
  """Sets an execution context that executes computations locally."""
  context = create_test_execution_context(
      default_num_clients=default_num_clients,
      clients_per_thread=clients_per_thread)
  context_stack_impl.context_stack.set_default_context(context)
