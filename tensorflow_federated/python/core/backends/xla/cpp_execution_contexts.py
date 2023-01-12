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
"""Execution contexts for the XLA backend."""

from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.context_stack import set_default_context
from tensorflow_federated.python.core.impl.execution_contexts import sync_execution_context
from tensorflow_federated.python.core.impl.executor_stacks import cpp_executor_factory
from tensorflow_federated.python.core.impl.executors import executor_bindings


def set_local_cpp_execution_context(default_num_clients: int = 0,
                                    max_concurrent_computation_calls: int = -1):
  context = create_local_cpp_execution_context(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls)
  set_default_context.set_default_context(context)


def create_local_cpp_execution_context(
    default_num_clients: int = 0, max_concurrent_computation_calls: int = -1):
  """Creates a local execution context backed by TFF-C++ runtime.

  Args:
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the CPP runtime. If nonpositive, there is no
      limit.

  Returns:
    An instance of `tff.framework.SyncContext` representing the TFF-C++ runtime.
  """

  def leaf_executor_fn():
    xla_executor_fn = executor_bindings.create_xla_executor()
    executor_bindings.create_sequence_executor(xla_executor_fn)

  factory = cpp_executor_factory.local_cpp_executor_factory(
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
      leaf_executor_fn=leaf_executor_fn)

  def compiler_fn(comp: computation_base.Computation):
    # TODO(b/255978089): Define compiler_fn - Integrate LocalComputationFactory
    # with intrinsic reductions
    del comp  # Unused.
    return None

  return sync_execution_context.SyncExecutionContext(
      executor_fn=factory, compiler_fn=compiler_fn
  )
