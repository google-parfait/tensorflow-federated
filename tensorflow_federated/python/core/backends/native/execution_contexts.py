# Copyright 2020, The TensorFlow Federated Authors.
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

from tensorflow_federated.python.core.backends.native import compiler
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.executors import execution_context
from tensorflow_federated.python.core.impl.executors import executor_stacks


def set_local_execution_context(num_clients=None,
                                max_fanout=100,
                                num_client_executors=32,
                                server_tf_device=None,
                                client_tf_devices=tuple()):
  """Sets an execution context that executes computations locally."""
  factory = executor_stacks.local_executor_factory(
      num_clients=num_clients,
      max_fanout=max_fanout,
      num_client_executors=num_client_executors,
      server_tf_device=server_tf_device,
      client_tf_devices=client_tf_devices)
  context = execution_context.ExecutionContext(
      executor_fn=factory, compiler_fn=compiler.transform_to_native_form)
  context_stack_impl.context_stack.set_default_context(context)
