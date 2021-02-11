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


def create_local_execution_context(num_clients=None,
                                   max_fanout=100,
                                   clients_per_thread=1,
                                   server_tf_device=None,
                                   client_tf_devices=tuple(),
                                   reference_resolving_clients=False):
  """Creates an execution context that executes computations locally."""
  factory = executor_stacks.local_executor_factory(
      num_clients=num_clients,
      max_fanout=max_fanout,
      clients_per_thread=clients_per_thread,
      server_tf_device=server_tf_device,
      client_tf_devices=client_tf_devices,
      reference_resolving_clients=reference_resolving_clients)

  def _compiler(comp):
    native_form = compiler.transform_to_native_form(comp)
    if not reference_resolving_clients:
      return compiler.transform_mathematical_functions_to_tensorflow(
          native_form)
    return native_form

  return execution_context.ExecutionContext(
      executor_fn=factory, compiler_fn=_compiler)


def set_local_execution_context(num_clients=None,
                                max_fanout=100,
                                clients_per_thread=1,
                                server_tf_device=None,
                                client_tf_devices=tuple(),
                                reference_resolving_clients=False):
  """Sets an execution context that executes computations locally."""
  context = create_local_execution_context(
      num_clients=num_clients,
      max_fanout=max_fanout,
      clients_per_thread=clients_per_thread,
      server_tf_device=server_tf_device,
      client_tf_devices=client_tf_devices,
      reference_resolving_clients=reference_resolving_clients)
  context_stack_impl.context_stack.set_default_context(context)


def create_sizing_execution_context(num_clients: int = None,
                                    max_fanout: int = 100,
                                    clients_per_thread: int = 1):
  """Creates an execution context that executes computations locally."""
  factory = executor_stacks.sizing_executor_factory(
      num_clients=num_clients,
      max_fanout=max_fanout,
      clients_per_thread=clients_per_thread)
  return execution_context.ExecutionContext(
      executor_fn=factory, compiler_fn=compiler.transform_to_native_form)


def create_thread_debugging_execution_context(num_clients=None,
                                              clients_per_thread=1):
  """Creates a simple execution context that executes computations locally."""
  factory = executor_stacks.thread_debugging_executor_factory(
      num_clients=num_clients,
      clients_per_thread=clients_per_thread,
  )

  def _debug_compiler(comp):
    native_form = compiler.transform_to_native_form(comp)
    return compiler.transform_mathematical_functions_to_tensorflow(native_form)

  return execution_context.ExecutionContext(
      executor_fn=factory, compiler_fn=_debug_compiler)


def set_thread_debugging_execution_context(num_clients=None,
                                           clients_per_thread=1):
  """Sets an execution context that executes computations locally."""
  context = create_thread_debugging_execution_context(
      num_clients=num_clients, clients_per_thread=clients_per_thread)
  context_stack_impl.context_stack.set_default_context(context)


def create_remote_execution_context(channels,
                                    rpc_mode='REQUEST_REPLY',
                                    thread_pool_executor=None,
                                    dispose_batch_size=20,
                                    max_fanout: int = 100,
                                    default_num_clients: int = 0):
  """Creates context to execute computations with workers on `channels`."""
  factory = executor_stacks.remote_executor_factory(
      channels=channels,
      rpc_mode=rpc_mode,
      thread_pool_executor=thread_pool_executor,
      dispose_batch_size=dispose_batch_size,
      max_fanout=max_fanout,
      default_num_clients=default_num_clients,
  )

  return execution_context.ExecutionContext(
      executor_fn=factory, compiler_fn=compiler.transform_to_native_form)


def set_remote_execution_context(channels,
                                 rpc_mode='REQUEST_REPLY',
                                 thread_pool_executor=None,
                                 dispose_batch_size=20,
                                 max_fanout: int = 100,
                                 default_num_clients: int = 0):
  """Installs context to execute computations with workers on `channels`."""
  context = create_remote_execution_context(
      channels=channels,
      rpc_mode=rpc_mode,
      thread_pool_executor=thread_pool_executor,
      dispose_batch_size=dispose_batch_size,
      max_fanout=max_fanout,
      default_num_clients=default_num_clients)
  context_stack_impl.context_stack.set_default_context(context)
