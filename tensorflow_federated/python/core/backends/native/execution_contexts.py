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

from concurrent import futures
from typing import List, Optional, Sequence

import grpc

from tensorflow_federated.python.core.backends.native import compiler
from tensorflow_federated.python.core.backends.native import mergeable_comp_compiler
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.execution_contexts import async_execution_context
from tensorflow_federated.python.core.impl.execution_contexts import mergeable_comp_execution_context
from tensorflow_federated.python.core.impl.execution_contexts import sync_execution_context
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import executor_stacks


def _make_basic_python_execution_context(*, executor_fn, compiler_fn,
                                         asynchronous):
  """Wires executor function and compiler into sync or async context."""

  if not asynchronous:
    context = sync_execution_context.ExecutionContext(
        executor_fn=executor_fn, compiler_fn=compiler_fn)
  else:
    context = async_execution_context.AsyncExecutionContext(
        executor_fn=executor_fn, compiler_fn=compiler_fn)

  return context


def create_local_python_execution_context(
    default_num_clients: int = 0,
    max_fanout: int = 100,
    clients_per_thread: int = 1,
    server_tf_device=None,
    client_tf_devices=tuple(),
    reference_resolving_clients=False
) -> sync_execution_context.ExecutionContext:
  """Creates an execution context that executes computations locally."""
  factory = executor_stacks.local_executor_factory(
      default_num_clients=default_num_clients,
      max_fanout=max_fanout,
      clients_per_thread=clients_per_thread,
      server_tf_device=server_tf_device,
      client_tf_devices=client_tf_devices,
      reference_resolving_clients=reference_resolving_clients)

  def _compiler(comp):
    native_form = compiler.transform_to_native_form(
        comp, transform_math_to_tf=not reference_resolving_clients)
    return native_form

  return _make_basic_python_execution_context(
      executor_fn=factory, compiler_fn=_compiler, asynchronous=False)


def set_local_python_execution_context(default_num_clients: int = 0,
                                       max_fanout: int = 100,
                                       clients_per_thread: int = 1,
                                       server_tf_device=None,
                                       client_tf_devices=tuple(),
                                       reference_resolving_clients=False):
  """Sets an execution context that executes computations locally."""
  context = create_local_python_execution_context(
      default_num_clients=default_num_clients,
      max_fanout=max_fanout,
      clients_per_thread=clients_per_thread,
      server_tf_device=server_tf_device,
      client_tf_devices=client_tf_devices,
      reference_resolving_clients=reference_resolving_clients,
  )
  context_stack_impl.context_stack.set_default_context(context)


def create_local_async_python_execution_context(
    default_num_clients: int = 0,
    max_fanout: int = 100,
    clients_per_thread: int = 1,
    server_tf_device=None,
    client_tf_devices=tuple(),
    reference_resolving_clients: bool = False
) -> async_execution_context.AsyncExecutionContext:
  """Creates a context that executes computations locally as coro functions."""
  factory = executor_stacks.local_executor_factory(
      default_num_clients=default_num_clients,
      max_fanout=max_fanout,
      clients_per_thread=clients_per_thread,
      server_tf_device=server_tf_device,
      client_tf_devices=client_tf_devices,
      reference_resolving_clients=reference_resolving_clients)

  def _compiler(comp):
    native_form = compiler.transform_to_native_form(
        comp, transform_math_to_tf=not reference_resolving_clients)
    return native_form

  return _make_basic_python_execution_context(
      executor_fn=factory, compiler_fn=_compiler, asynchronous=True)


def set_local_async_python_execution_context(
    default_num_clients: int = 0,
    max_fanout: int = 100,
    clients_per_thread: int = 1,
    server_tf_device=None,
    client_tf_devices=tuple(),
    reference_resolving_clients: bool = False):
  """Sets a context that executes computations locally as coro functions."""
  context = create_local_async_python_execution_context(
      default_num_clients=default_num_clients,
      max_fanout=max_fanout,
      clients_per_thread=clients_per_thread,
      server_tf_device=server_tf_device,
      client_tf_devices=client_tf_devices,
      reference_resolving_clients=reference_resolving_clients)
  context_stack_impl.context_stack.set_default_context(context)


def create_sizing_execution_context(default_num_clients: int = 0,
                                    max_fanout: int = 100,
                                    clients_per_thread: int = 1):
  """Creates an execution context that executes computations locally."""
  factory = executor_stacks.sizing_executor_factory(
      default_num_clients=default_num_clients,
      max_fanout=max_fanout,
      clients_per_thread=clients_per_thread)
  return sync_execution_context.ExecutionContext(
      executor_fn=factory, compiler_fn=compiler.transform_to_native_form)


def create_thread_debugging_execution_context(default_num_clients: int = 0,
                                              clients_per_thread=1):
  """Creates a simple execution context that executes computations locally."""
  factory = executor_stacks.thread_debugging_executor_factory(
      default_num_clients=default_num_clients,
      clients_per_thread=clients_per_thread,
  )

  def _debug_compiler(comp):
    return compiler.transform_to_native_form(comp, transform_math_to_tf=True)

  return sync_execution_context.ExecutionContext(
      executor_fn=factory, compiler_fn=_debug_compiler)


def set_thread_debugging_execution_context(default_num_clients: int = 0,
                                           clients_per_thread=1):
  """Sets an execution context that executes computations locally."""
  context = create_thread_debugging_execution_context(
      default_num_clients=default_num_clients,
      clients_per_thread=clients_per_thread)
  context_stack_impl.context_stack.set_default_context(context)


def create_remote_python_execution_context(
    channels,
    thread_pool_executor=None,
    dispose_batch_size=20,
    max_fanout: int = 100,
    default_num_clients: int = 0,
) -> sync_execution_context.ExecutionContext:
  """Creates context to execute computations with workers on `channels`."""
  factory = executor_stacks.remote_executor_factory(
      channels=channels,
      thread_pool_executor=thread_pool_executor,
      dispose_batch_size=dispose_batch_size,
      max_fanout=max_fanout,
      default_num_clients=default_num_clients,
  )

  return _make_basic_python_execution_context(
      executor_fn=factory,
      compiler_fn=compiler.transform_to_native_form,
      asynchronous=False)


def set_remote_python_execution_context(
    channels,
    thread_pool_executor=None,
    dispose_batch_size=20,
    max_fanout: int = 100,
    default_num_clients: int = 0,
):
  """Installs context to execute computations with workers on `channels`."""
  context = create_remote_python_execution_context(
      channels=channels,
      thread_pool_executor=thread_pool_executor,
      dispose_batch_size=dispose_batch_size,
      max_fanout=max_fanout,
      default_num_clients=default_num_clients,
  )
  context_stack_impl.context_stack.set_default_context(context)


def create_remote_async_python_execution_context(
    channels: List[grpc.Channel],
    thread_pool_executor: Optional[futures.Executor] = None,
    dispose_batch_size: int = 20,
    max_fanout: int = 100,
    default_num_clients: int = 0
) -> async_execution_context.AsyncExecutionContext:
  """Creates context executing computations async via workers on `channels`."""
  factory = executor_stacks.remote_executor_factory(
      channels=channels,
      thread_pool_executor=thread_pool_executor,
      dispose_batch_size=dispose_batch_size,
      max_fanout=max_fanout,
      default_num_clients=default_num_clients,
  )

  return _make_basic_python_execution_context(
      executor_fn=factory,
      compiler_fn=compiler.transform_to_native_form,
      asynchronous=True)


def set_remote_async_python_execution_context(channels,
                                              thread_pool_executor=None,
                                              dispose_batch_size=20,
                                              max_fanout: int = 100,
                                              default_num_clients: int = 0):
  """Installs context executing computations async via workers on `channels`."""
  context = create_remote_async_python_execution_context(
      channels=channels,
      thread_pool_executor=thread_pool_executor,
      dispose_batch_size=dispose_batch_size,
      max_fanout=max_fanout,
      default_num_clients=default_num_clients,
  )
  context_stack_impl.context_stack.set_default_context(context)


def create_mergeable_comp_execution_context(
    executor_factories: Sequence[executor_factory.ExecutorFactory]
) -> mergeable_comp_execution_context.MergeableCompExecutionContext:
  """Creates context which compiles to and executes mergeable comp form."""
  return mergeable_comp_execution_context.MergeableCompExecutionContext(
      executor_factories=executor_factories,
      # TODO(b/204258376): Enable this py-typecheck when possible.
      compiler_fn=mergeable_comp_compiler.compile_to_mergeable_comp_form)  # pytype: disable=wrong-arg-types


def set_mergeable_comp_execution_context(
    executor_factories: Sequence[executor_factory.ExecutorFactory]):
  """Sets context which compiles to and executes mergeable comp form."""
  context = create_mergeable_comp_execution_context(
      executor_factories=executor_factories)
  context_stack_impl.context_stack.set_default_context(context)
