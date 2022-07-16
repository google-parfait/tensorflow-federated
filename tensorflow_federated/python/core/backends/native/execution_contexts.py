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
import os
import signal
import subprocess
import sys
import time
from typing import List, Optional, Sequence, Tuple

from absl import logging
import grpc
import portpicker

from tensorflow_federated.python.core.backends.native import compiler
from tensorflow_federated.python.core.backends.native import mergeable_comp_compiler
from tensorflow_federated.python.core.impl.context_stack import context_base
from tensorflow_federated.python.core.impl.context_stack import context_stack_impl
from tensorflow_federated.python.core.impl.execution_contexts import async_execution_context
from tensorflow_federated.python.core.impl.execution_contexts import mergeable_comp_execution_context
from tensorflow_federated.python.core.impl.execution_contexts import sync_execution_context
from tensorflow_federated.python.core.impl.executor_stacks import python_executor_stacks
from tensorflow_federated.python.core.impl.executors import remote_executor
from tensorflow_federated.python.core.impl.executors import remote_executor_grpc_stub
from tensorflow_federated.python.core.impl.types import placements

_LOCALHOST_SERVER_WAIT_TIME_SEC = 1.


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
  factory = python_executor_stacks.local_executor_factory(
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
  factory = python_executor_stacks.local_executor_factory(
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
  factory = python_executor_stacks.sizing_executor_factory(
      default_num_clients=default_num_clients,
      max_fanout=max_fanout,
      clients_per_thread=clients_per_thread)
  return sync_execution_context.ExecutionContext(
      executor_fn=factory, compiler_fn=compiler.transform_to_native_form)


def create_thread_debugging_execution_context(default_num_clients: int = 0,
                                              clients_per_thread=1):
  """Creates a simple execution context that executes computations locally."""
  factory = python_executor_stacks.thread_debugging_executor_factory(
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
  factory = python_executor_stacks.remote_executor_factory(
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
  factory = python_executor_stacks.remote_executor_factory(
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
    async_contexts: Sequence[context_base.Context],
    num_subrounds: Optional[int] = None,
) -> mergeable_comp_execution_context.MergeableCompExecutionContext:
  """Creates context which compiles to and executes mergeable comp form.

  Args:
    async_contexts: Asynchronous TFF execution contexts across which to
      distribute work.
    num_subrounds: An optional integer, specifying total the number of subrounds
      desired. If unspecified, the length of `async_contexts` will determine the
      number of subrounds. If more subrounds are requested than contexts are
      passed, invocations will be sequentialized. If fewer, the work will be run
      in parallel across a subset of the `async_contexts`.

  Returns:
    An instance of
    `mergeable_comp_execution_context.MergeableCompExecutionContext` which
    orchestrates work as specified above.
  """
  return mergeable_comp_execution_context.MergeableCompExecutionContext(
      async_contexts=async_contexts,
      # TODO(b/204258376): Enable this py-typecheck when possible.
      compiler_fn=mergeable_comp_compiler.compile_to_mergeable_comp_form,  # pytype: disable=wrong-arg-types
      num_subrounds=num_subrounds,
  )


def set_mergeable_comp_execution_context(
    async_contexts: Sequence[context_base.Context],
    num_subrounds: Optional[int] = None,
):
  """Sets context which compiles to and executes mergeable comp form.

  Args:
    async_contexts: Asynchronous TFF execution contexts across which to
      distribute work.
    num_subrounds: An optional integer, specifying total the number of subrounds
      desired. If unspecified, the length of `async_contexts` will determine the
      number of subrounds. If more subrounds are requested than contexts are
      passed, invocations will be sequentialized. If fewer, the work will be run
      in parallel across a subset of the `async_contexts`.
  """
  context = create_mergeable_comp_execution_context(
      async_contexts=async_contexts,
      num_subrounds=num_subrounds,
  )
  context_stack_impl.context_stack.set_default_context(context)


def set_localhost_cpp_execution_context(
    binary_path: str,
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = 1,
):
  """Sets default context to a localhost TFF executor."""
  context = create_localhost_cpp_execution_context(
      binary_path=binary_path,
      default_num_clients=default_num_clients,
      max_concurrent_computation_calls=max_concurrent_computation_calls,
  )
  context_stack_impl.context_stack.set_default_context(context)


def create_localhost_cpp_execution_context(
    binary_path: str,
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = 0,
) -> sync_execution_context.ExecutionContext:
  """Creates an execution context backed by TFF-C++ runtime.

  This exexucion context starts a TFF-C++ worker assumed to be at path
  `binary_path`, serving on `port`, and constructs a simple (Python) remote
  execution context to talk to this worker.

  Args:
    binary_path: The absolute path to the binary defining the TFF-C++ worker.
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the CPP runtime. If `None`, there is no limit.

  Returns:
    An instance of `context_base.Context` representing the TFF-C++ runtime.
  """
  service_binary = binary_path

  def start_process() -> Tuple[subprocess.Popen[bytes], int]:
    port = portpicker.pick_unused_port()
    args = [
        service_binary, f'--port={port}',
        f'--max_concurrent_computation_calls={max_concurrent_computation_calls}'
    ]
    logging.debug('Starting TFF C++ server on port: %s', port)
    return subprocess.Popen(args, stdout=sys.stdout, stderr=sys.stderr), port

  class ServiceManager():
    """Class responsible for managing a local TFF executor service."""

    def __init__(self):
      self._stub = None
      self._process = None

    def __del__(self):
      if isinstance(self._process, subprocess.Popen):
        os.kill(self._process.pid, signal.SIGINT)
      self._process.wait()

    def get_stub(self) -> remote_executor_grpc_stub.RemoteExecutorGrpcStub:
      """Ensures a TFF service is running.

      Returns stub representing this service.

      This function ensures that the stub it returns is running, and managers
      the state of the process hosting the TFF service. It additionally ensures
      that it runs only one TFF service at a time.

      Returns:
        An TFF remote executor stub which is guaranteed to be running.
      """
      if self._stub is not None:
        if self._stub.is_ready:
          return self._stub
        # Stub is not ready; since we block below, this must imply that the
        # service is down. Kill the process and restart below.
        os.kill(self._process.pid, signal.SIGINT)
        logging.debug('Waiting for existing processes to complete')
        self._process.wait()
      # Start a process and block til the associated stub is ready.
      process, port = start_process()
      channel = grpc.insecure_channel('localhost:{}'.format(port))
      stub = remote_executor_grpc_stub.RemoteExecutorGrpcStub(channel)
      self._process = process
      self._stub = stub
      while not self._stub.is_ready:
        time.sleep(_LOCALHOST_SERVER_WAIT_TIME_SEC)
        logging.debug('TFF service manager sleeping; stub is not ready.')
      return self._stub

  service_manager = ServiceManager()

  def stack_fn(cardinalities):
    if cardinalities.get(placements.CLIENTS) is None:
      cardinalities[placements.CLIENTS] = default_num_clients
    stub = service_manager.get_stub()
    ex = remote_executor.RemoteExecutor(stub)
    ex.set_cardinalities(cardinalities)
    return ex

  factory = python_executor_stacks.ResourceManagingExecutorFactory(
      executor_stack_fn=stack_fn)
  return _make_basic_python_execution_context(
      executor_fn=factory,
      compiler_fn=compiler.desugar_and_transform_to_native,
      asynchronous=False)
