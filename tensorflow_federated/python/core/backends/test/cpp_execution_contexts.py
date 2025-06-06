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

import os
import os.path
import signal
import subprocess
import sys

from absl import flags
from absl import logging
import federated_language
import portpicker

from tensorflow_federated.python.core.backends.native import compiler as native_compiler
from tensorflow_federated.python.core.backends.test import compiler as test_compiler
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_executor_bindings
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.executor_stacks import cpp_executor_factory
from tensorflow_federated.python.core.impl.executors import executor_bindings


FLAGS = flags.FLAGS


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


def create_sync_interprocess_cpp_execution_context(
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
) -> federated_language.framework.SyncExecutionContext:
  """Creates an execution context backed by TFF-C++ runtime.

  This execution context starts a TFF-C++ worker in a subprocess on the local
  machine and constructs a C++ remote execution context to talk to this
  worker via intra-process gRPC channels.

  Args:
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the CPP runtime. If negative, there is no limit.
    stream_structs: The flag to enable decomposing and streaming struct values.

  Returns:
    An instance of `federated_language.framework.SyncExecutionContext`
    representing the TFF-C++
    runtime.

  Raises:
    RuntimeError: If an internal C++ worker binary can not be found.
  """
  # This path is specified relative to this file because the relative location
  # of the worker binary will remain the same when this function is executed
  # from the Python package and from a Bazel test.
  data_dir = os.path.join(
      os.path.dirname(__file__), '..', '..', '..', '..', 'data'
  )
  binary_name = 'worker_binary'
  binary_path = os.path.join(data_dir, binary_name)
  logging.debug('Found a worker binary at: %s', binary_path)

  def start_process() -> tuple[subprocess.Popen[bytes], int]:
    port = portpicker.pick_unused_port()
    args = [
        binary_path,
        f'--port={port}',
        f'--max_concurrent_computation_calls={max_concurrent_computation_calls}',
    ]
    if FLAGS['logtostderr'].value:
      args.append('--logtostderr')
    elif FLAGS['alsologtostderr'].value:
      args.append('--alsologtostderr')
    if FLAGS['vmodule'].value:
      args.append(f'--vmodule={FLAGS["vmodule"].value}')
    logging.info('Starting TFF C++ server on port: %s', port)
    logging.info('%s', args)
    return subprocess.Popen(args, stdout=sys.stdout, stderr=sys.stderr), port

  class ManagedServiceContext(cpp_executor_factory.CPPExecutorFactory):
    """Class responsible for managing a local TFF executor service."""

    def __init__(self):
      self._channel = None
      self._process = None
      self.initialize_channel()
      self._base_factory = cpp_executor_factory.remote_cpp_executor_factory(
          channels=[self._channel],
          default_num_clients=default_num_clients,
          stream_structs=stream_structs,
      )

    def __del__(self):
      if self._process is not None:
        os.kill(self._process.pid, signal.SIGINT)
        self._process.wait()

    def create_executor(self, cardinalities):
      return self._base_factory.create_executor(cardinalities)

    def clean_up_executor(self, cardinalities) -> None:
      self._base_factory.clean_up_executor(cardinalities)

    def initialize_channel(self) -> None:
      """Ensures a TFF service is running.

      This function ensures that the channel it returns is running, and manages
      the state of the process hosting the TFF service. It additionally ensures
      that it runs only one TFF service at a time.

      Returns:
        An TFF remote executor stub which is guaranteed to be running.
      """
      if self._channel is not None:
        return
      process, port = start_process()
      self._process = process
      self._channel = executor_bindings.create_insecure_grpc_channel(
          f'localhost:{port}'
      )

  return federated_language.framework.SyncExecutionContext(
      executor_fn=ManagedServiceContext(),
      compiler_fn=native_compiler.desugar_and_transform_to_native,
      transform_args=tensorflow_computation.transform_args,
      transform_result=tensorflow_computation.transform_result,
  )


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
