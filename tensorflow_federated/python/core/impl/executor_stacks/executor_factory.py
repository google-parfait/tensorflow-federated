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

import os
import os.path
import signal
import subprocess
import sys
import time

from absl import logging
import grpc
import portpicker

from tensorflow_federated.python.core.impl.executor_stacks import python_executor_stacks
from tensorflow_federated.python.core.impl.executors import executor_factory
from tensorflow_federated.python.core.impl.executors import remote_executor
from tensorflow_federated.python.core.impl.executors import remote_executor_grpc_stub
from tensorflow_federated.python.core.impl.types import placements

_LOCALHOST_SERVER_WAIT_TIME_SEC = 1.0

_GRPC_MAX_MESSAGE_LENGTH_BYTES = 2 * 1000 * 1000 * 1000
_GRPC_CHANNEL_OPTIONS = [
    ('grpc.max_message_length', _GRPC_MAX_MESSAGE_LENGTH_BYTES),
    ('grpc.max_receive_message_length', _GRPC_MAX_MESSAGE_LENGTH_BYTES),
    ('grpc.max_send_message_length', _GRPC_MAX_MESSAGE_LENGTH_BYTES),
]


def local_cpp_executor_factory(
    *,
    default_num_clients: int = 0,
    max_concurrent_computation_calls: int = -1,
    stream_structs: bool = False,
) -> executor_factory.ExecutorFactory:
  """Returns an execution context backed by C++ runtime.

  Args:
    default_num_clients: The number of clients to use as the default
      cardinality, if thus number cannot be inferred by the arguments of a
      computation.
    max_concurrent_computation_calls: The maximum number of concurrent calls to
      a single computation in the C++ runtime. If nonpositive, there is no
      limit.
    stream_structs: The flag to enable decomposing and streaming struct values.

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

  if not os.path.isfile(binary_path):
    logging.debug('Did not find a worker binary at: %s', binary_path)
    raise RuntimeError(f'Expected a worker binary at {binary_path}.')
  else:
    logging.debug('Found a worker binary at: %s', binary_path)

  def start_process() -> tuple[subprocess.Popen[bytes], int]:
    port = portpicker.pick_unused_port()
    args = [
        binary_path,
        f'--port={port}',
        f'--max_concurrent_computation_calls={max_concurrent_computation_calls}',
    ]

    def is_notebook():
      try:
        # `get_ipython` is automatically present in the environment when
        # running in a notebook.
        get_ipython()  # pylint: disable=undefined-variable  # pytype: disable=name-error
        return True
      except NameError:
        return False

    if is_notebook():
      stdout = None
      stderr = None
    else:
      stdout = sys.stdout
      stderr = sys.stderr

    logging.debug('Starting TFF C++ server on port: %s', port)
    return subprocess.Popen(args, stdout=stdout, stderr=stderr), port

  class ServiceManager:
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
      target = f'localhost:{port}'
      channel = grpc.insecure_channel(target, _GRPC_CHANNEL_OPTIONS)
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
    ex = remote_executor.RemoteExecutor(stub, stream_structs=stream_structs)
    ex.set_cardinalities(cardinalities)
    return ex

  return python_executor_stacks.ResourceManagingExecutorFactory(
      executor_stack_fn=stack_fn
  )
