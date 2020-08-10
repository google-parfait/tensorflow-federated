# Copyright 2019, The TensorFlow Federated Authors.
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
"""A set of utilities for components of simulation serving infrastructure."""

import concurrent
import time

import grpc

from tensorflow_federated.proto.v0 import executor_pb2_grpc
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.executors import executor_base
from tensorflow_federated.python.core.impl.executors import executor_service

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def run_server(executor, num_threads, port, credentials=None, options=None):
  """Runs a gRPC server hosting a simulation component in this process.

  The server runs indefinitely, but can be stopped by a keyboard interrrupt.

  Args:
    executor: The executor to be hosted by the server.
    num_threads: The number of network threads to use for handling gRPC calls.
    port: The port to listen on (for gRPC), must be a non-zero integer.
    credentials: The optional credentials to use for the secure connection if
      any, or `None` if the server should open an insecure port. If specified,
      must be a valid `ServerCredentials` object that can be accepted by the
      gRPC server's `add_secure_port()`.
    options: The optional `list` of server options, each in the `(key, value)`
      format accepted by the `grpc.server()` constructor.

  Raises:
    ValueError: If `num_threads` or `port` are invalid.
  """
  py_typecheck.check_type(executor, executor_base.Executor)
  py_typecheck.check_type(num_threads, int)
  py_typecheck.check_type(port, int)
  if credentials is not None:
    py_typecheck.check_type(credentials, grpc.ServerCredentials)
  if num_threads < 1:
    raise ValueError('The number of threads must be a positive integer.')
  if port < 1:
    raise ValueError('The server port must be a positive integer.')
  service = executor_service.ExecutorService(executor)
  server_kwargs = {}
  if options is not None:
    server_kwargs['options'] = options
  server = grpc.server(
      concurrent.futures.ThreadPoolExecutor(max_workers=num_threads),
      **server_kwargs)
  full_port_string = '[::]:{}'.format(port)
  if credentials is not None:
    server.add_secure_port(full_port_string, credentials)
  else:
    server.add_insecure_port(full_port_string)
  executor_pb2_grpc.add_ExecutorServicer_to_server(service, server)
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(None)
