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
"""Utilities for running a TFF remote runtime on localhost."""
import contextlib
import os
import signal
import subprocess
import sys
from typing import List, Sequence

from absl import logging

import grpc
import tensorflow as tf
import tensorflow_federated as tff


def create_localhost_remote_context(ports: Sequence[str],
                                    default_num_clients=None):
  """Connects remote executors to `ports`."""
  channels = [
      grpc.insecure_channel('localhost:{}'.format(port)) for port in ports
  ]
  if default_num_clients is None:
    context = tff.backends.native.create_remote_python_execution_context(
        channels)
  else:
    context = tff.backends.native.create_remote_python_execution_context(
        channels, default_num_clients=default_num_clients)
  return context


def create_inprocess_worker_contexts(
    ports: Sequence[str]) -> List[contextlib.AbstractContextManager]:
  """Constructs inprocess workers listening on `ports`.

  Starting and running inprocess workers and aggregators leads to lower
  overhead and faster tests than their subprocess-based equivalents, though
  subprocess versions are easier to keep isolated and reason more effectively
  about the cleanup. For this reason, we prefer inprocess versions unless
  isolation or corruption during cleanup is a major concern.

  Args:
    ports: Sequence of strings, defining the ports on which to serve.

  Returns:
    List of context managers which control serving and cleanup on `ports`.
  """
  server_contexts = []
  for port in ports:
    executor_factory = tff.framework.local_executor_factory()
    server_context = tff.simulation.server_context(
        executor_factory, num_threads=1, port=port)
    server_contexts.append(server_context)
  return server_contexts


def create_inprocess_aggregator_contexts(
    worker_ports: Sequence[str],
    aggregator_ports: Sequence[str]) -> List[contextlib.AbstractContextManager]:
  """Constructs inprocess aggregators listening on `aggregator_ports`.

  See comment in `create_inprocess_worker_contexts` for reasons to prefer
  inprocess or subprocess-based aggregators.

  Args:
    worker_ports: Sequence of strings, defining the ports which should host the
      (endpoint) workers in this runtime.
    aggregator_ports: Sequence of strings, defining the ports on which the
      aggregators should be listening.

  Returns:
    List of context managers which control serving and cleanup of aggregators
    and workers.
  """

  worker_contexts = create_inprocess_worker_contexts(worker_ports)

  aggregator_contexts = []

  for target_port, server_port in zip(worker_ports, aggregator_ports):
    channel = grpc.insecure_channel('localhost:{}'.format(target_port))
    executor_factory = tff.framework.remote_executor_factory([channel])
    server_context = tff.simulation.server_context(
        executor_factory, num_threads=1, port=server_port)
    aggregator_contexts.append(server_context)
  return worker_contexts + aggregator_contexts


def create_standalone_subprocess_aggregator_contexts(
    worker_ports: Sequence[str],
    aggregator_ports: Sequence[str]) -> List[contextlib.AbstractContextManager]:
  """Constructs aggregators in subprocess listening on appropriate ports.

  See comment in `create_inprocess_worker_contexts` for reasons to prefer
  inprocess or subprocess-based aggregators.

  Args:
    worker_ports: Sequence of strings, defining the ports which should host the
      (endpoint) workers in this runtime.
    aggregator_ports: Sequence of strings, defining the ports on which the
      aggregators should be listening.

  Returns:
    List of context managers which control serving and cleanup of aggregators
    only; workers must be started and stopped by other means.
  """

  aggregator_contexts = []

  @contextlib.contextmanager
  def _aggregator_subprocess(worker_port, aggregator_port):
    pids = []
    try:
      pids.append(start_python_aggregator(worker_port, aggregator_port))
      yield pids
    finally:
      for pid in pids:
        stop_service_process(pid)

  for worker_port, aggregator_port in zip(worker_ports, aggregator_ports):
    server_context = _aggregator_subprocess(worker_port, aggregator_port)
    aggregator_contexts.append(server_context)

  return aggregator_contexts


def start_python_aggregator(worker_port: str,
                            aggregator_port: str) -> subprocess.Popen:
  """Starts running Python aggregator in a subprocess."""
  python_service_binary = os.path.join(
      tf.compat.v1.resource_loader.get_root_dir_with_all_resources(),
      tf.compat.v1.resource_loader.get_path_to_datafile('test_aggregator'))

  args = [
      python_service_binary,
      f'--worker_port={worker_port}',
      f'--aggregator_port={aggregator_port}',
  ]
  logging.info('Starting python aggregator service via: %s', args)
  if logging.vlog_is_on(1):
    pid = subprocess.Popen(args, stdout=sys.stdout, stderr=sys.stderr)
  else:
    pid = subprocess.Popen(args)
  return pid


def stop_service_process(process: subprocess.Popen):
  logging.info('Sending SIGINT to executor service...')
  os.kill(process.pid, signal.SIGINT)
  logging.info('Waiting for process to complete...')
  process.wait()  # wait for exit from SIGINT
