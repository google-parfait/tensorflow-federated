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

import grpc
import tensorflow_federated as tff


def create_localhost_remote_context(ports, rpc_mode='REQUEST_REPLY'):
  """Connects remote executors to `ports`."""
  channels = [
      grpc.insecure_channel('localhost:{}'.format(port)) for port in ports
  ]
  context = tff.backends.native.create_remote_execution_context(
      channels, rpc_mode=rpc_mode)
  return context


def create_localhost_worker_contexts(ports):
  """Constructs executor service on `ports`; returns list of contextmanagers."""
  server_contexts = []
  for port in ports:
    executor_factory = tff.framework.local_executor_factory()
    server_context = tff.simulation.server_context(
        executor_factory, num_threads=1, port=port)
    server_contexts.append(server_context)
  return server_contexts


def create_localhost_aggregator_contexts(worker_ports,
                                         aggregator_ports,
                                         rpc_mode='REQUEST_REPLY'):
  """Constructs 2-tiered executor service; returns list of contextmanagers."""

  worker_contexts = create_localhost_worker_contexts(worker_ports)

  aggregator_contexts = []

  for target_port, server_port in zip(worker_ports, aggregator_ports):
    channel = [grpc.insecure_channel('localhost:{}'.format(target_port))]
    ex_factory = tff.framework.remote_executor_factory(
        channel, rpc_mode=rpc_mode)
    server_context = tff.simulation.server_context(
        ex_factory, num_threads=1, port=server_port)
    aggregator_contexts.append(server_context)

  return worker_contexts + aggregator_contexts


def create_standalone_localhost_aggregator_contexts(worker_ports,
                                                    aggregator_ports,
                                                    rpc_mode='REQUEST_REPLY'):
  """Constructs aggregators on appropriate ports without workers."""

  aggregator_contexts = []

  for target_port, server_port in zip(worker_ports, aggregator_ports):
    channel = [grpc.insecure_channel('localhost:{}'.format(target_port))]
    ex_factory = tff.framework.remote_executor_factory(
        channel, rpc_mode=rpc_mode)
    server_context = tff.simulation.server_context(
        ex_factory, num_threads=1, port=server_port)
    aggregator_contexts.append(server_context)

  return aggregator_contexts
