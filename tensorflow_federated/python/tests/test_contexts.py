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
"""Contexts and constructors for integration testing."""

import asyncio
import contextlib
import functools
from typing import Sequence

from absl.testing import parameterized
import grpc
import portpicker
import tensorflow_federated as tff

from tensorflow_federated.python.tests import remote_runtime_test_utils

WORKER_PORTS = [portpicker.pick_unused_port() for _ in range(2)]
AGGREGATOR_PORTS = [portpicker.pick_unused_port() for _ in range(2)]

_GRPC_MAX_MESSAGE_LENGTH_BYTES = 1024 * 1024 * 1024
_GRPC_CHANNEL_OPTIONS = [
    ('grpc.max_message_length', _GRPC_MAX_MESSAGE_LENGTH_BYTES),
    ('grpc.max_receive_message_length', _GRPC_MAX_MESSAGE_LENGTH_BYTES),
    ('grpc.max_send_message_length', _GRPC_MAX_MESSAGE_LENGTH_BYTES)
]


def create_native_local_caching_context():
  local_ex_factory = tff.framework.local_executor_factory()

  def _wrap_local_executor_with_caching(cardinalities):
    local_ex = local_ex_factory.create_executor(cardinalities)
    return tff.framework.CachingExecutor(local_ex)

  return tff.framework.ExecutionContext(
      tff.framework.ResourceManagingExecutorFactory(
          _wrap_local_executor_with_caching))


def _get_remote_executors_for_ports(ports):
  executors = []
  for port in ports:
    server_endpoint = f'[::]:{port}'
    channel = grpc.insecure_channel(
        server_endpoint, options=_GRPC_CHANNEL_OPTIONS)
    executors.append(tff.framework.RemoteExecutor(channel=channel))
  return executors


def create_localhost_remote_tf_context(
    tf_serving_ports: Sequence[str]) -> tff.framework.ExecutionContext:
  """Creates an execution context which pushes TensorFlow to remote workers."""
  remote_executors = _get_remote_executors_for_ports(tf_serving_ports)

  workers = [
      tff.framework.ThreadDelegatingExecutor(ex) for ex in remote_executors
  ]

  def _stack_fn(cardinalities):
    event_loop = asyncio.new_event_loop()
    for ex in remote_executors:
      # Configure each remote worker to have a single client.
      event_loop.run_until_complete(ex.set_cardinalities({tff.CLIENTS: 1}))
    if cardinalities.get(tff.CLIENTS) is not None and cardinalities[
        tff.CLIENTS] > len(remote_executors):
      raise ValueError(
          'Requested {} clients but this stack can only support at most {}.'
          .format(cardinalities.get(tff.CLIENTS), len(remote_executors)))

    if cardinalities.get(tff.CLIENTS) is None:
      requested_workers = workers
    else:
      requested_workers = workers[:cardinalities[tff.CLIENTS]]

    federating_strategy_factory = tff.framework.FederatedResolvingStrategy.factory(
        {
            tff.CLIENTS: requested_workers,
            tff.SERVER: tff.framework.EagerTFExecutor()
        })
    fed_ex = tff.framework.FederatingExecutor(federating_strategy_factory,
                                              tff.framework.EagerTFExecutor())
    top_rre = tff.framework.ReferenceResolvingExecutor(fed_ex)
    return top_rre

  ex_factory = tff.framework.ResourceManagingExecutorFactory(
      _stack_fn, ensure_closed=remote_executors)
  # When the RRE goes in we wont need this anymore
  compiler_fn = tff.backends.native.transform_mathematical_functions_to_tensorflow

  return tff.framework.ExecutionContext(
      executor_fn=ex_factory, compiler_fn=compiler_fn)


def _get_all_contexts():
  # pyformat: disable
  return [
      ('native_local', tff.backends.native.create_local_execution_context()),
      ('native_local_caching', create_native_local_caching_context()),
      ('native_remote',
       remote_runtime_test_utils.create_localhost_remote_context(WORKER_PORTS),
       remote_runtime_test_utils.create_inprocess_worker_contexts(WORKER_PORTS)),
      ('native_remote_intermediate_aggregator',
       remote_runtime_test_utils.create_localhost_remote_context(AGGREGATOR_PORTS),
       remote_runtime_test_utils.create_inprocess_aggregator_contexts(WORKER_PORTS, AGGREGATOR_PORTS)),
      ('native_sizing', tff.backends.native.create_sizing_execution_context()),
      ('native_thread_debug',
       tff.backends.native.create_thread_debugging_execution_context()),
      ('reference', tff.backends.reference.create_reference_context()),
      ('test', tff.backends.test.create_test_execution_context()),
  ]
  # pyformat: enable


def with_context(context):
  """A decorator for running tests in the given `context`."""

  def decorator_context(fn):

    @functools.wraps(fn)
    def wrapper_context(self):
      context_stack = tff.framework.get_context_stack()
      with context_stack.install(context):
        fn(self)

    return wrapper_context

  return decorator_context


def with_environment(server_contexts):
  """A decorator for running tests in an environment."""

  def decorator_environment(fn):

    @functools.wraps(fn)
    def wrapper_environment(self):
      with contextlib.ExitStack() as stack:
        for server_context in server_contexts:
          stack.enter_context(server_context)
        fn(self)

    return wrapper_environment

  return decorator_environment


def with_contexts(*args):
  """A decorator for creating tests parameterized by context."""

  def decorator_contexts(fn, *named_contexts):
    if not named_contexts:
      named_contexts = _get_all_contexts()

    @parameterized.named_parameters(*named_contexts)
    def wrapper_contexts(self, context, server_contexts=None):
      with_context_decorator = with_context(context)
      decorated_fn = with_context_decorator(fn)
      if server_contexts is not None:
        with_environment_decorator = with_environment(server_contexts)
        decorated_fn = with_environment_decorator(decorated_fn)
      decorated_fn(self)

    return wrapper_contexts

  if len(args) == 1 and callable(args[0]):
    return decorator_contexts(args[0])
  else:
    return lambda fn: decorator_contexts(fn, *args)
