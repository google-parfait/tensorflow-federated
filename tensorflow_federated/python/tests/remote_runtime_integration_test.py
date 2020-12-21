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

import contextlib

from absl.testing import absltest
from absl.testing import parameterized
import portpicker
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.tests import remote_runtime_test_utils

_WORKER_PORTS = [portpicker.pick_unused_port() for _ in range(2)]
_AGGREGATOR_PORTS = [portpicker.pick_unused_port() for _ in range(2)]


# TODO(b/168744510): This module is intended to be short-lived, and the
# coverage here should be moved down to unit tests when we have a better mocking
# infrastructure deeper in the runtime.
class WorkerFailureTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('native_remote_request_reply',
       remote_runtime_test_utils.create_localhost_remote_context(_WORKER_PORTS),
       remote_runtime_test_utils.create_localhost_worker_contexts(
           _WORKER_PORTS),
       remote_runtime_test_utils.create_localhost_worker_contexts(_WORKER_PORTS)
      ),
      ('native_remote_streaming',
       remote_runtime_test_utils.create_localhost_remote_context(
           _WORKER_PORTS, rpc_mode='STREAMING'),
       remote_runtime_test_utils.create_localhost_worker_contexts(
           _WORKER_PORTS),
       remote_runtime_test_utils.create_localhost_worker_contexts(_WORKER_PORTS)
      ),
  )
  def test_computations_run_with_worker_restarts(self, context, first_contexts,
                                                 second_contexts):

    @tff.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @tff.federated_computation(tff.type_at_clients(tf.int32))
    def map_add_one(federated_arg):
      return tff.federated_map(add_one, federated_arg)

    context_stack = tff.framework.get_context_stack()
    with context_stack.install(context):

      with contextlib.ExitStack() as stack:
        for server_context in first_contexts:
          stack.enter_context(server_context)
        result = map_add_one([0, 1])
        self.assertEqual(result, [1, 2])

      # Closing and re-entering the server contexts serves to simulate failures
      # and restarts at the workers. Restarts leave the workers in a state that
      # needs initialization again; entering the second context ensures that the
      # servers need to be reinitialized by the controller.
      with contextlib.ExitStack() as stack:
        for server_context in second_contexts:
          stack.enter_context(server_context)
        result = map_add_one([0, 1])
        self.assertEqual(result, [1, 2])

  @parameterized.named_parameters((
      'native_remote',
      remote_runtime_test_utils.create_localhost_remote_context(
          _AGGREGATOR_PORTS),
      remote_runtime_test_utils.create_standalone_localhost_aggregator_contexts(
          _WORKER_PORTS, _AGGREGATOR_PORTS),
      remote_runtime_test_utils.create_localhost_worker_contexts(_WORKER_PORTS),
      remote_runtime_test_utils.create_localhost_worker_contexts(_WORKER_PORTS)
  ), ('native_remote_streaming',
      remote_runtime_test_utils.create_localhost_remote_context(
          _AGGREGATOR_PORTS, rpc_mode='STREAMING'),
      remote_runtime_test_utils.create_standalone_localhost_aggregator_contexts(
          _WORKER_PORTS, _AGGREGATOR_PORTS, rpc_mode='STREAMING'),
      remote_runtime_test_utils.create_localhost_worker_contexts(_WORKER_PORTS),
      remote_runtime_test_utils.create_localhost_worker_contexts(_WORKER_PORTS))
                                 )
  def test_computations_run_with_worker_restarts_and_aggregation(
      self, context, aggregation_contexts, first_worker_contexts,
      second_worker_contexts):

    @tff.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @tff.federated_computation(tff.type_at_clients(tf.int32))
    def map_add_one(federated_arg):
      return tff.federated_map(add_one, federated_arg)

    context_stack = tff.framework.get_context_stack()
    with context_stack.install(context):

      with contextlib.ExitStack() as aggregation_stack:
        for server_context in aggregation_contexts:
          aggregation_stack.enter_context(server_context)
        with contextlib.ExitStack() as first_worker_stack:
          for server_context in first_worker_contexts:
            first_worker_stack.enter_context(server_context)

          result = map_add_one([0, 1])
          self.assertEqual(result, [1, 2])

        # Reinitializing the workers without leaving the aggregation context
        # simulates a worker failure, while the aggregator keeps running.
        with contextlib.ExitStack() as second_worker_stack:
          for server_context in second_worker_contexts:
            second_worker_stack.enter_context(server_context)
          result = map_add_one([0, 1])
          self.assertEqual(result, [1, 2])

  @parameterized.named_parameters(
      (
          'native_remote_request_reply',
          remote_runtime_test_utils.create_localhost_remote_context(
              _WORKER_PORTS),
          remote_runtime_test_utils.create_localhost_worker_contexts(
              [_WORKER_PORTS[0]]),
      ),
      (
          'native_remote_streaming',
          remote_runtime_test_utils.create_localhost_remote_context(
              _WORKER_PORTS, rpc_mode='STREAMING'),
          remote_runtime_test_utils.create_localhost_worker_contexts(
              [_WORKER_PORTS[0]]),
      ),
  )
  def test_computations_run_with_partially_available_workers(
      self, tff_context, server_contexts):

    self.skipTest('b/174679820')

    @tff.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @tff.federated_computation(tff.type_at_clients(tf.int32))
    def map_add_one(federated_arg):
      return tff.federated_map(add_one, federated_arg)

    context_stack = tff.framework.get_context_stack()
    with context_stack.install(tff_context):

      with contextlib.ExitStack() as stack:
        for server_context in server_contexts:
          stack.enter_context(server_context)
        result = map_add_one([0, 1])
        self.assertEqual(result, [1, 2])


# TODO(b/172025644): Promote streaming plus intermediate aggregation to a
# proper backend test when the final cleanup issues are diagnosed and fixed.
class StreamingWithIntermediateAggTest(absltest.TestCase):

  def test_runs_computation_streaming_with_intermediate_agg(self):

    @tff.tf_computation(tf.int32)
    def add_one(x):
      return x + 1

    @tff.federated_computation(tff.type_at_clients(tf.int32))
    def map_add_one_and_sum(federated_arg):
      return tff.federated_sum(tff.federated_map(add_one, federated_arg))

    execution_context = remote_runtime_test_utils.create_localhost_remote_context(
        _AGGREGATOR_PORTS, rpc_mode='STREAMING')
    worker_contexts = remote_runtime_test_utils.create_localhost_aggregator_contexts(
        _WORKER_PORTS, _AGGREGATOR_PORTS, rpc_mode='STREAMING')

    context_stack = tff.framework.get_context_stack()
    with context_stack.install(execution_context):

      with contextlib.ExitStack() as stack:
        for server_context in worker_contexts:
          stack.enter_context(server_context)
        result = map_add_one_and_sum([0, 1])
        self.assertEqual(result, 3)


@parameterized.named_parameters((
    'native_remote_request_reply',
    remote_runtime_test_utils.create_localhost_remote_context(_WORKER_PORTS),
    remote_runtime_test_utils.create_localhost_worker_contexts(_WORKER_PORTS),
), (
    'native_remote_streaming',
    remote_runtime_test_utils.create_localhost_remote_context(
        _WORKER_PORTS, rpc_mode='STREAMING'),
    remote_runtime_test_utils.create_localhost_worker_contexts(_WORKER_PORTS),
), (
    'native_remote_intermediate_aggregator',
    remote_runtime_test_utils.create_localhost_remote_context(
        _AGGREGATOR_PORTS),
    remote_runtime_test_utils.create_localhost_aggregator_contexts(
        _WORKER_PORTS, _AGGREGATOR_PORTS),
))
class RemoteRuntimeConfigurationChangeTest(absltest.TestCase):

  def test_computations_run_with_changing_clients(self, context,
                                                  server_contexts):

    @tff.tf_computation(tf.int32)
    @tf.function
    def add_one(x):
      return x + 1

    @tff.federated_computation(tff.type_at_clients(tf.int32))
    def map_add_one(federated_arg):
      return tff.federated_map(add_one, federated_arg)

    context_stack = tff.framework.get_context_stack()
    with context_stack.install(context):

      with contextlib.ExitStack() as stack:
        for server_context in server_contexts:
          stack.enter_context(server_context)
        result_two_clients = map_add_one([0, 1])
        self.assertEqual(result_two_clients, [1, 2])
        # Moving to three clients should be fine
        result_three_clients = map_add_one([0, 1, 2])
        # Running a 0-client function should also be OK
        self.assertEqual(add_one(0), 1)
        self.assertEqual(result_three_clients, [1, 2, 3])
        # Changing back to 2 clients should still succeed.
        second_result_two_clients = map_add_one([0, 1])
        self.assertEqual(second_result_two_clients, [1, 2])
        # Similarly, 3 clients again should be fine.
        second_result_three_clients = map_add_one([0, 1, 2])
        self.assertEqual(second_result_three_clients, [1, 2, 3])


if __name__ == '__main__':
  absltest.main()
