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


def _get_unused_ports(num_ports=2):
  return [portpicker.pick_unused_port() for _ in range(num_ports)]


_WORKER_PORTS = _get_unused_ports()
_AGGREGATOR_PORTS = _get_unused_ports()


# TODO(b/168744510): This module is intended to be short-lived, and the
# coverage here should be moved down to unit tests when we have a better mocking
# infrastructure deeper in the runtime.
class WorkerFailureTest(parameterized.TestCase):

  def test_computations_run_with_worker_restarts(self):

    context = remote_runtime_test_utils.create_localhost_remote_context(
        _WORKER_PORTS)
    first_contexts = remote_runtime_test_utils.create_inprocess_worker_contexts(
        _WORKER_PORTS)
    second_contexts = remote_runtime_test_utils.create_inprocess_worker_contexts(
        _WORKER_PORTS)

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

  def test_computations_run_with_worker_restarts_and_aggregation(self):

    context = remote_runtime_test_utils.create_localhost_remote_context(
        _AGGREGATOR_PORTS)
    # TODO(b/180524229): Swap for inprocess aggregator when mutex
    # corruption on shutdown is understood.
    aggregation_contexts = remote_runtime_test_utils.create_standalone_subprocess_aggregator_contexts(
        _WORKER_PORTS, _AGGREGATOR_PORTS)
    first_worker_contexts = remote_runtime_test_utils.create_inprocess_worker_contexts(
        _WORKER_PORTS)
    second_worker_contexts = remote_runtime_test_utils.create_inprocess_worker_contexts(
        _WORKER_PORTS)

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

  def test_computations_run_with_partially_available_workers(self):

    tff_context = remote_runtime_test_utils.create_localhost_remote_context(
        _WORKER_PORTS)
    server_contexts = remote_runtime_test_utils.create_inprocess_worker_contexts(
        [_WORKER_PORTS[0]])

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

  def test_worker_going_down_with_fixed_clients_per_round(self):
    tff_context = remote_runtime_test_utils.create_localhost_remote_context(
        _WORKER_PORTS, default_num_clients=10)
    worker_contexts = remote_runtime_test_utils.create_inprocess_worker_contexts(
        _WORKER_PORTS)

    @tff.federated_computation(tff.type_at_server(tf.int32))
    def sum_arg(x):
      return tff.federated_sum(tff.federated_broadcast(x))

    context_stack = tff.framework.get_context_stack()
    with context_stack.install(tff_context):

      with worker_contexts[0]:
        with worker_contexts[1]:
          # With both workers live, we should get 10 back.
          self.assertEqual(sum_arg(1), 10)
        # Leaving the inner context kills the second worker, but should leave
        # the result untouched.
        self.assertEqual(sum_arg(1), 10)


@parameterized.named_parameters(
    # pylint: disable=g-long-lambda
    # pylint: disable=unnecessary-lambda
    (
        'native_remote',
        lambda: remote_runtime_test_utils.create_localhost_remote_context(
            _WORKER_PORTS),
        lambda: remote_runtime_test_utils.create_inprocess_worker_contexts(
            _WORKER_PORTS),
    ),
    (
        'native_remote_intermediate_aggregator',
        lambda: remote_runtime_test_utils.create_localhost_remote_context(
            _AGGREGATOR_PORTS),
        lambda: remote_runtime_test_utils.create_inprocess_aggregator_contexts(
            _WORKER_PORTS, _AGGREGATOR_PORTS),
    ))
class RemoteRuntimeConfigurationChangeTest(parameterized.TestCase):

  def test_computations_run_with_changing_clients(self, context_fn,
                                                  server_contexts_fn):

    @tff.tf_computation(tf.int32)
    @tf.function
    def add_one(x):
      return x + 1

    @tff.federated_computation(tff.type_at_clients(tf.int32))
    def map_add_one(federated_arg):
      return tff.federated_map(add_one, federated_arg)

    context_stack = tff.framework.get_context_stack()
    with context_stack.install(context_fn()):

      with contextlib.ExitStack() as stack:
        for server_context in server_contexts_fn():
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
