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
class RemoteRuntimeIntegrationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('native_remote',
       remote_runtime_test_utils.create_localhost_remote_context(_WORKER_PORTS),
       remote_runtime_test_utils.create_localhost_worker_contexts(
           _WORKER_PORTS),
       remote_runtime_test_utils.create_localhost_worker_contexts(_WORKER_PORTS)
      ),
      ('native_remote_intermediate_aggregator',
       remote_runtime_test_utils.create_localhost_remote_context(
           _AGGREGATOR_PORTS),
       remote_runtime_test_utils.create_localhost_aggregator_contexts(
           _WORKER_PORTS, _AGGREGATOR_PORTS),
       remote_runtime_test_utils.create_localhost_aggregator_contexts(
           _WORKER_PORTS, _AGGREGATOR_PORTS)),
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


if __name__ == '__main__':
  absltest.main()
