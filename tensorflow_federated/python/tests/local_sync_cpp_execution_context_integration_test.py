# Copyright 2023, The TensorFlow Federated Authors.
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
"""End-to-end tests for the local synchronous C++ execution context."""

from absl.testing import absltest
import tensorflow as tf
import tensorflow_federated as tff


# TODO: b/303513498 - Delete these tests once we add unit tests on the executor
# stack directly (eg. testing that it contains a SequenceExecutor).
class LocalSyncCppExecutionContextIntegrationTest(absltest.TestCase):

  @tff.test.with_context(
      tff.backends.native.create_sync_local_cpp_execution_context
  )
  def test_runs_sequence_reduce(self):
    sequence = list(range(10))

    @tff.federated_computation(tff.type_at_server(tff.SequenceType(tf.int32)))
    def sum_reduce(x):
      zero = tff.federated_value(0, tff.SERVER)

      @tff.tf_computation(tf.int32, tf.int32)
      def add(x, y):
        return x + y

      return tff.sequence_reduce(x, zero, add)

    actual_value = sum_reduce(sequence)
    expected_value = sum(sequence)
    self.assertEqual(actual_value, expected_value)


if __name__ == "__main__":
  absltest.main()
