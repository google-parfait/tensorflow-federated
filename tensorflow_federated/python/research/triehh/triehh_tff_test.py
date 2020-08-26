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

import string

import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.analytics.heavy_hitters import heavy_hitters_testcase as hh_test
from tensorflow_federated.python.research.triehh import triehh_tf
from tensorflow_federated.python.research.triehh import triehh_tff


class TriehhTffTest(hh_test.HeavyHittersTest):

  def create_dataset(self, size):

    def create_dataset_fn(client_id):
      del client_id
      return tf.data.Dataset.from_tensor_slices(['hello', 'hey', 'hi'])

    client_ids = list(range(size))

    return tff.simulation.ClientData.from_clients_and_fn(
        client_ids=client_ids,
        create_tf_dataset_for_client_fn=create_dataset_fn)

  def perform_execution(self, num_sub_rounds=1):
    clients = 3
    max_rounds = 10
    max_num_prefixes = 3
    threshold = 1
    max_user_contribution = 100
    roots = string.ascii_lowercase + string.digits + "'@#-;*:./"
    possible_prefix_extensions = list(roots)

    iterative_process = triehh_tff.build_triehh_process(
        possible_prefix_extensions,
        num_sub_rounds,
        max_num_prefixes,
        threshold,
        max_user_contribution,
        default_terminator=triehh_tf.DEFAULT_TERMINATOR)

    server_state = iterative_process.initialize()
    expected_discovered_prefixes = tf.constant([''], dtype=tf.string)
    expected_discovered_heavy_hitters = tf.constant([], dtype=tf.string)
    expected_accumulated_votes = tf.zeros(
        dtype=tf.int32,
        shape=[max_num_prefixes,
               len(possible_prefix_extensions)])
    expected_round_num = tf.constant(0, dtype=tf.int32)

    self.assertSetAllEqual(server_state.discovered_prefixes,
                           expected_discovered_prefixes)
    self.assertSetAllEqual(server_state.discovered_heavy_hitters,
                           expected_discovered_heavy_hitters)
    self.assertAllEqual(server_state.accumulated_votes,
                        expected_accumulated_votes)
    self.assertAllEqual(server_state.round_num, expected_round_num)

    client_data = self.create_dataset(100)

    for _ in range(max_rounds * num_sub_rounds):
      sampled_clients = list(range(clients))
      sampled_datasets = [
          client_data.create_tf_dataset_for_client(client_id)
          for client_id in sampled_clients
      ]
      server_state, _ = iterative_process.next(server_state, sampled_datasets)

    expected_discovered_heavy_hitters = tf.constant(['hi', 'hey', 'hello'],
                                                    dtype=tf.string)

    self.assertSetAllEqual(server_state.discovered_heavy_hitters,
                           expected_discovered_heavy_hitters)

  def test_build_triehh_process_works_as_expeted(self):
    self.perform_execution(num_sub_rounds=4)

  def test_sub_round_partitioning_work_as_expected(self):
    self.perform_execution(num_sub_rounds=1)


if __name__ == '__main__':
  tf.test.main()
