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

from tensorflow_federated.python.research.triehh import triehh_tf
from tensorflow_federated.python.research.triehh import triehh_tff


class TriehhTffTest(tf.test.TestCase):

  def test_build_triehh_process_works_as_expeted(self):
    clients = 3
    num_sub_rounds = 4
    max_rounds = 6
    max_num_heavy_hitters = 3
    max_user_contribution = 100
    roots = (
        string.ascii_lowercase + string.digits + "'@#-;*:./" +
        triehh_tf.DEFAULT_TERMINATOR)
    possible_prefix_extensions = list(roots)

    iterative_process = triehh_tff.build_triehh_process(
        possible_prefix_extensions,
        num_sub_rounds,
        max_num_heavy_hitters,
        max_user_contribution,
        default_terminator=triehh_tf.DEFAULT_TERMINATOR)

    server_state = iterative_process.initialize()
    expected_discovered_prefixes = tf.constant([''], dtype=tf.string)
    expected_discovered_heavy_hitters = tf.constant([], dtype=tf.string)
    expected_accumulated_votes = tf.zeros(
        dtype=tf.int32,
        shape=[max_num_heavy_hitters,
               len(possible_prefix_extensions)])
    expected_round_num = tf.constant(0, dtype=tf.int32)

    self.assertAllEqual(server_state.discovered_prefixes,
                        expected_discovered_prefixes)
    self.assertAllEqual(server_state.discovered_heavy_hitters,
                        expected_discovered_heavy_hitters)
    self.assertAllEqual(server_state.accumulated_votes,
                        expected_accumulated_votes)
    self.assertAllEqual(server_state.round_num, expected_round_num)

    def create_dataset_fn(client_id):
      del client_id
      return tf.data.Dataset.from_tensor_slices(['hello', 'hey', 'hi'])

    client_ids = list(range(100))

    client_data = tff.simulation.ClientData.from_clients_and_fn(
        client_ids=client_ids,
        create_tf_dataset_for_client_fn=create_dataset_fn)

    for round_num in range(max_rounds * num_sub_rounds):
      # TODO(b/152051528): Remove this once lookup table state is cleared in
      # eager executer.
      tff.framework.set_default_executor(tff.framework.local_executor_factory())
      sampled_clients = list(range(clients))
      sampled_datasets = [
          client_data.create_tf_dataset_for_client(client_id)
          for client_id in sampled_clients
      ]
      server_state, _ = iterative_process.next(server_state, sampled_datasets)

      if (round_num + 1) % num_sub_rounds == 0:
        if (max_num_heavy_hitters - len(server_state.discovered_heavy_hitters) <
            1) or (server_state.discovered_prefixes.size == 0):
          # Training is done.
          # All max_num_heavy_hitters have been discovered.
          break

    expected_discovered_heavy_hitters = tf.constant(['hi', 'hey', 'hello'],
                                                    dtype=tf.string)

    self.assertAllEqual(server_state.discovered_heavy_hitters,
                        expected_discovered_heavy_hitters)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
