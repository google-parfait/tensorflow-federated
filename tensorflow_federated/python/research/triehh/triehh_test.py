# Lint as: python3
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

import tensorflow as tf

from tensorflow_federated.python.research.triehh import triehh_tf


class TriehhTest(tf.test.TestCase):

  def test_accumulate_client_votes_works_as_expected(self):
    possible_prefix_extensions = tf.constant(['a', 'b', 'c', 'd', '$'],
                                             dtype=tf.string)
    discovered_prefixes = tf.constant(['a', 'b', 'c', 'd'], dtype=tf.string)
    round_num = tf.constant(1)
    example = tf.constant('ab', dtype=tf.string)

    discovered_prefixes_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            discovered_prefixes, tf.range(tf.shape(discovered_prefixes)[0])),
        triehh_tf.DEFAULT_VALUE)

    possible_prefix_extensions_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            possible_prefix_extensions,
            tf.range(tf.shape(possible_prefix_extensions)[0])),
        triehh_tf.DEFAULT_VALUE)

    accumulate_client_votes = triehh_tf.make_accumulate_client_votes_fn(
        round_num, discovered_prefixes_table, possible_prefix_extensions_table)

    initial_votes = tf.constant(
        [[1, 2, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)

    accumulated_votes = accumulate_client_votes(initial_votes, example)
    expected_accumulated_votes = tf.constant(
        [[1, 3, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)
    self.assertAllEqual(accumulated_votes, expected_accumulated_votes)

  def test_client_update_works_as_expected(self):
    max_num_heavy_hitters = tf.constant(10)
    max_user_contribution = tf.constant(10)
    possible_prefix_extensions = tf.constant(['a', 'b', 'c', 'd', 'e'],
                                             dtype=tf.string)
    discovered_prefixes = tf.constant(['a', 'b', 'c', 'd', 'e'],
                                      dtype=tf.string)
    round_num = tf.constant(1)
    sample_data = tf.data.Dataset.from_tensor_slices(
        ['a', '', 'abc', 'bac', 'abb', 'aaa', 'acc', 'hi'])
    client_output = triehh_tf.client_update(sample_data, discovered_prefixes,
                                            possible_prefix_extensions,
                                            round_num, max_num_heavy_hitters,
                                            max_user_contribution)

    expected_client_votes = tf.constant(
        [[1, 2, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)
    self.assertAllEqual(client_output.client_votes, expected_client_votes)

  def test_client_update_works_on_empty_local_datasets(self):
    max_num_heavy_hitters = tf.constant(10)
    max_user_contribution = tf.constant(10)
    possible_prefix_extensions = tf.constant(['a', 'b', 'c', 'd', 'e'],
                                             dtype=tf.string)
    discovered_prefixes = tf.constant(['a', 'b', 'c', 'd', 'e'],
                                      dtype=tf.string)
    round_num = tf.constant(1)
    sample_data = tf.data.Dataset.from_tensor_slices([])
    client_output = triehh_tf.client_update(sample_data, discovered_prefixes,
                                            possible_prefix_extensions,
                                            round_num, max_num_heavy_hitters,
                                            max_user_contribution)

    expected_client_votes = tf.constant(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)
    self.assertAllEqual(client_output.client_votes, expected_client_votes)

  def test_extend_prefixes_works_as_expected(self):
    possible_prefix_extensions = tf.constant(['a', 'b', 'c', 'd', '$'],
                                             dtype=tf.string)
    prefixes_to_extend = tf.constant(['a', 'b', 'c', 'd', 'e'], dtype=tf.string)
    extended_prefixes = triehh_tf.extend_prefixes(prefixes_to_extend,
                                                  possible_prefix_extensions)
    expected_extended_prefixes = tf.constant([
        'aa', 'ab', 'ac', 'ad', 'a$', 'ba', 'bb', 'bc', 'bd', 'b$', 'ca', 'cb',
        'cc', 'cd', 'c$', 'da', 'db', 'dc', 'dd', 'd$', 'ea', 'eb', 'ec', 'ed',
        'e$'
    ],
                                             dtype=tf.string)
    self.assertAllEqual(extended_prefixes, expected_extended_prefixes)

  def test_extend_prefixes_and_discover_new_heavy_hitters_works_as_expected(
      self):
    discovered_prefixes = tf.constant(['sta', 'sun', 'moo'], dtype=tf.string)
    possible_prefix_extensions = tf.constant(['r', 'n', '$'], dtype=tf.string)
    discovered_prefixes_indices = tf.constant([0, 1, 2], dtype=tf.int32)
    prefix_extensions_indices = tf.constant([0, 2, 1], dtype=tf.int32)
    default_terminator = tf.constant('$', dtype=tf.string)

    extended_prefixes, new_heavy_hitters = triehh_tf.extend_prefixes_and_discover_new_heavy_hitters(
        discovered_prefixes, possible_prefix_extensions,
        discovered_prefixes_indices, prefix_extensions_indices,
        default_terminator)

    expected_extended_prefixes = tf.constant(['star', 'moon'], dtype=tf.string)
    expected_new_heavy_hitters = tf.constant(['sun'], dtype=tf.string)

    self.assertAllEqual(extended_prefixes, expected_extended_prefixes)
    self.assertAllEqual(new_heavy_hitters, expected_new_heavy_hitters)

  def test_accumulate_server_votes_works_as_expected(self):
    possible_prefix_extensions = ['a', 'b', 'c', 'd', 'e']
    discovered_prefixes = ['a', 'b']
    discovered_heavy_hitters = []
    initial_votes = tf.constant(
        [[1, 2, 1, 0, 0], [1, 2, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)

    server_state = triehh_tf.ServerState(
        discovered_heavy_hitters=tf.constant(
            discovered_heavy_hitters, dtype=tf.string),
        discovered_prefixes=tf.constant(discovered_prefixes, dtype=tf.string),
        possible_prefix_extensions=tf.constant(
            possible_prefix_extensions, dtype=tf.string),
        round_num=tf.constant(0, dtype=tf.int32),
        accumulated_votes=initial_votes)

    sub_round_votes = tf.constant(
        [[1, 2, 1, 0, 0], [1, 2, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)

    server_state = triehh_tf.accumulate_server_votes(server_state,
                                                     sub_round_votes)
    expected_accumulated_votes = tf.constant(
        [[2, 4, 2, 0, 0], [2, 4, 2, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)

    self.assertAllEqual(server_state.accumulated_votes,
                        expected_accumulated_votes)

  def test_accumulate_server_votes_and_decode_works_as_expected(self):
    max_num_heavy_hitters = tf.constant(4)
    default_terminator = tf.constant('$', tf.string)
    possible_prefix_extensions = ['a', 'n', 's', 't', 'u']
    discovered_prefixes = ['su', 'st']
    discovered_heavy_hitters = []
    initial_votes = tf.constant(
        [[1, 2, 1, 0, 0], [1, 2, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)

    server_state = triehh_tf.ServerState(
        discovered_heavy_hitters=tf.constant(
            discovered_heavy_hitters, dtype=tf.string),
        discovered_prefixes=tf.constant(discovered_prefixes, dtype=tf.string),
        possible_prefix_extensions=tf.constant(
            possible_prefix_extensions, dtype=tf.string),
        round_num=tf.constant(3, dtype=tf.int32),
        accumulated_votes=initial_votes)

    sub_round_votes = tf.constant(
        [[3, 3, 1, 0, 0], [5, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)

    server_state = triehh_tf.accumulate_server_votes_and_decode(
        server_state, sub_round_votes, max_num_heavy_hitters,
        default_terminator)

    expected_accumulated_votes = tf.constant(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)
    expected_discovered_prefixes = tf.constant(['sta', 'sun', 'sua', 'stn'],
                                               dtype=tf.string)
    expected_discovered_heavy_hitters = tf.constant([], dtype=tf.string)

    self.assertAllEqual(server_state.accumulated_votes,
                        expected_accumulated_votes)
    self.assertAllEqual(server_state.discovered_prefixes,
                        expected_discovered_prefixes)
    self.assertAllEqual(server_state.discovered_heavy_hitters,
                        expected_discovered_heavy_hitters)

  def test_server_update_works_as_expected(self):
    max_num_heavy_hitters = tf.constant(10)
    default_terminator = tf.constant('$', tf.string)
    num_sub_rounds = tf.constant(1, dtype=tf.int32)
    possible_prefix_extensions = ['a', 'b', 'c', 'd', 'e']
    discovered_prefixes = ['a', 'b', 'c', 'd', 'e']
    discovered_heavy_hitters = []

    server_state = triehh_tf.ServerState(
        discovered_heavy_hitters=tf.constant(
            discovered_heavy_hitters, dtype=tf.string),
        discovered_prefixes=tf.constant(discovered_prefixes, dtype=tf.string),
        possible_prefix_extensions=tf.constant(
            possible_prefix_extensions, dtype=tf.string),
        round_num=tf.constant(1, dtype=tf.int32),
        accumulated_votes=tf.zeros(
            dtype=tf.int32,
            shape=[max_num_heavy_hitters,
                   len(possible_prefix_extensions)]))

    sub_round_votes = tf.constant(
        [[10, 9, 8, 7, 6], [5, 4, 3, 2, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)

    server_state = triehh_tf.server_update(server_state, sub_round_votes,
                                           num_sub_rounds,
                                           max_num_heavy_hitters,
                                           default_terminator)
    expected_discovered_prefixes = tf.constant(
        ['aa', 'ab', 'ac', 'ad', 'ae', 'ba', 'bb', 'bc', 'bd', 'be'],
        dtype=tf.string)
    expected_discovered_heavy_hitters = tf.constant([], dtype=tf.string)
    expected_accumulated_votes = tf.constant(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)

    self.assertAllEqual(server_state.discovered_prefixes,
                        expected_discovered_prefixes)
    self.assertAllEqual(server_state.discovered_heavy_hitters,
                        expected_discovered_heavy_hitters)
    self.assertAllEqual(server_state.accumulated_votes,
                        expected_accumulated_votes)

  def test_server_update_finds_heavy_hitters(self):
    max_num_heavy_hitters = tf.constant(10)
    default_terminator = tf.constant('$', tf.string)
    num_sub_rounds = tf.constant(1, dtype=tf.int32)
    possible_prefix_extensions = ['a', 'b', 'c', 'd', '$']
    discovered_prefixes = ['a', 'b', 'c', 'd', '$']
    discovered_heavy_hitters = []

    server_state = triehh_tf.ServerState(
        discovered_heavy_hitters=tf.constant(
            discovered_heavy_hitters, dtype=tf.string),
        discovered_prefixes=tf.constant(discovered_prefixes, dtype=tf.string),
        possible_prefix_extensions=tf.constant(
            possible_prefix_extensions, dtype=tf.string),
        round_num=tf.constant(1, dtype=tf.int32),
        accumulated_votes=tf.zeros(
            dtype=tf.int32,
            shape=[max_num_heavy_hitters,
                   len(possible_prefix_extensions)]))

    sub_round_votes = tf.constant(
        [[10, 9, 8, 7, 6], [5, 4, 3, 0, 0], [2, 1, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)

    server_state = triehh_tf.server_update(server_state, sub_round_votes,
                                           num_sub_rounds,
                                           max_num_heavy_hitters,
                                           default_terminator)
    expected_discovered_prefixes = tf.constant(
        ['aa', 'ab', 'ac', 'ad', 'ba', 'bb', 'bc', 'ca', 'cb'], dtype=tf.string)
    expected_discovered_heavy_hitters = tf.constant(['a'], dtype=tf.string)
    expected_accumulated_votes = tf.constant(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)

    self.assertAllEqual(server_state.discovered_prefixes,
                        expected_discovered_prefixes)
    self.assertAllEqual(server_state.discovered_heavy_hitters,
                        expected_discovered_heavy_hitters)
    self.assertAllEqual(server_state.accumulated_votes,
                        expected_accumulated_votes)

  def test_server_update_does_not_decode_in_a_subround(self):
    max_num_heavy_hitters = tf.constant(10)
    default_terminator = tf.constant('$', tf.string)
    num_sub_rounds = tf.constant(2, dtype=tf.int32)
    possible_prefix_extensions = ['a', 'b', 'c', 'd', 'e']
    discovered_prefixes = ['']
    discovered_heavy_hitters = []

    server_state = triehh_tf.ServerState(
        discovered_heavy_hitters=tf.constant(
            discovered_heavy_hitters, dtype=tf.string),
        discovered_prefixes=tf.constant(discovered_prefixes, dtype=tf.string),
        possible_prefix_extensions=tf.constant(
            possible_prefix_extensions, dtype=tf.string),
        round_num=tf.constant(0, dtype=tf.int32),
        accumulated_votes=tf.zeros(
            dtype=tf.int32,
            shape=[max_num_heavy_hitters,
                   len(possible_prefix_extensions)]))

    sub_round_votes = tf.constant(
        [[1, 2, 1, 2, 0], [2, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)

    server_state = triehh_tf.server_update(server_state, sub_round_votes,
                                           num_sub_rounds,
                                           max_num_heavy_hitters,
                                           default_terminator)
    expected_discovered_prefixes = tf.constant([''], dtype=tf.string)
    expected_discovered_heavy_hitters = tf.constant([], dtype=tf.string)
    expected_accumulated_votes = sub_round_votes

    self.assertAllEqual(server_state.discovered_prefixes,
                        expected_discovered_prefixes)
    self.assertAllEqual(server_state.discovered_heavy_hitters,
                        expected_discovered_heavy_hitters)
    self.assertAllEqual(server_state.accumulated_votes,
                        expected_accumulated_votes)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
