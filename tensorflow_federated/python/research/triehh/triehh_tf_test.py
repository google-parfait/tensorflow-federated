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


class TriehhTfTest(hh_test.HeavyHittersTest):

  def test_accumulate_client_votes_works_as_expected(self):
    possible_prefix_extensions = tf.constant(
        ['a', 'b', 'c', 'd', 'e', triehh_tf.DEFAULT_TERMINATOR],
        dtype=tf.string)
    discovered_prefixes = tf.constant(['a', 'b', 'c', 'd'], dtype=tf.string)
    round_num = tf.constant(1)
    num_sub_rounds = tf.constant(1)
    example1 = tf.constant('ab', dtype=tf.string)

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
        round_num, num_sub_rounds, discovered_prefixes_table,
        possible_prefix_extensions_table,
        tf.constant(triehh_tf.DEFAULT_TERMINATOR, dtype=tf.string))

    initial_votes = tf.constant(
        [[1, 2, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)

    accumulated_votes = accumulate_client_votes(initial_votes, example1)

    expected_accumulated_votes = tf.constant(
        [[1, 3, 1, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)

    self.assertAllEqual(accumulated_votes, expected_accumulated_votes)

    # An example that the prefix is not in the discovered prefixes.
    # The expected result is that the vote is not counted.
    example2 = tf.constant('ea', dtype=tf.string)
    accumulated_votes = accumulate_client_votes(initial_votes, example2)
    self.assertAllEqual(accumulated_votes, initial_votes)

  def test_client_update_works_as_expected(self):
    max_num_prefixes = tf.constant(10)
    max_user_contribution = tf.constant(10)
    possible_prefix_extensions = tf.constant(
        ['a', 'b', 'c', 'd', 'e', triehh_tf.DEFAULT_TERMINATOR],
        dtype=tf.string)
    discovered_prefixes = tf.constant(['a', 'b', 'c', 'd', 'e'],
                                      dtype=tf.string)
    round_num = tf.constant(1)
    num_sub_rounds = tf.constant(1)
    sample_data = tf.data.Dataset.from_tensor_slices(
        ['a', '', 'abc', 'bac', 'abb', 'aaa', 'acc', 'hi'])
    client_output = triehh_tf.client_update(
        sample_data, discovered_prefixes, possible_prefix_extensions, round_num,
        num_sub_rounds, max_num_prefixes, max_user_contribution,
        tf.constant(triehh_tf.DEFAULT_TERMINATOR, dtype=tf.string))

    # Each string is attached with triehh_tf.DEFAULT_TERMINATOR before the
    # client votes, so 'a$' get a vote here.
    expected_client_votes = tf.constant(
        [[1, 2, 1, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],
        dtype=tf.int32)
    self.assertAllEqual(client_output.client_votes, expected_client_votes)

  def test_client_update_works_on_empty_local_datasets(self):
    max_num_prefixes = tf.constant(10)
    max_user_contribution = tf.constant(10)
    possible_prefix_extensions = tf.constant(
        ['a', 'b', 'c', 'd', 'e', triehh_tf.DEFAULT_TERMINATOR],
        dtype=tf.string)
    discovered_prefixes = tf.constant(['a', 'b', 'c', 'd', 'e'],
                                      dtype=tf.string)
    round_num = tf.constant(1)
    num_sub_rounds = tf.constant(1)
    # Force an empty dataset that yields tf.string. Using `from_tensor_slices`
    # defaults to yielding tf.int32 values.
    sample_data = tf.data.Dataset.from_generator(
        generator=lambda: iter(()), output_types=tf.string, output_shapes=())
    client_output = triehh_tf.client_update(
        sample_data, discovered_prefixes, possible_prefix_extensions, round_num,
        num_sub_rounds, max_num_prefixes, max_user_contribution,
        tf.constant(triehh_tf.DEFAULT_TERMINATOR, dtype=tf.string))

    expected_client_votes = tf.constant(
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],
        dtype=tf.int32)
    self.assertAllEqual(client_output.client_votes, expected_client_votes)

  def test_client_update_works_on_empty_discovered_prefixes(self):
    max_num_prefixes = tf.constant(10)
    max_user_contribution = tf.constant(10)
    possible_prefix_extensions = tf.constant(
        ['a', 'b', 'c', 'd', 'e', triehh_tf.DEFAULT_TERMINATOR],
        dtype=tf.string)
    discovered_prefixes = tf.constant([], dtype=tf.string)
    round_num = tf.constant(1)
    num_sub_rounds = tf.constant(1)
    sample_data = tf.data.Dataset.from_tensor_slices(
        ['a', '', 'abc', 'bac', 'abb', 'aaa', 'acc', 'hi'])
    client_output = triehh_tf.client_update(
        sample_data, discovered_prefixes, possible_prefix_extensions, round_num,
        num_sub_rounds, max_num_prefixes, max_user_contribution,
        tf.constant(triehh_tf.DEFAULT_TERMINATOR, dtype=tf.string))

    expected_client_votes = tf.constant(
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],
        dtype=tf.int32)
    self.assertAllEqual(client_output.client_votes, expected_client_votes)

  def test_get_extended_prefix_candidates_works_as_expected(self):
    extensions_wo_terminator = tf.constant(['a', 'b', 'c', 'd'],
                                           dtype=tf.string)
    discovered_prefixes = tf.constant(['a', 'b', 'c'], dtype=tf.string)
    extended_prefix_candidates = triehh_tf.get_extended_prefix_candidates(
        discovered_prefixes, extensions_wo_terminator)
    expected_extended_prefix_candidates = tf.constant([
        'aa', 'ab', 'ac', 'ad', 'ba', 'bb', 'bc', 'bd', 'ca', 'cb', 'cc', 'cd'
    ],
                                                      dtype=tf.string)
    self.assertSetAllEqual(extended_prefix_candidates,
                           expected_extended_prefix_candidates)

  def test_extend_prefixes_works_as_expected(self):
    extensions_wo_terminator = tf.constant(['a', 'b', 'c', 'd'],
                                           dtype=tf.string)
    discovered_prefixes = tf.constant(['a', 'b', 'c'], dtype=tf.string)
    threshold = threshold = tf.constant(1)
    max_num_prefixes = tf.constant(3)
    prefixes_votes = tf.constant([4, 2, 3, 0, 7, 1, 0, 0, 0, 0, 0, 8],
                                 dtype=tf.int32)
    extended_prefixes = triehh_tf.extend_prefixes(prefixes_votes,
                                                  discovered_prefixes,
                                                  extensions_wo_terminator,
                                                  max_num_prefixes, threshold)
    expected_extended_prefixes = tf.constant(['cd', 'ba', 'aa'],
                                             dtype=tf.string)
    self.assertSetAllEqual(extended_prefixes, expected_extended_prefixes)

  def test_extend_prefixes_with_threshold_works_as_expected(self):
    extensions_wo_terminator = tf.constant(['a', 'b', 'c', 'd'],
                                           dtype=tf.string)
    discovered_prefixes = tf.constant(['a', 'b', 'c'], dtype=tf.string)
    threshold = threshold = tf.constant(3)
    max_num_prefixes = tf.constant(20)
    prefixes_votes = tf.constant([4, 2, 3, 0, 7, 1, 0, 0, 0, 0, 0, 8],
                                 dtype=tf.int32)
    extended_prefixes = triehh_tf.extend_prefixes(prefixes_votes,
                                                  discovered_prefixes,
                                                  extensions_wo_terminator,
                                                  max_num_prefixes, threshold)
    expected_extended_prefixes = tf.constant(['aa', 'ac', 'ba', 'cd'],
                                             dtype=tf.string)
    self.assertSetAllEqual(extended_prefixes, expected_extended_prefixes)

  def test_accumulate_server_votes_works_as_expected(self):
    discovered_prefixes = ['a', 'b']
    discovered_heavy_hitters = []
    heavy_hitters_counts = []
    initial_votes = tf.constant(
        [[1, 2, 1, 0, 0], [1, 2, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)

    server_state = triehh_tf.ServerState(
        discovered_heavy_hitters=tf.constant(
            discovered_heavy_hitters, dtype=tf.string),
        heavy_hitters_counts=tf.constant(heavy_hitters_counts, dtype=tf.int32),
        discovered_prefixes=tf.constant(discovered_prefixes, dtype=tf.string),
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
    max_num_prefixes = tf.constant(4)
    threshold = tf.constant(1)
    possible_prefix_extensions = [
        'a', 'n', 's', 't', 'u', triehh_tf.DEFAULT_TERMINATOR
    ]
    possible_prefix_extensions = tf.constant(
        possible_prefix_extensions, dtype=tf.string)
    discovered_prefixes = ['su', 'st']
    discovered_heavy_hitters = []
    heavy_hitters_counts = []
    initial_votes = tf.constant([[1, 2, 1, 0, 0, 0], [1, 2, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                                dtype=tf.int32)

    server_state = triehh_tf.ServerState(
        discovered_heavy_hitters=tf.constant(
            discovered_heavy_hitters, dtype=tf.string),
        heavy_hitters_counts=tf.constant(heavy_hitters_counts, dtype=tf.int32),
        discovered_prefixes=tf.constant(discovered_prefixes, dtype=tf.string),
        round_num=tf.constant(3, dtype=tf.int32),
        accumulated_votes=initial_votes)

    sub_round_votes = tf.constant([[3, 3, 1, 0, 0, 0], [5, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                                  dtype=tf.int32)

    server_state = triehh_tf.accumulate_server_votes_and_decode(
        server_state, possible_prefix_extensions, sub_round_votes,
        max_num_prefixes, threshold)

    expected_accumulated_votes = tf.constant(
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],
        dtype=tf.int32)

    expected_discovered_prefixes = tf.constant(['sta', 'sun', 'sua', 'stn'],
                                               dtype=tf.string)
    expected_discovered_heavy_hitters = tf.constant([], dtype=tf.string)
    expected_heavy_hitters_counts = tf.constant([], dtype=tf.int32)

    self.assertAllEqual(server_state.accumulated_votes,
                        expected_accumulated_votes)
    self.assertSetAllEqual(server_state.discovered_prefixes,
                           expected_discovered_prefixes)
    self.assertSetAllEqual(server_state.discovered_heavy_hitters,
                           expected_discovered_heavy_hitters)
    self.assertHistogramsEqual(server_state.discovered_heavy_hitters,
                               server_state.heavy_hitters_counts,
                               expected_discovered_heavy_hitters,
                               expected_heavy_hitters_counts)

  def test_accumulate_server_votes_and_decode_threhold_works_as_expected(self):
    max_num_prefixes = tf.constant(4)
    threshold = tf.constant(5)
    possible_prefix_extensions = [
        'a', 'n', 's', 't', 'u', triehh_tf.DEFAULT_TERMINATOR
    ]
    possible_prefix_extensions = tf.constant(
        possible_prefix_extensions, dtype=tf.string)
    discovered_prefixes = ['su', 'st']
    discovered_heavy_hitters = []
    heavy_hitters_counts = []
    initial_votes = tf.constant([[1, 2, 1, 0, 0, 0], [1, 2, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                                dtype=tf.int32)

    server_state = triehh_tf.ServerState(
        discovered_heavy_hitters=tf.constant(
            discovered_heavy_hitters, dtype=tf.string),
        heavy_hitters_counts=tf.constant(heavy_hitters_counts, dtype=tf.int32),
        discovered_prefixes=tf.constant(discovered_prefixes, dtype=tf.string),
        round_num=tf.constant(3, dtype=tf.int32),
        accumulated_votes=initial_votes)

    sub_round_votes = tf.constant([[3, 3, 1, 0, 0, 0], [5, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
                                  dtype=tf.int32)

    server_state = triehh_tf.accumulate_server_votes_and_decode(
        server_state, possible_prefix_extensions, sub_round_votes,
        max_num_prefixes, threshold)

    expected_accumulated_votes = tf.constant(
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],
        dtype=tf.int32)
    expected_discovered_prefixes = tf.constant(['sta', 'sun'], dtype=tf.string)
    expected_discovered_heavy_hitters = tf.constant([], dtype=tf.string)
    expected_heavy_hitters_counts = tf.constant([], dtype=tf.int32)

    self.assertAllEqual(server_state.accumulated_votes,
                        expected_accumulated_votes)
    self.assertSetAllEqual(server_state.discovered_prefixes,
                           expected_discovered_prefixes)
    self.assertSetAllEqual(server_state.discovered_heavy_hitters,
                           expected_discovered_heavy_hitters)
    self.assertHistogramsEqual(server_state.discovered_heavy_hitters,
                               server_state.heavy_hitters_counts,
                               expected_discovered_heavy_hitters,
                               expected_heavy_hitters_counts)

  def test_server_update_works_as_expected(self):
    max_num_prefixes = tf.constant(10)
    threshold = tf.constant(1)
    num_sub_rounds = tf.constant(1, dtype=tf.int32)
    possible_prefix_extensions = [
        'a', 'b', 'c', 'd', 'e', triehh_tf.DEFAULT_TERMINATOR
    ]
    possible_prefix_extensions = tf.constant(
        possible_prefix_extensions, dtype=tf.string)
    discovered_prefixes = ['a', 'b', 'c', 'd', 'e']
    discovered_heavy_hitters = []
    heavy_hitters_counts = []

    server_state = triehh_tf.ServerState(
        discovered_heavy_hitters=tf.constant(
            discovered_heavy_hitters, dtype=tf.string),
        heavy_hitters_counts=tf.constant(heavy_hitters_counts, dtype=tf.int32),
        discovered_prefixes=tf.constant(discovered_prefixes, dtype=tf.string),
        round_num=tf.constant(1, dtype=tf.int32),
        accumulated_votes=tf.zeros(
            dtype=tf.int32,
            shape=[max_num_prefixes,
                   len(possible_prefix_extensions)]))

    sub_round_votes = tf.constant(
        [[10, 9, 8, 7, 6, 0], [5, 4, 3, 2, 1, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],
        dtype=tf.int32)

    server_state = triehh_tf.server_update(server_state,
                                           possible_prefix_extensions,
                                           sub_round_votes, num_sub_rounds,
                                           max_num_prefixes, threshold)
    expected_discovered_prefixes = tf.constant(
        ['aa', 'ab', 'ac', 'ad', 'ae', 'ba', 'bb', 'bc', 'bd', 'be'],
        dtype=tf.string)
    expected_discovered_heavy_hitters = tf.constant([], dtype=tf.string)
    expected_heavy_hitters_counts = tf.constant([], dtype=tf.int32)
    expected_accumulated_votes = tf.constant(
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],
        dtype=tf.int32)

    self.assertSetAllEqual(server_state.discovered_prefixes,
                           expected_discovered_prefixes)
    self.assertSetAllEqual(server_state.discovered_heavy_hitters,
                           expected_discovered_heavy_hitters)
    self.assertHistogramsEqual(server_state.discovered_heavy_hitters,
                               server_state.heavy_hitters_counts,
                               expected_discovered_heavy_hitters,
                               expected_heavy_hitters_counts)
    self.assertAllEqual(server_state.accumulated_votes,
                        expected_accumulated_votes)

  def test_server_update_works_on_empty_discovered_prefixes(self):
    max_num_prefixes = tf.constant(10)
    threshold = tf.constant(1)
    num_sub_rounds = tf.constant(1, dtype=tf.int32)
    possible_prefix_extensions = [
        'a', 'b', 'c', 'd', 'e', triehh_tf.DEFAULT_TERMINATOR
    ]
    possible_prefix_extensions = tf.constant(
        possible_prefix_extensions, dtype=tf.string)
    discovered_prefixes = []
    discovered_heavy_hitters = []
    heavy_hitters_counts = []

    server_state = triehh_tf.ServerState(
        discovered_heavy_hitters=tf.constant(
            discovered_heavy_hitters, dtype=tf.string),
        heavy_hitters_counts=tf.constant(heavy_hitters_counts, dtype=tf.int32),
        discovered_prefixes=tf.constant(discovered_prefixes, dtype=tf.string),
        round_num=tf.constant(1, dtype=tf.int32),
        accumulated_votes=tf.zeros(
            dtype=tf.int32,
            shape=[max_num_prefixes,
                   len(possible_prefix_extensions)]))

    sub_round_votes = tf.constant(
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],
        dtype=tf.int32)

    server_state = triehh_tf.server_update(server_state,
                                           possible_prefix_extensions,
                                           sub_round_votes, num_sub_rounds,
                                           max_num_prefixes, threshold)
    expected_discovered_prefixes = tf.constant([], dtype=tf.string)
    expected_discovered_heavy_hitters = tf.constant([], dtype=tf.string)
    expected_heavy_hitters_counts = tf.constant([], dtype=tf.int32)
    expected_accumulated_votes = tf.constant(
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],
        dtype=tf.int32)

    self.assertSetAllEqual(server_state.discovered_prefixes,
                           expected_discovered_prefixes)
    self.assertSetAllEqual(server_state.discovered_heavy_hitters,
                           expected_discovered_heavy_hitters)
    self.assertHistogramsEqual(server_state.discovered_heavy_hitters,
                               server_state.heavy_hitters_counts,
                               expected_discovered_heavy_hitters,
                               expected_heavy_hitters_counts)
    self.assertAllEqual(server_state.accumulated_votes,
                        expected_accumulated_votes)

  def test_server_update_threshold_works_as_expected(self):
    max_num_prefixes = tf.constant(10)
    threshold = tf.constant(5)
    num_sub_rounds = tf.constant(1, dtype=tf.int32)
    possible_prefix_extensions = [
        'a', 'b', 'c', 'd', 'e', triehh_tf.DEFAULT_TERMINATOR
    ]
    possible_prefix_extensions = tf.constant(
        possible_prefix_extensions, dtype=tf.string)
    discovered_prefixes = ['a', 'b', 'c', 'd', 'e']
    discovered_heavy_hitters = []
    heavy_hitters_counts = []

    server_state = triehh_tf.ServerState(
        discovered_heavy_hitters=tf.constant(
            discovered_heavy_hitters, dtype=tf.string),
        heavy_hitters_counts=tf.constant(heavy_hitters_counts, dtype=tf.int32),
        discovered_prefixes=tf.constant(discovered_prefixes, dtype=tf.string),
        round_num=tf.constant(1, dtype=tf.int32),
        accumulated_votes=tf.zeros(
            dtype=tf.int32,
            shape=[max_num_prefixes,
                   len(possible_prefix_extensions)]))

    sub_round_votes = tf.constant(
        [[10, 9, 8, 7, 6, 0], [5, 4, 3, 2, 1, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],
        dtype=tf.int32)

    server_state = triehh_tf.server_update(server_state,
                                           possible_prefix_extensions,
                                           sub_round_votes, num_sub_rounds,
                                           max_num_prefixes, threshold)
    expected_discovered_prefixes = tf.constant(
        ['aa', 'ab', 'ac', 'ad', 'ae', 'ba'], dtype=tf.string)
    expected_discovered_heavy_hitters = tf.constant([], dtype=tf.string)
    expected_heavy_hitters_counts = tf.constant([], dtype=tf.int32)
    expected_accumulated_votes = tf.constant(
        [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],
        dtype=tf.int32)

    self.assertSetAllEqual(server_state.discovered_prefixes,
                           expected_discovered_prefixes)
    self.assertSetAllEqual(server_state.discovered_heavy_hitters,
                           expected_discovered_heavy_hitters)
    self.assertHistogramsEqual(server_state.discovered_heavy_hitters,
                               server_state.heavy_hitters_counts,
                               expected_discovered_heavy_hitters,
                               expected_heavy_hitters_counts)
    self.assertAllEqual(server_state.accumulated_votes,
                        expected_accumulated_votes)

  def test_server_update_finds_heavy_hitters(self):
    max_num_prefixes = tf.constant(10)
    threshold = tf.constant(1)
    num_sub_rounds = tf.constant(1, dtype=tf.int32)
    possible_prefix_extensions = [
        'a', 'b', 'c', 'd', triehh_tf.DEFAULT_TERMINATOR
    ]
    possible_prefix_extensions = tf.constant(
        possible_prefix_extensions, dtype=tf.string)
    discovered_prefixes = ['a', 'b', 'c', 'd', triehh_tf.DEFAULT_TERMINATOR]
    discovered_heavy_hitters = []
    heavy_hitters_counts = []

    server_state = triehh_tf.ServerState(
        discovered_heavy_hitters=tf.constant(
            discovered_heavy_hitters, dtype=tf.string),
        heavy_hitters_counts=tf.constant(heavy_hitters_counts, dtype=tf.int32),
        discovered_prefixes=tf.constant(discovered_prefixes, dtype=tf.string),
        round_num=tf.constant(1, dtype=tf.int32),
        accumulated_votes=tf.zeros(
            dtype=tf.int32,
            shape=[max_num_prefixes,
                   len(possible_prefix_extensions)]))

    sub_round_votes = tf.constant(
        [[10, 9, 8, 7, 6], [5, 4, 3, 0, 4], [2, 1, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)

    server_state = triehh_tf.server_update(server_state,
                                           possible_prefix_extensions,
                                           sub_round_votes, num_sub_rounds,
                                           max_num_prefixes, threshold)
    expected_discovered_prefixes = tf.constant(
        ['aa', 'ab', 'ac', 'ad', 'ba', 'bb', 'bc', 'ca', 'cb'], dtype=tf.string)
    expected_discovered_heavy_hitters = tf.constant(['a', 'b'], dtype=tf.string)
    expected_heavy_hitters_counts = tf.constant([6, 4], dtype=tf.int32)
    expected_accumulated_votes = tf.constant(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)

    self.assertSetAllEqual(server_state.discovered_prefixes,
                           expected_discovered_prefixes)
    self.assertSetAllEqual(server_state.discovered_heavy_hitters,
                           expected_discovered_heavy_hitters)
    self.assertHistogramsEqual(server_state.discovered_heavy_hitters,
                               server_state.heavy_hitters_counts,
                               expected_discovered_heavy_hitters,
                               expected_heavy_hitters_counts)
    self.assertAllEqual(server_state.accumulated_votes,
                        expected_accumulated_votes)

  def test_server_update_finds_heavy_hitters_with_threshold(self):
    max_num_prefixes = tf.constant(10)
    threshold = tf.constant(5)
    num_sub_rounds = tf.constant(1, dtype=tf.int32)
    possible_prefix_extensions = [
        'a', 'b', 'c', 'd', triehh_tf.DEFAULT_TERMINATOR
    ]
    possible_prefix_extensions = tf.constant(
        possible_prefix_extensions, dtype=tf.string)
    discovered_prefixes = ['a', 'b', 'c', 'd', triehh_tf.DEFAULT_TERMINATOR]
    discovered_heavy_hitters = []
    heavy_hitters_counts = []

    server_state = triehh_tf.ServerState(
        discovered_heavy_hitters=tf.constant(
            discovered_heavy_hitters, dtype=tf.string),
        heavy_hitters_counts=tf.constant(heavy_hitters_counts, dtype=tf.int32),
        discovered_prefixes=tf.constant(discovered_prefixes, dtype=tf.string),
        round_num=tf.constant(1, dtype=tf.int32),
        accumulated_votes=tf.zeros(
            dtype=tf.int32,
            shape=[max_num_prefixes,
                   len(possible_prefix_extensions)]))

    sub_round_votes = tf.constant(
        [[10, 9, 8, 7, 6], [5, 4, 3, 0, 4], [2, 1, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)

    server_state = triehh_tf.server_update(server_state,
                                           possible_prefix_extensions,
                                           sub_round_votes, num_sub_rounds,
                                           max_num_prefixes, threshold)
    expected_discovered_prefixes = tf.constant(['aa', 'ab', 'ac', 'ad', 'ba'],
                                               dtype=tf.string)
    expected_discovered_heavy_hitters = tf.constant(['a'], dtype=tf.string)
    expected_heavy_hitters_counts = tf.constant([6], dtype=tf.int32)
    expected_accumulated_votes = tf.constant(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=tf.int32)

    self.assertSetAllEqual(server_state.discovered_prefixes,
                           expected_discovered_prefixes)
    self.assertSetAllEqual(server_state.discovered_heavy_hitters,
                           expected_discovered_heavy_hitters)
    self.assertHistogramsEqual(server_state.discovered_heavy_hitters,
                               server_state.heavy_hitters_counts,
                               expected_discovered_heavy_hitters,
                               expected_heavy_hitters_counts)
    self.assertAllEqual(server_state.accumulated_votes,
                        expected_accumulated_votes)

  def test_server_update_does_not_decode_in_a_subround(self):
    max_num_prefixes = tf.constant(10)
    threshold = tf.constant(1)
    num_sub_rounds = tf.constant(2, dtype=tf.int32)
    possible_prefix_extensions = [
        'a', 'b', 'c', 'd', 'e', triehh_tf.DEFAULT_TERMINATOR
    ]
    possible_prefix_extensions = tf.constant(
        possible_prefix_extensions, dtype=tf.string)
    discovered_prefixes = ['']
    discovered_heavy_hitters = []
    heavy_hitters_counts = []

    server_state = triehh_tf.ServerState(
        discovered_heavy_hitters=tf.constant(
            discovered_heavy_hitters, dtype=tf.string),
        heavy_hitters_counts=tf.constant(heavy_hitters_counts, dtype=tf.int32),
        discovered_prefixes=tf.constant(discovered_prefixes, dtype=tf.string),
        round_num=tf.constant(0, dtype=tf.int32),
        accumulated_votes=tf.zeros(
            dtype=tf.int32,
            shape=[max_num_prefixes,
                   len(possible_prefix_extensions)]))

    sub_round_votes = tf.constant(
        [[1, 2, 1, 2, 0, 0], [2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],
        dtype=tf.int32)

    server_state = triehh_tf.server_update(server_state,
                                           possible_prefix_extensions,
                                           sub_round_votes, num_sub_rounds,
                                           max_num_prefixes, threshold)
    expected_discovered_prefixes = tf.constant([''], dtype=tf.string)
    expected_discovered_heavy_hitters = tf.constant([], dtype=tf.string)
    expected_heavy_hitters_counts = tf.constant([], dtype=tf.int32)
    expected_accumulated_votes = sub_round_votes

    self.assertSetAllEqual(server_state.discovered_prefixes,
                           expected_discovered_prefixes)
    self.assertSetAllEqual(server_state.discovered_heavy_hitters,
                           expected_discovered_heavy_hitters)
    self.assertHistogramsEqual(server_state.discovered_heavy_hitters,
                               server_state.heavy_hitters_counts,
                               expected_discovered_heavy_hitters,
                               expected_heavy_hitters_counts)
    self.assertAllEqual(server_state.accumulated_votes,
                        expected_accumulated_votes)

  def test_all_tf_functions_work_together(self):
    clients = 3
    num_sub_rounds = 4
    max_rounds = 6
    max_num_prefixes = 3
    threshold = 1
    max_user_contribution = 100
    roots = (
        string.ascii_lowercase + string.digits + "'@#-;*:./" +
        triehh_tf.DEFAULT_TERMINATOR)
    possible_prefix_extensions = list(roots)
    possible_prefix_extensions_num = len(possible_prefix_extensions)
    possible_prefix_extensions = tf.constant(
        possible_prefix_extensions, dtype=tf.string)

    server_state = triehh_tf.ServerState(
        discovered_heavy_hitters=tf.constant([], dtype=tf.string),
        heavy_hitters_counts=tf.constant([], dtype=tf.int32),
        discovered_prefixes=tf.constant([''], dtype=tf.string),
        round_num=tf.constant(0, dtype=tf.int32),
        accumulated_votes=tf.zeros(
            dtype=tf.int32,
            shape=[max_num_prefixes, possible_prefix_extensions_num]))

    def create_dataset_fn(client_id):
      del client_id
      return tf.data.Dataset.from_tensor_slices(['hello', 'hey', 'hi'])

    client_ids = list(range(100))

    client_data = tff.simulation.ClientData.from_clients_and_fn(
        client_ids=client_ids,
        create_tf_dataset_for_client_fn=create_dataset_fn)

    for round_num in range(max_rounds * num_sub_rounds):
      sampled_clients = list(range(clients))
      sampled_datasets = [
          client_data.create_tf_dataset_for_client(client_id)
          for client_id in sampled_clients
      ]
      accumulated_votes = tf.zeros(
          dtype=tf.int32,
          shape=[max_num_prefixes, possible_prefix_extensions_num])

      # This is a workaround to clear the graph cache in the `tf.function`; this
      # is necessary because we need to construct a new lookup table every round
      # based on new prefixes.
      client_update = tf.function(triehh_tf.client_update.python_function)

      for dataset in sampled_datasets:
        client_output = client_update(
            dataset, server_state.discovered_prefixes,
            possible_prefix_extensions, round_num, tf.constant(num_sub_rounds),
            tf.constant(max_num_prefixes, dtype=tf.int32),
            tf.constant(max_user_contribution, dtype=tf.int32),
            tf.constant(triehh_tf.DEFAULT_TERMINATOR, dtype=tf.string))
        accumulated_votes += client_output.client_votes

      server_state = triehh_tf.server_update(
          server_state, possible_prefix_extensions, accumulated_votes,
          tf.constant(num_sub_rounds, dtype=tf.int32),
          tf.constant(max_num_prefixes, dtype=tf.int32),
          tf.constant(threshold, dtype=tf.int32))

    expected_discovered_heavy_hitters = tf.constant(['hi', 'hey', 'hello'],
                                                    dtype=tf.string)
    expected_heavy_hitters_counts = tf.constant([12, 12, 12], dtype=tf.int32)
    self.assertSetAllEqual(server_state.discovered_heavy_hitters,
                           expected_discovered_heavy_hitters)
    self.assertHistogramsEqual(server_state.discovered_heavy_hitters,
                               server_state.heavy_hitters_counts,
                               expected_discovered_heavy_hitters,
                               expected_heavy_hitters_counts)

  def test_all_tf_functions_work_together_high_threshold(self):
    clients = 3
    num_sub_rounds = 4
    max_rounds = 6
    max_num_prefixes = 3
    threshold = 100
    max_user_contribution = 100
    roots = (
        string.ascii_lowercase + string.digits + "'@#-;*:./" +
        triehh_tf.DEFAULT_TERMINATOR)
    possible_prefix_extensions = list(roots)
    possible_prefix_extensions_num = len(possible_prefix_extensions)
    possible_prefix_extensions = tf.constant(
        possible_prefix_extensions, dtype=tf.string)

    server_state = triehh_tf.ServerState(
        discovered_heavy_hitters=tf.constant([], dtype=tf.string),
        heavy_hitters_counts=tf.constant([], dtype=tf.int32),
        discovered_prefixes=tf.constant([''], dtype=tf.string),
        round_num=tf.constant(0, dtype=tf.int32),
        accumulated_votes=tf.zeros(
            dtype=tf.int32,
            shape=[max_num_prefixes, possible_prefix_extensions_num]))

    def create_dataset_fn(client_id):
      del client_id
      return tf.data.Dataset.from_tensor_slices(['hello', 'hey', 'hi'])

    client_ids = list(range(100))

    client_data = tff.simulation.ClientData.from_clients_and_fn(
        client_ids=client_ids,
        create_tf_dataset_for_client_fn=create_dataset_fn)

    for round_num in range(max_rounds * num_sub_rounds):
      sampled_clients = list(range(clients))
      sampled_datasets = [
          client_data.create_tf_dataset_for_client(client_id)
          for client_id in sampled_clients
      ]
      accumulated_votes = tf.zeros(
          dtype=tf.int32,
          shape=[max_num_prefixes, possible_prefix_extensions_num])

      # This is a workaround to clear the graph cache in the `tf.function`; this
      # is necessary because we need to construct a new lookup table every round
      # based on new prefixes.
      client_update = tf.function(triehh_tf.client_update.python_function)

      for dataset in sampled_datasets:
        client_output = client_update(
            dataset, server_state.discovered_prefixes,
            possible_prefix_extensions, round_num, tf.constant(num_sub_rounds),
            tf.constant(max_num_prefixes, dtype=tf.int32),
            tf.constant(max_user_contribution, dtype=tf.int32),
            tf.constant(triehh_tf.DEFAULT_TERMINATOR, dtype=tf.string))
        accumulated_votes += client_output.client_votes

      server_state = triehh_tf.server_update(
          server_state, possible_prefix_extensions, accumulated_votes,
          tf.constant(num_sub_rounds, dtype=tf.int32),
          tf.constant(max_num_prefixes, dtype=tf.int32),
          tf.constant(threshold, dtype=tf.int32))

    expected_discovered_heavy_hitters = tf.constant([], dtype=tf.string)
    expected_heavy_hitters_counts = tf.constant([], dtype=tf.int32)
    expected_discovered_prefixes = tf.constant([], dtype=tf.string)

    self.assertSetAllEqual(server_state.discovered_heavy_hitters,
                           expected_discovered_heavy_hitters)
    self.assertHistogramsEqual(server_state.discovered_heavy_hitters,
                               server_state.heavy_hitters_counts,
                               expected_discovered_heavy_hitters,
                               expected_heavy_hitters_counts)
    self.assertSetAllEqual(server_state.discovered_prefixes,
                           expected_discovered_prefixes)


if __name__ == '__main__':
  tf.test.main()
