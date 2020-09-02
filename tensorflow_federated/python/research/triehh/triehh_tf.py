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
"""An implementation of the TrieHH Algorithm.

This is intended to be a stand-alone implementation of TrieHH, suitable for
branching as a starting point for algorithm modifications;

Based on the paper:

Federated Heavy Hitters Discovery with Differential Privacy
    Wennan Zhu, Peter Kairouz, H. Brendan McMahan,
    Haicheng Sun, Wei Li. AISTATS 2020.
    https://arxiv.org/pdf/1902.08534.pdf
"""

import attr
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.analytics.heavy_hitters import heavy_hitters_utils as hh_utils

DEFAULT_VALUE = -1  # The value to use if a key is missing in the hash table.
DEFAULT_TERMINATOR = '$'  # The end of sequence symbol.


@attr.s(cmp=False, frozen=True)
class ClientOutput(object):
  """Structure for outputs returned from clients during federated heavy hitters.

  Fields:
    `client_votes`: A tensor containing the client's votes.
  """
  client_votes = attr.ib()


@attr.s(cmp=False, frozen=True)
class ServerState(object):
  """Structure for state on the server.

  Fields:
  `discovered_heavy_hitters`: A tf.string containing discovered heavy
  hitters.
  `heavy_hitters_counts`: A tf.int32 containing the counts of the
  heavy hitter in the round it is discovered.
  `discovered_prefixes`: A tf.tstring containing candidate prefixes.
  `round_num`: A tf.constant dictating the algorithm's round number.
  `accumulated_votes`: A tf.constant that holds the votes accumulated over
  sub-rounds.
  """
  discovered_heavy_hitters = attr.ib()
  heavy_hitters_counts = attr.ib()
  discovered_prefixes = attr.ib()
  round_num = attr.ib()
  accumulated_votes = attr.ib()


def make_accumulate_client_votes_fn(round_num, num_sub_rounds,
                                    discovered_prefixes_table,
                                    possible_prefix_extensions_table,
                                    default_terminator):
  """Returns a reduce function that is used to accumulate client votes.

  This function creates an accumulate_client_votes reduce function that can be
  consumed by a tf.data.Dataset.reduce method. The reduce function maps
  (old_state, example) to a new_state. It must take two arguments and return a
  new element with a structure that matches that of the initial_state.

  Args:
    round_num: A tf.constant containing the round number.
    num_sub_rounds: A tf.constant containing the number of sub rounds in a
      round.
    discovered_prefixes_table: A tf.lookup.StaticHashTable containing the
      discovered prefixes.
    possible_prefix_extensions_table: A tf.lookup.StaticHashTable containing the
      possible prefix extensions that a client can vote on.
    default_terminator: A tf.string containing the end of sequence symbol.

  Returns:
    An accumulate_client_votes reduce function for a specific round, set of
      discovered prefixes, and a set of possbile prefix extensions.
  """

  @tf.function
  def accumulate_client_votes(vote_accumulator, example):
    """Accumulates client votes on prefix extensions."""

    example = tf.strings.lower(example)
    # Append the default terminator to the example.
    example = tf.strings.join([example, default_terminator])

    # Compute effective round number.
    effective_round_num = tf.math.floordiv(round_num, num_sub_rounds)

    if tf.strings.length(example) < effective_round_num:
      return vote_accumulator
    else:
      discovered_prefixes_index = discovered_prefixes_table.lookup(
          tf.strings.substr(example, 0, effective_round_num))
      possible_prefix_extensions_index = possible_prefix_extensions_table.lookup(
          tf.strings.substr(example, effective_round_num, 1))

      # If the character extension is not in the alphabet, or the prefix is not
      # already in the discovered prefixes, do not add client's vote.
      if tf.math.logical_or(
          tf.math.equal(possible_prefix_extensions_index,
                        tf.constant(DEFAULT_VALUE)),
          tf.math.equal(discovered_prefixes_index, tf.constant(DEFAULT_VALUE))):
        return vote_accumulator

      else:
        indices = [[
            discovered_prefixes_index, possible_prefix_extensions_index
        ]]
        updates = tf.constant([1])
        return tf.tensor_scatter_nd_add(vote_accumulator, indices, updates)

  return accumulate_client_votes


@tf.function
def client_update(dataset, discovered_prefixes, possible_prefix_extensions,
                  round_num, num_sub_rounds, max_num_prefixes,
                  max_user_contribution, default_terminator):
  """Creates a ClientOutput object that holds the client's votes.

  This function takes in a 'tf.data.Dataset' containing the client's words,
  selects (up to) `max_user_contribution` words the given `dataset`, and creates
  a `ClientOutput` object that holds the client's votes on chracter extensions
  to `discovered_prefixes`. The allowed character extensions are found in
  `possible_prefix_extensions`. `round_num` and `num_sub_round` are needed to
  compute the length of the prefix to be extended. `max_num_prefixes` is
  needed to set the shape of the tensor holding the client votes.

  Args:
    dataset: A 'tf.data.Dataset' containing the client's on-device words.
    discovered_prefixes: A tf.string containing candidate prefixes.
    possible_prefix_extensions: A tf.string of shape (num_discovered_prefixes, )
      containing possible prefix extensions.
    round_num: A tf.constant dictating the algorithm's round number.
    num_sub_rounds: A tf.constant containing the number of sub rounds in a
      round.
    max_num_prefixes: A tf.constant dictating the maximum number of prefixes we
      can keep in the trie.
    max_user_contribution: A tf.constant dictating the maximum number of
      examples a client can contribute.
    default_terminator: A tf.string containing the end of sequence symbol.

  Returns:
    A ClientOutput object holding the client's votes.
  """
  # Create all zero client vote tensor.
  client_votes = tf.zeros(
      dtype=tf.int32,
      shape=[max_num_prefixes,
             tf.shape(possible_prefix_extensions)[0]])

  # If discovered_prefixes is emtpy (training is done), skip the voting.
  if tf.math.equal(tf.size(discovered_prefixes), 0):
    return ClientOutput(client_votes)
  else:
    discovered_prefixes_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            discovered_prefixes, tf.range(tf.shape(discovered_prefixes)[0])),
        DEFAULT_VALUE)

    possible_prefix_extensions_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            possible_prefix_extensions,
            tf.range(tf.shape(possible_prefix_extensions)[0])), DEFAULT_VALUE)

    accumulate_client_votes_fn = make_accumulate_client_votes_fn(
        round_num, num_sub_rounds, discovered_prefixes_table,
        possible_prefix_extensions_table, default_terminator)

    sampled_data_list = hh_utils.get_top_elements(dataset,
                                                  max_user_contribution)
    sampled_data = tf.data.Dataset.from_tensor_slices(sampled_data_list)

    return ClientOutput(
        sampled_data.reduce(client_votes, accumulate_client_votes_fn))


@tf.function()
def accumulate_server_votes(server_state, sub_round_votes):
  """Accumulates votes and returns an updated server state."""
  accumulated_votes = server_state.accumulated_votes + sub_round_votes
  round_num = server_state.round_num + 1
  return tff.utils.update_state(
      server_state,
      accumulated_votes=accumulated_votes,
      round_num=round_num)


@tf.function()
def get_extended_prefix_candidates(discovered_prefixes,
                                   extensions_wo_terminator):
  """Extend all discovered_prefixes with all possible extensions.

  Args:
    discovered_prefixes: A 1D tf.string containing discovered prefixes.
    extensions_wo_terminator: A 1D tf.string containing containing all possible
      extensions except `default_terminator`.

  Returns:
    A 1D tf.string tensor containing all combinations of each item in
    `discovered_prefixes` extended by each item in `extensions_wo_terminator`.
    Shape: (len(`discovered_prefixes`) * len(`extensions_wo_terminator`), ).
  """
  extended_prefixes = tf.TensorArray(
      dtype=tf.string,
      size=tf.shape(discovered_prefixes)[0] *
      tf.shape(extensions_wo_terminator)[0])
  position = tf.constant(0, dtype=tf.int32)
  for prefix in discovered_prefixes:
    for possible_extension in extensions_wo_terminator:
      # [-1] is passed to tf.reshape to flatten the extended prefix. This is
      # important to ensure consistency of shapes.
      extended_prefix = tf.reshape(
          tf.strings.reduce_join([prefix, possible_extension]), [-1])
      extended_prefixes = extended_prefixes.write(position, extended_prefix)
      position += 1
  return extended_prefixes.concat()


@tf.function()
def extend_prefixes(prefixes_votes, discovered_prefixes,
                    extensions_wo_terminator, max_num_prefixes, threshold):
  """Extends prefixes in `discovered_prefixes` by `extensions_wo_terminator`.

  For any prefix in `discovered_prefixes` with an extension in
  `extensions_wo_terminator`, we only save this extension if the number of votes
  for it is at least `threshold` and it is in the highest `max_num_prefixes`
  votes.

  Args:
    prefixes_votes: A 1D tf.int32 containing flattern votes of all candidates
      for extended prefixes.
    discovered_prefixes: A 1D tf.string containing prefixes to be extended.
    extensions_wo_terminator: A 1D tf.string containing all possible prefix
      extensions except `default_terminator`.
    max_num_prefixes: A tf.constant dictating the maximum number of prefixes we
      can keep in the trie.
    threshold: The threshold for heavy hitters and discovered prefixes. Only
      those get at least `threshold` votes are discovered.

  Returns:
    A 1D tf.string containing all the extended prefixes.
  """
  extended_prefix_candiates = get_extended_prefix_candidates(
      discovered_prefixes, extensions_wo_terminator)
  extended_prefix_candiates_num = tf.shape(extended_prefix_candiates)[0]

  prefixes_mask = tf.math.greater_equal(prefixes_votes, threshold)

  # If the number of candidates for extended prefixes <= max_num_prefixes, we
  # only need to filter the votes by the threshold. Otherwise, the votes needs
  # to be both >= threhold and in top `max_num_prefixes`.
  if tf.shape(prefixes_votes)[0] > max_num_prefixes:
    _, top_indices = tf.math.top_k(prefixes_votes, max_num_prefixes)

    # Create a 1-D tensor filled with tf.bool True of shape (max_num_prefixes,)
    top_indices = tf.cast(top_indices, dtype=tf.int64)
    top_indices = tf.sort(top_indices)
    top_indices = tf.reshape(top_indices, (tf.shape(top_indices)[0], 1))
    top_indices_mask_values = tf.cast(
        tf.ones(shape=(max_num_prefixes,)), dtype=tf.bool)

    # Create a mask tensor that only the indices of the top `max_num_prefixes`
    # candidates are set to True.
    top_indices_mask = tf.sparse.SparseTensor(
        indices=top_indices,
        values=top_indices_mask_values,
        dense_shape=[extended_prefix_candiates_num])
    top_indices_mask = tf.sparse.to_dense(top_indices_mask)
    prefixes_mask = tf.math.logical_and(prefixes_mask, top_indices_mask)

  extended_prefixes = tf.boolean_mask(extended_prefix_candiates, prefixes_mask)
  return extended_prefixes


@tf.function()
def accumulate_server_votes_and_decode(server_state, possible_prefix_extensions,
                                       sub_round_votes, max_num_prefixes,
                                       threshold):
  """Accumulates server votes and executes a decoding round.

  Args:
    server_state: A `ServerState`, the state to be updated.
    possible_prefix_extensions: A 1D tf.string containing all possible prefix
      extensions.
    sub_round_votes: A tensor of shape = (max_num_prefixes,
      len(possible_prefix_extensions)) containing aggregated client votes.
    max_num_prefixes: A tf.constant dictating the maximum number of prefixes we
      can keep in the trie.
    threshold: The threshold for heavy hitters and discovered prefixes. Only
      those get at least `threshold` votes are discovered.

  Returns:
    An updated `ServerState`.
  """
  possible_extensions_num = tf.shape(possible_prefix_extensions)[0]

  # Get a list of possible extensions without `default_terminator` (the last
  # item in `possible_prefix_extensions`)
  extensions_wo_terminator_num = possible_extensions_num - 1
  extensions_wo_terminator = tf.slice(possible_prefix_extensions, [0],
                                      [extensions_wo_terminator_num])

  accumulated_votes = server_state.accumulated_votes + sub_round_votes

  # The last column of `accumulated_votes` are those ending with
  # 'default_terminator`, which are full length heavy hitters.
  heavy_hitters_votes = tf.slice(
      accumulated_votes, [0, extensions_wo_terminator_num],
      [tf.shape(server_state.discovered_prefixes)[0], 1])
  heavy_hitters_votes = tf.reshape(heavy_hitters_votes, [-1])

  heavy_hitters_mask = tf.math.greater_equal(heavy_hitters_votes, threshold)

  # The candidates of heavy hitters are `discovered_prefixes` ending with
  # `default_terminator`. We don't attach `default_terminator` here because it
  # is supposed to be removed after the full length heavy hitters are
  # discovered.
  heavy_hitters_candidates = server_state.discovered_prefixes
  new_heavy_hitters = tf.boolean_mask(heavy_hitters_candidates,
                                      heavy_hitters_mask)

  new_heavy_hitters_counts = tf.boolean_mask(heavy_hitters_votes,
                                             heavy_hitters_mask)

  # All but the last column of `accumulated_votes` are votes of prefixes.
  prefixes_votes = tf.slice(accumulated_votes, [0, 0], [
      tf.shape(server_state.discovered_prefixes)[0],
      extensions_wo_terminator_num
  ])
  prefixes_votes = tf.reshape(prefixes_votes, [-1])

  extended_prefixes = extend_prefixes(prefixes_votes,
                                      server_state.discovered_prefixes,
                                      extensions_wo_terminator,
                                      max_num_prefixes, threshold)

  discovered_heavy_hitters = tf.concat(
      [server_state.discovered_heavy_hitters, new_heavy_hitters], 0)
  heavy_hitters_counts = tf.concat(
      [server_state.heavy_hitters_counts, new_heavy_hitters_counts], 0)

  # Reinitialize the server's vote tensor.
  accumulated_votes = tf.zeros(
      dtype=tf.int32, shape=[max_num_prefixes, possible_extensions_num])

  # Increment the server's round_num.
  round_num = server_state.round_num + 1

  # Return an updated server state.
  return tff.utils.update_state(
      server_state,
      discovered_heavy_hitters=discovered_heavy_hitters,
      heavy_hitters_counts=heavy_hitters_counts,
      round_num=round_num,
      discovered_prefixes=extended_prefixes,
      accumulated_votes=accumulated_votes)


@tf.function
def server_update(server_state, possible_prefix_extensions, sub_round_votes,
                  num_sub_rounds, max_num_prefixes, threshold):
  """Updates `server_state` based on `client_votes`.

  Args:
    server_state: A `ServerState`, the state to be updated.
    possible_prefix_extensions: A 1D tf.string containing all possible prefix
      extensions.
    sub_round_votes: A tensor of shape = (max_num_prefixes,
      len(possible_prefix_extensions)) containing aggregated client votes.
    num_sub_rounds: The total number of sub rounds to be executed before
      decoding aggregated votes.
    max_num_prefixes: A tf.constant dictating the maximum number of prefixes we
      can keep in the trie.
    threshold: The threshold for heavy hitters and discovered prefixes. Only
      those get at least `threshold` votes are discovered.

  Returns:
    An updated `ServerState`.
  """

  # If discovered_prefixes is emtpy (training is done), skip the voting.
  if tf.math.equal(tf.size(server_state.discovered_prefixes), 0):
    return server_state

  if tf.math.equal((server_state.round_num + 1) % num_sub_rounds, 0):
    return accumulate_server_votes_and_decode(server_state,
                                              possible_prefix_extensions,
                                              sub_round_votes, max_num_prefixes,
                                              threshold)
  else:
    return accumulate_server_votes(server_state, sub_round_votes)
