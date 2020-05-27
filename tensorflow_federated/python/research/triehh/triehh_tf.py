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
  `discovered_prefixes`: A tf.tstring containing candidate prefixes.
  `possible_prefix_extensions`: A tf.tstring containing possible prefix
  extensions.
  `round_num`: A tf.constant dictating the algorithm's round number.
  `accumulated_votes`: A tf.constant that holds the votes accumulated over
  sub-rounds.
  """
  discovered_heavy_hitters = attr.ib()
  discovered_prefixes = attr.ib()
  possible_prefix_extensions = attr.ib()
  round_num = attr.ib()
  accumulated_votes = attr.ib()


def make_accumulate_client_votes_fn(round_num, num_sub_rounds,
                                    discovered_prefixes_table,
                                    possible_prefix_extensions_table):
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

  Returns:
    An accumulate_client_votes reduce function for a specific round, set of
      discovered prefixes, and a set of possbile prefix extensions.
  """

  @tf.function
  def accumulate_client_votes(vote_accumulator, example):
    """Accumulates client votes on prefix extensions."""

    example = tf.strings.lower(example)
    # Append the default terminator to the example.
    default_terminator = tf.constant(DEFAULT_TERMINATOR, dtype=tf.string)
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

      if tf.math.equal(possible_prefix_extensions_index,
                       tf.constant(DEFAULT_VALUE)):
        return vote_accumulator

      elif tf.math.equal(discovered_prefixes_index, tf.constant(DEFAULT_VALUE)):
        indices = [[0, possible_prefix_extensions_index]]
        updates = tf.constant([1])
        return tf.tensor_scatter_nd_add(vote_accumulator, indices, updates)

      else:
        indices = [[
            discovered_prefixes_index, possible_prefix_extensions_index
        ]]
        updates = tf.constant([1])
        return tf.tensor_scatter_nd_add(vote_accumulator, indices, updates)

  return accumulate_client_votes


@tf.function
def client_update(dataset, discovered_prefixes, possible_prefix_extensions,
                  round_num, num_sub_rounds, max_num_heavy_hitters,
                  max_user_contribution):
  """Creates a ClientOutput object that holds the client's votes.

  This function takes in a 'tf.data.Dataset' containing the client's words,
  selects (up to) `max_user_contribution` words the given `dataset`, and creates
  a `ClientOutput` object that holds the client's votes on chracter extensions
  to `discovered_prefixes`. The allowed character extensions are found in
  `possible_prefix_extensions`. `round_num` and `num_sub_round` are needed to
  compute the length of the prefix to be extended. `max_num_heavy_hitters` is
  needed to set the shape of the tensor holding the client votes.

  Args:
    dataset: A 'tf.data.Dataset' containing the client's on-device words.
    discovered_prefixes: A tf.string containing candidate prefixes.
    possible_prefix_extensions: A tf.string of shape (num_discovered_prefixes, )
      containing possible prefix extensions.
    round_num: A tf.constant dictating the algorithm's round number.
    num_sub_rounds: A tf.constant containing the number of sub rounds in a
      round.
    max_num_heavy_hitters: A tf.constant dictating the maximum number of heavy
      hitters to discover.
    max_user_contribution: A tf.constant dictating the maximum number of
      examples a client can contribute.

  Returns:
    A ClientOutput object holding the client's votes.
  """
  # Create all zero client vote tensor.
  client_votes = tf.zeros(
      dtype=tf.int32,
      shape=[max_num_heavy_hitters,
             tf.shape(possible_prefix_extensions)[0]])

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
      possible_prefix_extensions_table)

  sampled_data = tf.data.Dataset.from_tensor_slices(
      hh_utils.get_top_elements(
          hh_utils.listify(dataset), max_user_contribution))

  return ClientOutput(
      sampled_data.reduce(client_votes, accumulate_client_votes_fn))


@tf.function()
def extend_prefixes(prefixes_to_extend, possible_prefix_extensions):
  """Extends prefixes in `prefixes_to_extend` by `possible_prefix_extensions`.

  Args:
    prefixes_to_extend: A 1D tf.string containing prefixes to be extended.
    possible_prefix_extensions: A 1D tf.string containing all possible prefix
      extensions.

  Returns:
    A 1D tf.string containing all the extended prefixes.
  """
  num_new_prefixes = tf.shape(prefixes_to_extend)[0] * tf.shape(
      possible_prefix_extensions)[0]
  extended_prefixes = tf.TensorArray(dtype=tf.string, size=num_new_prefixes)
  position = tf.constant(0, dtype=tf.int32)
  for prefix in prefixes_to_extend:
    for possible_extension in possible_prefix_extensions:
      # [-1] is passed to tf.reshape to flatten the extended prefix. This is
      # important to ensure consistency of shapes.
      extended_prefix = tf.reshape(
          tf.strings.reduce_join([prefix, possible_extension]), [-1])
      extended_prefixes = extended_prefixes.write(position, extended_prefix)
      position += 1
  return extended_prefixes.concat()


@tf.function()
def extend_prefixes_and_discover_new_heavy_hitters(discovered_prefixes,
                                                   possible_prefix_extensions,
                                                   discovered_prefixes_indices,
                                                   prefix_extensions_indices,
                                                   default_terminator):
  """Extends prefixes and discovers new heavy hitters.

  Args:
    discovered_prefixes: A 1D tf.string containing prefixes to be extended.
    possible_prefix_extensions: A 1D tf.string containing all possible prefix
      extensions.
    discovered_prefixes_indices: A 1D tf.int32 tensor cotaining the indices of
      the prefixes to be extended.
    prefix_extensions_indices: A 1D tf.int32 tensor cotaining the indices of the
      prefix extensions. For example, discovered_prefixes at
      discovered_prefixes_indices[0] will be extended by
      possible_prefix_extensions at prefix_extensions_indices[0].
    default_terminator: A 0D tf.string tensor holding the end of word symbol.

  Returns:
    extended_prefixes: A 1D tf.string containing the extended prefixes.
    new_heavy_hitters: A 1D tf.string containing the discovered heavy hitters.
  """
  remaining_num_of_heavy_hitters = tf.shape(prefix_extensions_indices)[0]
  extended_prefixes = tf.TensorArray(
      dtype=tf.string, size=remaining_num_of_heavy_hitters, element_shape=(1,))
  new_heavy_hitters = tf.TensorArray(
      dtype=tf.string, size=remaining_num_of_heavy_hitters, element_shape=(1,))

  num_new_heavy_hitters = tf.constant(0, dtype=tf.int32)
  num_extended_prefixes = tf.constant(0, dtype=tf.int32)

  for i in tf.range(remaining_num_of_heavy_hitters):
    extension = possible_prefix_extensions[prefix_extensions_indices[i]]
    prefix = discovered_prefixes[discovered_prefixes_indices[i]]
    if tf.equal(extension, default_terminator):
      new_heavy_hitters = new_heavy_hitters.write(num_new_heavy_hitters,
                                                  tf.reshape(prefix, [1]))
      num_new_heavy_hitters += 1
    else:
      extended_prefix = tf.reshape(
          tf.strings.reduce_join([prefix, extension]), [1])
      extended_prefixes = extended_prefixes.write(num_extended_prefixes,
                                                  extended_prefix)
      num_extended_prefixes += 1

  if num_new_heavy_hitters == 0:
    new_heavy_hitters = tf.zeros((0,), dtype=tf.string)
  else:
    new_heavy_hitters = new_heavy_hitters.concat()[:num_new_heavy_hitters]

  if num_extended_prefixes == 0:
    extended_prefixes = tf.zeros((0,), dtype=tf.string)
  else:
    extended_prefixes = extended_prefixes.concat()[:num_extended_prefixes]

  return extended_prefixes, new_heavy_hitters


@tf.function()
def accumulate_server_votes(server_state, sub_round_votes):
  """Accumulates votes and returns an updated server state."""
  accumulated_votes = server_state.accumulated_votes + sub_round_votes
  round_num = server_state.round_num + 1
  return tff.utils.update_state(
      server_state, accumulated_votes=accumulated_votes, round_num=round_num)


@tf.function()
def accumulate_server_votes_and_decode(server_state, sub_round_votes,
                                       max_num_heavy_hitters,
                                       default_terminator):
  """Accumulates server votes and executes a decoding round.

  Args:
    server_state: A `ServerState`, the state to be updated.
    sub_round_votes: A tensor of shape = (max_num_heavy_hitters,
      len(possible_prefix_extensions)) containing aggregated client votes.
    max_num_heavy_hitters: The total number of heavy hitters to be discovered.
    default_terminator: The end of sequence symbol.

  Returns:
    An updated `ServerState`.
  """
  # Compute the remaining number of heavy hitters to be discovered.
  remaining_num_of_heavy_hitters = max_num_heavy_hitters - tf.shape(
      server_state.discovered_heavy_hitters)[0]

  # Calculate the total number of possible extensions.
  num_possible_extensions = tf.shape(
      server_state.discovered_prefixes)[0] * tf.shape(
          server_state.possible_prefix_extensions)[0]

  # If num_possible_extensions <= remaining_num_of_heavy_hitters, extend all.
  if num_possible_extensions <= remaining_num_of_heavy_hitters:
    extended_prefixes = extend_prefixes(server_state.discovered_prefixes,
                                        server_state.possible_prefix_extensions)

    # Reinitialize the vote tensor.
    accumulated_votes = tf.zeros(
        dtype=tf.int32,
        shape=[
            max_num_heavy_hitters,
            tf.shape(server_state.possible_prefix_extensions)[0]
        ])
    round_num = server_state.round_num + 1

    # Update discovered_prefixes and round_num.
    return tff.utils.update_state(
        server_state,
        round_num=round_num,
        discovered_prefixes=extended_prefixes,
        accumulated_votes=accumulated_votes)
  else:
    # Accumulate votes.
    accumulated_votes = server_state.accumulated_votes + sub_round_votes

    # Get top `remaining_num_of_heavy_hitters` candidates.
    relevant_votes = tf.slice(
        accumulated_votes, [0, 0],
        [tf.shape(server_state.discovered_prefixes)[0], -1])
    relevant_votes = tf.reshape(relevant_votes, [
        tf.shape(server_state.discovered_prefixes)[0] *
        tf.shape(server_state.possible_prefix_extensions)[0]
    ])
    _, flattened_indices = tf.math.top_k(relevant_votes,
                                         remaining_num_of_heavy_hitters)
    prefix_extensions_indices = tf.math.mod(
        flattened_indices,
        tf.shape(server_state.possible_prefix_extensions)[0])
    discovered_prefixes_indices = tf.math.floordiv(
        flattened_indices,
        tf.shape(server_state.possible_prefix_extensions)[0])

    # TODO(b/150705615): Keep only candidates with positive votes.

    # Extend prefixes and save newly discovered heavy hitters.
    extended_prefixes, new_heavy_hitters = extend_prefixes_and_discover_new_heavy_hitters(
        server_state.discovered_prefixes,
        server_state.possible_prefix_extensions, discovered_prefixes_indices,
        prefix_extensions_indices, default_terminator)

    discovered_heavy_hitters = tf.concat(
        [server_state.discovered_heavy_hitters, new_heavy_hitters], 0)

    # Reinitialize the server's vote tensor.
    accumulated_votes = tf.zeros(
        dtype=tf.int32,
        shape=[
            max_num_heavy_hitters,
            tf.shape(server_state.possible_prefix_extensions)[0]
        ])

    # Increment the server's round_num.
    round_num = server_state.round_num + 1

    # Return an udpated server state.
    return tff.utils.update_state(
        server_state,
        discovered_heavy_hitters=discovered_heavy_hitters,
        round_num=round_num,
        discovered_prefixes=extended_prefixes,
        accumulated_votes=accumulated_votes)


@tf.function
def server_update(server_state, sub_round_votes, num_sub_rounds,
                  max_num_heavy_hitters, default_terminator):
  """Updates `server_state` based on `client_votes`.

  Args:
    server_state: A `ServerState`, the state to be updated.
    sub_round_votes: A tensor of shape = (max_num_heavy_hitters,
      len(possible_prefix_extensions)) containing aggregated client votes.
    num_sub_rounds: The total number of sub rounds to be executed before
      decoding aggregated votes.
    max_num_heavy_hitters: The total number of heavy hitters to be discovered.
    default_terminator: The end of sequence symbol.

  Returns:
    An updated `ServerState`.
  """
  if tf.math.equal((server_state.round_num + 1) % num_sub_rounds, 0):
    return accumulate_server_votes_and_decode(server_state, sub_round_votes,
                                              max_num_heavy_hitters,
                                              default_terminator)
  else:
    return accumulate_server_votes(server_state, sub_round_votes)
