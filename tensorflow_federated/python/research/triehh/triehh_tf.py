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

DEFAULT_VALUE = -1  # The value to use if a key is missing in the hash table.


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


# TODO(b/150700360): Consolidate reused TF functions in a common utils library.
@tf.function
def get_top_elements(list_of_elements, max_user_contribution):
  """Gets the top max_user_contribution words from the input list.

  Note that the returned set of top words will not necessarily be sorted.

  Args:
    list_of_elements: A tensor containing a list of elements.
    max_user_contribution: The maximum number of elements to keep.

  Returns:
    A tensor of a list of strings.
    If the total number of unique words is less than or equal to
    max_user_contribution, returns the set of unique words.
  """
  words, _, counts = tf.unique_with_counts(list_of_elements)
  if tf.size(words) > max_user_contribution:
    # This logic is influenced by the focus on global heavy hitters and
    # thus implements clipping by chopping the tail of the distribution
    # of the words as present on a single client. Another option could
    # be to provide pick max_words_per_user random words out of the unique
    # words present locally.
    top_indices = tf.argsort(
        counts, axis=-1, direction='DESCENDING')[:max_user_contribution]
    top_words = tf.gather(words, top_indices)
    return top_words
  return words


@tf.function()
def listify(dataset):
  """Turns a stream of strings into a 1D tensor of strings."""
  data = tf.constant([], dtype=tf.string)
  for item in dataset:
    # Empty datasets return a zero tf.float32 tensor for some reason.
    # so we need to protect against that.
    if item.dtype == tf.string:
      items = tf.expand_dims(item, 0)
      data = tf.concat([data, items], axis=0)
  return data


def make_accumulate_client_votes_fn(round_num, discovered_prefixes_table,
                                    possible_prefix_extensions_table):
  """Returns a reduce function that is used to accumulate client votes.

  This function creates an accumulate_client_votes reduce function that can be
  consumed by a tf.data.Dataset.reduce method. The reduce function maps
  (old_state, example) to a new_state. It must take two arguments and return a
  new element with a structure that matches that of the initial_state.

  Args:
    round_num: A tf.constant containing the round number.
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
    if tf.strings.length(example) <= round_num:
      return vote_accumulator
    else:
      discovered_prefixes_index = discovered_prefixes_table.lookup(
          tf.strings.substr(example, 0, round_num))
      possible_prefix_extensions_index = possible_prefix_extensions_table.lookup(
          tf.strings.substr(example, round_num, 1))

      if (discovered_prefixes_index == DEFAULT_VALUE) or (
          possible_prefix_extensions_index == DEFAULT_VALUE):
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
                  round_num, max_num_heavy_hitters, max_user_contribution):
  """Creates a ClientOutput object hold the client's votes.

  Args:
    dataset: A 'tf.data.Dataset'.
    discovered_prefixes: A tf.string containing candidate prefixes.
    possible_prefix_extensions: A tf.string of shape (num_discovered_prefixes, )
      containing possible prefix extensions.
    round_num: A tf.constant dictating the algorithm's round number.
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
      round_num, discovered_prefixes_table, possible_prefix_extensions_table)

  sampled_data = tf.data.Dataset.from_tensor_slices(
      get_top_elements(listify(dataset), max_user_contribution))

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
  if (server_state.round_num + 1) % num_sub_rounds == 0:
    return accumulate_server_votes_and_decode(server_state, sub_round_votes,
                                              max_num_heavy_hitters,
                                              default_terminator)
  else:
    return accumulate_server_votes(server_state, sub_round_votes)
