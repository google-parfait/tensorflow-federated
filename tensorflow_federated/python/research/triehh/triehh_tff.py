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

from typing import List

import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.research.triehh.triehh_tf import client_update
from tensorflow_federated.python.research.triehh.triehh_tf import server_update
from tensorflow_federated.python.research.triehh.triehh_tf import ServerState


def build_triehh_process(possible_prefix_extensions: List[str],
                         num_sub_rounds: int,
                         max_num_heavy_hitters: int,
                         max_user_contribution: int,
                         default_terminator: str = '$'):
  """Builds the TFF computations for heavy hitters discovery with TrieHH.

  TrieHH works by interactively keeping track of popular prefixes. In each
  round, the server broadcasts the popular prefixes it has
  discovered so far and the list of `possible_prefix_extensions` to a small
  fraction of selected clients. The select clients sample
  `max_user_contributions` words from their local datasets, and use them to vote
  on character extensions to the broadcasted popular prefixes. Client votes are
  accumulated across `num_sub_rounds` rounds, and then the top
  `max_num_heavy_hitters` extensions are used to extend the already discovered
  prefixes, and the extended prefixes are used in the next round. When an
  already discovered prefix is extended by `default_terminator` it is added to
  the list of discovered heavy hitters.

  Args:
    possible_prefix_extensions: A list containing all the possible extensions to
      learned prefixes. Each extensions must be a single character strings.
    num_sub_rounds: The total number of sub rounds to be executed before
      decoding aggregated votes. Must be positive.
    max_num_heavy_hitters: The maximum number of discoverable heavy hitters.
      Must be positive.
    max_user_contribution: The maximum number of examples a user can contribute.
      Must be positive.
    default_terminator: The end of sequence symbol.

  Returns:
    A `tff.utils.IterativeProcess`.
  """

  @tff.tf_computation
  def server_init_tf():
    return ServerState(
        discovered_heavy_hitters=tf.constant([], dtype=tf.string),
        discovered_prefixes=tf.constant([''], dtype=tf.string),
        possible_prefix_extensions=tf.constant(
            possible_prefix_extensions, dtype=tf.string),
        round_num=tf.constant(0, dtype=tf.int32),
        accumulated_votes=tf.zeros(
            dtype=tf.int32,
            shape=[max_num_heavy_hitters,
                   len(possible_prefix_extensions)]))

  # We cannot use server_init_tf.type_signature.result because the
  # discovered_* fields need to have [None] shapes, since they will grow over
  # time.
  server_state_type = (
      tff.to_type(
          ServerState(
              discovered_heavy_hitters=tff.TensorType(
                  dtype=tf.string, shape=[None]),
              discovered_prefixes=tff.TensorType(dtype=tf.string, shape=[None]),
              possible_prefix_extensions=tff.TensorType(
                  dtype=tf.string, shape=[len(possible_prefix_extensions)]),
              round_num=tff.TensorType(dtype=tf.int32, shape=[]),
              accumulated_votes=tff.TensorType(
                  dtype=tf.int32, shape=[None,
                                         len(possible_prefix_extensions)]),
          )))

  sub_round_votes_type = tff.TensorType(
      dtype=tf.int32,
      shape=[max_num_heavy_hitters,
             len(possible_prefix_extensions)])

  @tff.tf_computation(server_state_type, sub_round_votes_type)
  @tf.function
  def server_update_fn(server_state, sub_round_votes):
    server_state = server_update(
        server_state,
        sub_round_votes,
        num_sub_rounds=tf.constant(num_sub_rounds),
        max_num_heavy_hitters=tf.constant(max_num_heavy_hitters),
        default_terminator=tf.constant(default_terminator, dtype=tf.string))
    return server_state

  tf_dataset_type = tff.SequenceType(tf.string)
  discovered_prefixes_type = tff.TensorType(dtype=tf.string, shape=[None])
  round_num_type = tff.TensorType(dtype=tf.int32, shape=[])

  @tff.tf_computation(tf_dataset_type, discovered_prefixes_type, round_num_type)
  @tf.function
  def client_update_fn(tf_dataset, discovered_prefixes, round_num):
    result = client_update(tf_dataset, discovered_prefixes,
                           tf.constant(possible_prefix_extensions), round_num,
                           num_sub_rounds, max_num_heavy_hitters,
                           max_user_contribution)
    return result

  federated_server_state_type = tff.FederatedType(server_state_type, tff.SERVER)
  federated_dataset_type = tff.FederatedType(
      tf_dataset_type, tff.CLIENTS, all_equal=False)

  @tff.federated_computation(federated_server_state_type,
                             federated_dataset_type)
  def run_one_round(server_state, federated_dataset):
    """Orchestration logic for one round of TrieHH computation.

    Args:
      server_state: A `ServerState`.
      federated_dataset: A federated `tf.Dataset` with placement `tff.CLIENTS`.

    Returns:
      An updated `ServerState`
    """
    discovered_prefixes = tff.federated_broadcast(
        server_state.discovered_prefixes)
    round_num = tff.federated_broadcast(server_state.round_num)

    client_outputs = tff.federated_map(
        client_update_fn,
        tff.federated_zip([federated_dataset, discovered_prefixes, round_num]))

    accumulated_votes = tff.federated_sum(client_outputs.client_votes)

    server_state = tff.federated_map(server_update_fn,
                                     (server_state, accumulated_votes))

    server_output = tff.federated_value([], tff.SERVER)

    return server_state, server_output

  return tff.utils.IterativeProcess(
      initialize_fn=tff.federated_computation(
          lambda: tff.federated_value(server_init_tf(), tff.SERVER)),
      next_fn=run_one_round)
