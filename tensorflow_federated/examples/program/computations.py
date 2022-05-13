# Copyright 2022, The TensorFlow Federated Authors.
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
"""An example of computations to use in a federated program.

These computations compute the sum of the integer data accross all clients.
"""

import collections
from typing import Any, Tuple

import tensorflow as tf
import tensorflow_federated as tff

METRICS_TOTAL_SUM = 'total_sum'


@tff.tf_computation()
def initialize() -> int:
  """Returns the initial state."""
  return 0


@tff.tf_computation(tff.SequenceType(tf.int32))
def _sum_dataset(dataset: tf.data.Dataset) -> int:
  """Returns the sum of all the integers in `dataset`."""
  return dataset.reduce(tf.cast(0, tf.int32), tf.add)


@tff.tf_computation(tf.int32, tf.int32)
def _sum_integers(x: int, y: int) -> int:
  """Returns the sum two integers."""
  return x + y


@tff.federated_computation(
    tff.type_at_server(tf.int32),
    tff.type_at_clients(tff.SequenceType(tf.int32)))
def train(
    server_state: int, client_data: tf.data.Dataset
) -> Tuple[int, collections.OrderedDict[str, Any]]:
  """Computes the sum of all the integers on the clients.

  Computes the sum of all the integers on the clients, updates the server state,
  and returns the updated server state and the following metrics:

  * `sum_client_data.METRICS_TOTAL_SUM`: The sum of all the client_data on the
    clients.

  Args:
    server_state: The server state.
    client_data: The data on the clients.

  Returns:
    A tuple of the updated server state and the train metrics.
  """
  client_sums = tff.federated_map(_sum_dataset, client_data)
  total_sum = tff.federated_sum(client_sums)
  updated_state = tff.federated_map(_sum_integers, (server_state, total_sum))
  metrics = collections.OrderedDict([
      (METRICS_TOTAL_SUM, total_sum),
  ])
  return updated_state, metrics


@tff.federated_computation(
    tff.type_at_server(tf.int32),
    tff.type_at_clients(tff.SequenceType(tf.int32)))
def evaluation(
    server_state: int,
    client_data: tf.data.Dataset) -> collections.OrderedDict[str, Any]:
  """Computes the sum of all the integers on the clients.

  Computes the sum of all the integers on the clients and returns the following
  metrics:

  * `sum_client_data.METRICS_TOTAL_SUM`: The sum of all the client_data on the
    clients.

  Args:
    server_state: The server state.
    client_data: The data on the clients.

  Returns:
    The evaluation metrics.
  """
  del server_state  # Unused.
  client_sums = tff.federated_map(_sum_dataset, client_data)
  total_sum = tff.federated_sum(client_sums)
  metrics = collections.OrderedDict([
      (METRICS_TOTAL_SUM, total_sum),
  ])
  return metrics
