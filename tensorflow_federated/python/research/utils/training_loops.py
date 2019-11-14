# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
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
"""TFF training loops."""

import logging
import time

import tensorflow_federated as tff


def federated_averaging_training_loop(model_fn,
                                      server_optimizer_fn,
                                      client_datasets_fn,
                                      total_rounds=10,
                                      rounds_per_eval=1,
                                      metrics_hook=lambda *args: None,
                                      client_weight_fn=None,
                                      stateful_model_broadcast_fn=None,
                                      stateful_delta_aggregate_fn=None):
  """A simple example of training loop for the Federated Averaging algorithm.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.
    client_datasets_fn: A function that takes the round number, and returns a
      list of `tf.data.Datset`, one per client.
    total_rounds: Number of rounds to train.
    rounds_per_eval: How often to call the  `metrics_hook` function.
    metrics_hook: A function taking arguments (server_state, train_metrics,
      round_num) and performs evaluation. Optional.
    client_weight_fn: Optional function that takes the output of
      `model.report_local_outputs` and returns a tensor that provides the weight
      in the federated average of model deltas. If not provided, the default is
      the total number of examples processed on device.
    stateful_model_broadcast_fn: A `tff.utils.StatefulBroadcastFn`. (See
      documentation for `tff.learning.build_federated_averaging_process`.)
    stateful_delta_aggregate_fn: A `tff.utils.StatefulAggregateFn`. (See
      documentation for `tff.learning.build_federated_averaging_process`.)

  Returns:
    Final `ServerState`.
  """

  logging.info('Starting federated_training_loop')

  iterative_process = tff.learning.build_federated_averaging_process(
      model_fn,
      server_optimizer_fn=server_optimizer_fn,
      client_weight_fn=client_weight_fn,
      stateful_model_broadcast_fn=stateful_model_broadcast_fn,
      stateful_delta_aggregate_fn=stateful_delta_aggregate_fn)

  server_state = iterative_process.initialize()
  train_metrics = {}

  start_time = time.time()

  for round_num in range(total_rounds):
    if round_num % rounds_per_eval == 0:
      metrics_hook(server_state, train_metrics, round_num)

    federated_train_data = client_datasets_fn(round_num)
    server_state, train_metrics = iterative_process.next(
        server_state, federated_train_data)

    train_metrics = train_metrics._asdict(recursive=True)
    logging.info('Round {:2d}, {:.2f}s per round in average.'.format(
        round_num, (time.time() - start_time) / (round_num + 1)))

  metrics_hook(server_state, train_metrics, total_rounds)

  return server_state
