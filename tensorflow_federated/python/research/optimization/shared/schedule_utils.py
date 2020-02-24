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
"""Library for sequentiality scheduling utilities in federated training."""

import numpy as np


def build_preprocess_fn(batch_size: int, epochs_per_round: int,
                        shuffle_then_repeat: bool, shuffle_buffer_size: int):
  """Builds a preprocessing function for shuffle, repeat, and batch operations."""
  if shuffle_then_repeat:

    def preprocess_fn(dataset):
      return dataset.shuffle(buffer_size=shuffle_buffer_size).repeat(
          epochs_per_round).batch(batch_size)
  else:

    def preprocess_fn(dataset):
      return dataset.repeat(epochs_per_round).shuffle(
          buffer_size=shuffle_buffer_size).batch(batch_size)

  return preprocess_fn


def build_scheduled_client_datasets_fn(train_dataset,
                                       clients_per_round,
                                       client_batch_size,
                                       client_epochs_per_round,
                                       total_rounds,
                                       num_stages=1,
                                       batch_growth_factor=1,
                                       epochs_decrease_amount=0,
                                       shuffle_then_repeat=True,
                                       shuffle_buffer_size=10000):
  """Builds the function for generating client datasets at each round.

  Args:
    train_dataset: A `tff.simulation.ClientData` object.
    clients_per_round: The number of client participants in each round.
    client_batch_size: An integer, the batch size on the clients.
    client_epochs_per_round: An integer, the number of local client epochs.
    total_rounds: The total number of training rounds.
    num_stages: The number of stages in a batch size/client epochs schedule.
    batch_growth_factor: A nonnegative integer specifying how much to increase
      the client batch size at each stage of the schedule. This is done
      multiplicatively, so that batch_growth_factor=1 is a no-op.
    epochs_decrease_amount: A nonnegative integer specifying how much to
      decrease the client epochs at each stage of the schedule. The change is
      performed via subtraction, so that epochs_decrease_amount=0 is a no-op. If
      the number of client epochs is ever nonpositive, it is set to 1.
    shuffle_then_repeat: A boolean. If True, we shuffle then repeat the client
      datasets (which prevents shuffling between epochs).
    shuffle_buffer_size: An integer specifying the buffersize for shuffling a
      `tf.data.Dataset` object.

  Raises:
    ValueError: If batch_growth_factor is not a nonnegative integer.
    ValueError: If epochs_decrease_amount is not a nonnegative integer.

  Returns:
    A function which returns a list of `tff.simulation.ClientData` objects at a
    given round round_num.
  """

  if not isinstance(batch_growth_factor, int) or batch_growth_factor < 0:
    raise ValueError(
        'Argument "batch_growth_factor" must be a nonnegative integer.')
  if not isinstance(epochs_decrease_amount, int) or epochs_decrease_amount < 0:
    raise ValueError(
        'Argument "epochs_decrease_amount" must be a nonnegative integer.')

  rounds_per_stage = total_rounds // num_stages
  total_rounds_no_remainder = rounds_per_stage * num_stages

  def client_datasets_fn(round_num):
    """Returns a list of preprocessed client datasets for a given round."""
    stage_num = min(round_num,
                    total_rounds_no_remainder - 1) // rounds_per_stage

    sampled_clients = np.random.choice(
        train_dataset.client_ids, size=clients_per_round, replace=False)
    current_batch_size = int(client_batch_size *
                             batch_growth_factor**(stage_num))
    current_client_epochs = max(
        1, int(client_epochs_per_round - (stage_num) * epochs_decrease_amount))

    preprocess_fn = build_preprocess_fn(current_batch_size,
                                        current_client_epochs,
                                        shuffle_then_repeat,
                                        shuffle_buffer_size)
    sampled_client_datasets = [
        preprocess_fn(train_dataset.create_tf_dataset_for_client(client))
        for client in sampled_clients
    ]
    return sampled_client_datasets, sampled_clients

  return client_datasets_fn
