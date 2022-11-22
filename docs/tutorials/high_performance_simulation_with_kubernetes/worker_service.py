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
"""TFF worker service."""

import collections
from collections.abc import Mapping, Sequence
import random
from absl import app
import tensorflow as tf
import tensorflow_federated as tff

# Path to sqlite database containing EMNIST partition.
_EMNIST_PARTITION_PATH = '/root/worker/data/emnist_partition.sqlite'

_PORT = 8000
_GRPC_OPTIONS = [('grpc.max_message_length', 20 * 1024 * 1024),
                 ('grpc.max_receive_message_length', 20 * 1024 * 1024)]
# Number of worker threads in thread pool.
_THREADS = 10

_NUM_EPOCHS = 5
_SHUFFLE_BUFFER = 100


class _EMNISTPartitionDataBackend(tff.framework.DataBackend):
  """Loads a partition of the EMNIST dataset and returns examples uniformly at random.
  """

  def __init__(self):
    element_spec = collections.OrderedDict([
        ('label', tf.TensorSpec(shape=(), dtype=tf.int32, name=None)),
        ('pixels', tf.TensorSpec(shape=(28, 28), dtype=tf.float32, name=None))
    ])

    self._partition = tff.simulation.datasets.sql_client_data_utils.load_and_parse_sql_client_data(
        database_filepath=_EMNIST_PARTITION_PATH, element_spec=element_spec)

  def preprocess(self, dataset: tf.data.Dataset):

    def map_fn(element: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
      return collections.OrderedDict(
          x=tf.reshape(element['pixels'], [-1, 784]),
          y=tf.reshape(element['label'], [-1, 1]))

    return dataset.repeat(_NUM_EPOCHS).shuffle(
        _SHUFFLE_BUFFER, seed=1).map(map_fn)

  async def materialize(self, data, type_spec: tff.Type):
    client_id = random.sample(population=self._partition.client_ids, k=1)[0]

    return self.preprocess(
        self._partition.create_tf_dataset_for_client(client_id))


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  def ex_fn(device: tf.config.LogicalDevice) -> tff.framework.DataExecutor:
    return tff.framework.DataExecutor(
        tff.framework.EagerTFExecutor(device),
        data_backend=_EMNISTPartitionDataBackend())

  executor_factory = tff.framework.local_executor_factory(
      default_num_clients=1,
      # Max fanout in the hierarchy of local executors
      max_fanout=100,
      leaf_executor_fn=ex_fn)

  tff.simulation.run_server(executor_factory, _THREADS, _PORT, None,
                            _GRPC_OPTIONS)


if __name__ == '__main__':
  app.run(main)
