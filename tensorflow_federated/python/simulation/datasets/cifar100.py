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
"""Libraries for the federated CIFAR-100 dataset for simulation."""

import collections

import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import download
from tensorflow_federated.python.simulation.datasets import sql_client_data


def _add_proto_parsing(dataset: tf.data.Dataset) -> tf.data.Dataset:
  """Add parsing of the tf.Example proto to the dataset pipeline."""

  def parse_proto(tensor_proto):
    parse_spec = {
        'coarse_label': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
        'image': tf.io.FixedLenFeature(shape=(32, 32, 3), dtype=tf.int64),
        'label': tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
    }
    parsed_features = tf.io.parse_example(tensor_proto, parse_spec)
    return collections.OrderedDict(
        coarse_label=parsed_features['coarse_label'],
        image=tf.cast(parsed_features['image'], tf.uint8),
        label=parsed_features['label'])

  return dataset.map(parse_proto, num_parallel_calls=tf.data.AUTOTUNE)


def load_data(cache_dir=None):
  """Loads a federated version of the CIFAR-100 dataset.

  The dataset is downloaded and cached locally. If previously downloaded, it
  tries to load the dataset from cache.

  The dataset is derived from the [CIFAR-100
  dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The training and
  testing examples are partitioned across 500 and 100 clients (respectively).
  No clients share any data samples, so it is a true partition of CIFAR-100. The
  train clients have string client IDs in the range [0-499], while the test
  clients have string client IDs in the range [0-99]. The train clients form a
  true partition of the CIFAR-100 training split, while the test clients form a
  true partition of the CIFAR-100 testing split.

  The data partitioning is done using a hierarchical Latent Dirichlet Allocation
  (LDA) process, referred to as the [Pachinko Allocation Method]
  (https://people.cs.umass.edu/~mccallum/papers/pam-icml06.pdf) (PAM).
  This method uses a two-stage LDA process, where each client has an associated
  multinomial distribution over the coarse labels of CIFAR-100, and a
  coarse-to-fine label multinomial distribution for that coarse label over the
  labels under that coarse label. The coarse label multinomial is drawn from a
  symmetric Dirichlet with parameter 0.1, and each coarse-to-fine multinomial
  distribution is drawn from a symmetric Dirichlet with parameter 10. Each
  client has 100 samples. To generate a sample for the client, we first select
  a coarse label by drawing from the coarse label multinomial distribution, and
  then draw a fine label using the coarse-to-fine multinomial distribution. We
  then randomly draw a sample from CIFAR-100 with that label (without
  replacement). If this exhausts the set of samples with this label, we
  remove the label from the coarse-to-fine multinomial and renormalize the
  multinomial distribution.

  Data set sizes:
  -   train: 500,000 examples
  -   test: 100,000 examples

  The `tf.data.Datasets` returned by
  `tff.simulation.datasets.ClientData.create_tf_dataset_for_client` will yield
  `collections.OrderedDict` objects at each iteration, with the following keys
  and values, in lexicographic order by key:

    -   `'coarse_label'`: a `tf.Tensor` with `dtype=tf.int64` and shape [1] that
        corresponds to the coarse label of the associated image. Labels are
        in the range [0-19].
    -   `'image'`: a `tf.Tensor` with `dtype=tf.uint8` and shape [32, 32, 3],
        containing the red/blue/green pixels of the image. Each pixel is a value
        in the range [0, 255].
    -   `'label'`: a `tf.Tensor` with `dtype=tf.int64` and shape [1], the class
        label of the corresponding image. Labels are in the range [0-99].

  Args:
    cache_dir: (Optional) directory to cache the downloaded file. If `None`,
      caches in Keras' default cache directory.

  Returns:
    Tuple of (train, test) where the tuple elements are
    `tff.simulation.datasets.ClientData` objects.
  """
  database_path = download.get_compressed_file(
      origin='https://storage.googleapis.com/tff-datasets-public/cifar100.sqlite.lzma',
      cache_dir=cache_dir)
  train_client_data = sql_client_data.SqlClientData(
      database_path, 'train').preprocess(_add_proto_parsing)
  test_client_data = sql_client_data.SqlClientData(
      database_path, 'test').preprocess(_add_proto_parsing)
  return train_client_data, test_client_data
