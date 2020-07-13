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

import os.path

import tensorflow as tf

from tensorflow_federated.python.simulation import hdf5_client_data


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
  `tff.simulation.ClientData.create_tf_dataset_for_client` will yield
  `collections.OrderedDict` objects at each iteration, with the following keys
  and values:

    -   `'coarse_label'`: a `tf.Tensor` with `dtype=tf.int64` and shape [1] that
        corresponds to the coarse label of the associated image. Labels are
        in the range [0-19].
    -   `'image'`: a `tf.Tensor` with `dtype=tf.uint8` and shape [32, 32, 3],
        corresponding to the pixels of the handwritten digit, with values in
        the range [0, 255].
    -   `'label'`: a `tf.Tensor` with `dtype=tf.int64` and shape [1], the class
        label of the corresponding image. Labels are in the range [0-99].

  Args:
    cache_dir: (Optional) directory to cache the downloaded file. If `None`,
      caches in Keras' default cache directory.

  Returns:
    Tuple of (train, test) where the tuple elements are
    `tff.simulation.ClientData` objects.
  """
  path = tf.keras.utils.get_file(
      'fed_cifar100.tar.bz2',
      origin='https://storage.googleapis.com/tff-datasets-public/fed_cifar100.tar.bz2',
      file_hash='e8575e22c038ecef1ce6c7d492d7abee7da13b1e1ba9b70a7fc18531ba7590de',
      hash_algorithm='sha256',
      extract=True,
      archive_format='tar',
      cache_dir=cache_dir)

  dir_path = os.path.dirname(path)
  train_client_data = hdf5_client_data.HDF5ClientData(
      os.path.join(dir_path, 'fed_cifar100_train.h5'))
  test_client_data = hdf5_client_data.HDF5ClientData(
      os.path.join(dir_path, 'fed_cifar100_test.h5'))

  return train_client_data, test_client_data
