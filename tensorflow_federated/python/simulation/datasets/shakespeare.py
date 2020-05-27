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
"""Libraries for the Shakespeare dataset for federated learning simulation."""

import os.path

import tensorflow as tf

from tensorflow_federated.python.simulation import hdf5_client_data


def load_data(cache_dir=None):
  """Loads the federated Shakespeare dataset.

  Downloads and caches the dataset locally. If previously downloaded, tries to
  load the dataset from cache.

  This dataset is derived from the Leaf repository
  (https://github.com/TalwalkarLab/leaf) pre-processing on the works of
  Shakespeare, which is published in "LEAF: A Benchmark for Federated Settings"
  https://arxiv.org/abs/1812.01097.

  The data set consists of 715 users (characters of Shakespeare plays), where
  each
  example corresponds to a contiguous set of lines spoken by the character in a
  given play.

  Data set sizes:

  -   train: 16,068 examples
  -   test: 2,356 examples

  Rather than holding out specific users, each user's examples are split across
  _train_ and _test_ so that all users have at least one example in _train_ and
  one example in _test_. Characters that had less than 2 examples are excluded
  from the data set.

  The `tf.data.Datasets` returned by
  `tff.simulation.ClientData.create_tf_dataset_for_client` will yield
  `collections.OrderedDict` objects at each iteration, with the following keys
  and values:

    -   `'snippets'`: a `tf.Tensor` with `dtype=tf.string`, the snippet of
      contiguous text.

  Args:
    cache_dir: (Optional) directory to cache the downloaded file. If `None`,
      caches in Keras' default cache directory.

  Returns:
    Tuple of (train, test) where the tuple elements are
    `tff.simulation.ClientData` objects.
  """
  path = tf.keras.utils.get_file(
      'shakespeare.tar.bz2',
      origin='https://storage.googleapis.com/tff-datasets-public/shakespeare.tar.bz2',
      file_hash='0285be9906cb5f268092eee4edeeacfc2af4574f2941f7cc2f08a321d7f5c707',
      hash_algorithm='sha256',
      extract=True,
      archive_format='tar',
      cache_dir=cache_dir)

  dir_path = os.path.dirname(path)
  train_client_data = hdf5_client_data.HDF5ClientData(
      os.path.join(dir_path, 'shakespeare_train.h5'))
  test_client_data = hdf5_client_data.HDF5ClientData(
      os.path.join(dir_path, 'shakespeare_test.h5'))

  return train_client_data, test_client_data
