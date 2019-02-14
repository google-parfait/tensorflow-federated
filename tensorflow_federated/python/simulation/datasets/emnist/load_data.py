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
"""Libraries for the federated EMNIST dataset for simulation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf

from tensorflow_federated.python.simulation import hdf5_client_data


def load_data(only_digits=True, cache_dir=None):
  """Loads the Federated EMNIST dataset.

  Downloads and caches the dataset locally. If previously downloaded, tries to
  load the dataset from cache.

  This dataset is derived from the Leaf repository
  (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
  dataset, grouping examples by writer. Details about Leaf were published in
  "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.

  Data set sizes:

  *only_digits=True*: 3,383 users, 10 label classes

  -   train: 341,873 examples
  -   test: 40,832 examples

  *only_digits=False*: 3,400 users, 62 label classes

  -   train: 671,585 examples
  -   test: 77,483 examples

  Rather than holding out specific users, each user's examples are split across
  _train_ and _test_ so that all users have at least one example in _train_ and
  one example in _test_. Writers that had less than 2 examples are excluded from
  the data set.

  The `tf.data.Datasets` returned by
  `tff.simulation.ClientData.create_tf_dataset_for_client` will yield
  `colletions.OrderedDict` objects at each iteration, with the following keys
  and values:

    -   `'pixels'`: a `tf.Tensor` with `dtype=tf.float32` and shape [28, 28],
        containing the pixels of the handwritten digit.
    -   `'label'`: a `tf.Tensor` with `dtype=tf.int32` and shape [1], the class
        label of the corresponding pixels.

  Args:
    only_digits: (Optional) whether to only include examples that are from the
      digits [0-9] classes. If `False`, includes lower and upper case
      characters, for a total of 62 class labels.
    cache_dir: (Optional) directory to cache the downloaded file. If `None`,
      caches in Keras' default cache directory.

  Returns:
    Tuple of (train, test) where the tuple elements are
    `tff.simulation.ClientData` objects.
  """
  if only_digits:
    fileprefix = 'fed_emnist_digitsonly'
    sha256 = '55333deb8546765427c385710ca5e7301e16f4ed8b60c1dc5ae224b42bd5b14b'
  else:
    fileprefix = 'fed_emnist'
    sha256 = 'fe1ed5a502cea3a952eb105920bff8cffb32836b5173cb18a57a32c3606f3ea0'

  filename = fileprefix + '.tar.bz2'
  path = tf.keras.utils.get_file(
      filename,
      origin='https://storage.googleapis.com/tff-datasets-public/' + filename,
      file_hash=sha256,
      hash_algorithm='sha256',
      extract=True,
      archive_format='tar',
      cache_dir=cache_dir)

  dir_path = os.path.dirname(path)
  train_client_data = hdf5_client_data.HDF5ClientData(
      os.path.join(dir_path, fileprefix + '_train.h5'))
  test_client_data = hdf5_client_data.HDF5ClientData(
      os.path.join(dir_path, fileprefix + '_test.h5'))

  return train_client_data, test_client_data
