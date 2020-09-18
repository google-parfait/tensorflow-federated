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

import collections
import os
from typing import Optional, Tuple

import tensorflow as tf

from tensorflow_federated.python.simulation import client_data
from tensorflow_federated.python.simulation import from_tensor_slices_client_data
from tensorflow_federated.python.simulation import hdf5_client_data


def load_data(
    cache_dir: Optional[str] = None
) -> Tuple[client_data.ClientData, client_data.ClientData]:
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


def get_synthetic() -> client_data.ClientData:
  """Creates `tff.simulation.ClientData` for a synthetic in-memory example of Shakespeare.

  The returned `tff.simulation.ClientData` will have the same data schema as
  `load_data()`, but uses a very small set of client data loaded in-memory.

  This synthetic data is useful for validation in small tests.

  Returns:
    A `tff.simulation.ClientData` of synthentic Shakespeare text.
  """
  return from_tensor_slices_client_data.FromTensorSlicesClientData(
      _SYNTHETIC_SHAKESPEARE_DATA)


# A small sub-sample of snippets from the Shakespeare dataset.
_SYNTHETIC_SHAKESPEARE_DATA = {
    'THE_TRAGEDY_OF_KING_LEAR_MACBETH':
        collections.OrderedDict(snippets=[
            b'Hark!',
            b'When?',
            b"My name's Macbeth.",
            b"'Twas a rough fight.",
            b'Came they not by you?',
            b'No, nor more fearful.',
        ]),
    'MUCH_ADO_ABOUT_NOTHING_EMILIA':
        collections.OrderedDict(snippets=[
            b'Never.',
            b'But now, my lord.',
            b'How if fair and foolish?',
            b'Is not this man jealous?',
            b'Why, with my lord, madam.',
            b'[Within.] I do beseech you',
        ]),
    'THE_TAMING_OF_THE_SHREW_CUPID':
        collections.OrderedDict(snippets=[
            b'Hail to thee, worthy Timon, and to all',
        ]),
}
