# Copyright 2021, The TensorFlow Federated Authors.
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
"""Libraries for the federated CelebA dataset for simulation."""

import collections

import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import download
from tensorflow_federated.python.simulation.datasets import sql_client_data

IMAGE_NAME = 'image'
ATTRIBUTE_NAMES = [
    'five_o_clock_shadow',
    'arched_eyebrows',
    'attractive',
    'bags_under_eyes',
    'bald',
    'bangs',
    'big_lips',
    'big_nose',
    'black_hair',
    'blond_hair',
    'blurry',
    'brown_hair',
    'bushy_eyebrows',
    'chubby',
    'double_chin',
    'eyeglasses',
    'goatee',
    'gray_hair',
    'heavy_makeup',
    'high_cheekbones',
    'male',
    'mouth_slightly_open',
    'mustache',
    'narrow_eyes',
    'no_beard',
    'oval_face',
    'pale_skin',
    'pointy_nose',
    'receding_hairline',
    'rosy_cheeks',
    'sideburns',
    'smiling',
    'straight_hair',
    'wavy_hair',
    'wearing_earrings',
    'wearing_hat',
    'wearing_lipstick',
    'wearing_necklace',
    'wearing_necktie',
    'young',
]


def _add_proto_parsing(dataset: tf.data.Dataset) -> tf.data.Dataset:
  """Add parsing of the tf.Example proto to the dataset pipeline."""

  def parse_proto(tensor_proto):
    parse_spec = collections.OrderedDict(
        sorted(
            [(
                IMAGE_NAME,
                tf.io.FixedLenFeature(shape=(84, 84, 3), dtype=tf.int64),
            )]
            + [
                (
                    attribute_name,
                    tf.io.FixedLenFeature(shape=(), dtype=tf.int64),
                )
                for attribute_name in ATTRIBUTE_NAMES
            ]
        )
    )
    parsed_features = tf.io.parse_example(tensor_proto, parse_spec)
    return collections.OrderedDict(
        sorted(
            [(IMAGE_NAME, parsed_features[IMAGE_NAME])]
            + [
                (
                    attribute_name,
                    tf.cast(parsed_features[attribute_name], tf.bool),
                )
                for attribute_name in ATTRIBUTE_NAMES
            ]
        )
    )

  return dataset.map(parse_proto, num_parallel_calls=tf.data.AUTOTUNE)


def load_data(split_by_clients=True, cache_dir=None):
  """Loads the Federated CelebA dataset.

  Downloads and caches the dataset locally. If previously downloaded, tries to
  load the dataset from cache.

  This dataset is derived from the
  [LEAF repository](https://github.com/TalwalkarLab/leaf) preprocessing of the
  [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html),
  grouping examples by celebrity id. Details about LEAF were published in
  ["LEAF: A Benchmark for Federated
  Settings"](https://arxiv.org/abs/1812.01097), and details about CelebA were
  published in ["Deep Learning Face Attributes in the
  Wild"](https://arxiv.org/abs/1411.7766).

  The raw CelebA dataset contains 10,177 unique identities. During LEAF
  preprocessing, all clients with less than 5 examples are removed; this leaves
  9,343 clients.

  The data is available with train and test splits by clients or by examples.
  That is, when split by clients, ~90% of clients are selected for the train
  set, ~10% of clients are selected for test, and all the examples for a given
  user are part of the same data split.  When split by examples, each client is
  located in both the train data and the test data, with ~90% of the examples
  on each client selected for train and ~10% of the examples selected for test.

  Data set sizes:

  *split_by_clients=True*:

  -   train: 8,408 clients, 180,429 total examples
  -   test: 935 clients, 19,859 total examples

  *split_by_clients=False*:

  -   train: 9,343 clients, 177,457 total examples
  -   test: 9,343 clients, 22,831 total examples

  The `tf.data.Datasets` returned by
  `tff.simulation.datasets.ClientData.create_tf_dataset_for_client` will yield
  `collections.OrderedDict` objects at each iteration. These objects have a
  key/value pair storing the image of the celebrity:

    -   `'image'`: a `tf.Tensor` with `dtype=tf.int64` and shape [84, 84, 3],
        containing the red/blue/green pixels of the image. Each pixel is a value
        in the range [0, 255].

  The OrderedDict objects also contain an additional 40 key/value pairs for the
  celebrity image attributes, each of the format:

    -   `{attribute name}`: a `tf.Tensor` with `dtype=tf.bool` and shape [1],
        set to True if the celebrity has this attribute in the image, or False
        if they don't.

  The attribute names are:
    'five_o_clock_shadow', 'arched_eyebrows', 'attractive', 'bags_under_eyes',
    'bald', 'bangs', 'big_lips', 'big_nose', 'black_hair', 'blond_hair',
    'blurry', 'brown_hair', 'bushy_eyebrows', 'chubby', 'double_chin',
    'eyeglasses', 'goatee', 'gray_hair', 'heavy_makeup', 'high_cheekbones',
    'male', 'mouth_slightly_open', 'mustache', 'narrow_eyes', 'no_beard',
    'oval_face', 'pale_skin', 'pointy_nose', 'receding_hairline', 'rosy_cheeks',
    'sideburns', 'smiling', 'straight_hair', 'wavy_hair', 'wearing_earrings',
    'wearing_hat', 'wearing_lipstick', 'wearing_necklace', 'wearing_necktie',
    'young'

  Note: The CelebA dataset may contain potential bias. The
  [fairness indicators TF tutorial](
  https://www.tensorflow.org/responsible_ai/fairness_indicators/tutorials/Fairness_Indicators_TFCO_CelebA_Case_Study)
  goes into detail about several considerations to keep in mind while using the
  CelebA dataset.

  Args:
    split_by_clients: There are 9,343 clients in the federated CelebA dataset
      with 5 or more examples. If this argument is True, clients are divided
      into train and test groups, with 8,408 and 935 clients respectively. If
      this argument is False, the data is divided by examples instead, i.e., all
      clients participate in both the train and test groups, with ~90% of the
      examples belonging to the train group and the rest belonging to the test
      group.
    cache_dir: (Optional) directory to cache the downloaded file. If `None`,
      caches in Keras' default cache directory.

  Returns:
    Tuple of `(train, test)` where the tuple elements are
    `tff.simulation.datasets.ClientData` objects.
  """
  database_path = download.get_compressed_file(
      origin='https://storage.googleapis.com/tff-datasets-public/celeba.sqlite.lzma',
      cache_dir=cache_dir,
  )
  if split_by_clients:
    train_client_data = sql_client_data.SqlClientData(
        database_path, 'split_by_clients_train'
    ).preprocess(_add_proto_parsing)
    test_client_data = sql_client_data.SqlClientData(
        database_path, 'split_by_clients_test'
    ).preprocess(_add_proto_parsing)
  else:
    train_client_data = sql_client_data.SqlClientData(
        database_path, 'split_by_examples_train'
    ).preprocess(_add_proto_parsing)
    test_client_data = sql_client_data.SqlClientData(
        database_path, 'split_by_examples_test'
    ).preprocess(_add_proto_parsing)
  return train_client_data, test_client_data
