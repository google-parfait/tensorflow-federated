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
"""Libraries for the Stackoverflow dataset for federated learning simulation."""

import collections
import json
import os.path

import tensorflow as tf

from tensorflow_federated.python.simulation import hdf5_client_data


def load_data(cache_dir=None):
  """Loads the federated Stack Overflow dataset.

  Downloads and caches the dataset locally. If previously downloaded, tries to
  load the dataset from cache.

  This dataset is derived from the Stack Overflow Data hosted by kaggle.com and
  available to query through Kernels using the BigQuery API:
  https://www.kaggle.com/stackoverflow/stackoverflow. The Stack Overflow Data
  is licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported
  License. To view a copy of this license, visit
  http://creativecommons.org/licenses/by-sa/3.0/ or send a letter to
  Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

  The data consists of the body text of all questions and answers. The bodies
  were parsed into sentences, and any user with fewer than 100 sentences was
  expunged from the data. Minimal preprocessing was performed as follows:

  1. Lowercase the text,
  2. Unescape HTML symbols,
  3. Remove non-ascii symbols,
  4. Separate punctuation as individual tokens (except apostrophes and hyphens),
  5. Removing extraneous whitespace,
  6. Replacing URLS with a special token.

  In addition the following metadata is available:

  1. Creation date
  2. Question title
  3. Question tags
  4. Question score
  5. Type ('question' or 'answer')

  The data is divided into three sets:

    -   Train: Data before 2018-01-01 UTC except the held-out users. 342,477
        unique users with 135,818,730 examples.
    -   Held-out: All examples from users with user_id % 10 == 0 (all dates).
        38,758 unique users with 16,491,230 examples.
    -   Test: All examples after 2018-01-01 UTC except from held-out users.
        204,088 unique users with 16,586,035 examples.

  The `tf.data.Datasets` returned by
  `tff.simulation.ClientData.create_tf_dataset_for_client` will yield
  `collections.OrderedDict` objects at each iteration, with the following keys
  and values:

    -   `'creation_date'`: a `tf.Tensor` with `dtype=tf.string` and shape []
        containing the date/time of the question or answer in UTC format.
    -   `'title'`: a `tf.Tensor` with `dtype=tf.string` and shape [] containing
        the title of the question.
    -   `'score'`: a `tf.Tensor` with `dtype=tf.int64` and shape [] containing
        the score of the question.
    -   `'tags'`: a `tf.Tensor` with `dtype=tf.string` and shape [] containing
        the tags of the question, separated by '|' characters.
    -   `'tokens'`: a `tf.Tensor` with `dtype=tf.string` and shape []
        containing the tokens of the question/answer, separated by space (' ')
        characters.
    -   `'type'`: a `tf.Tensor` with `dtype=tf.string` and shape []
        containing either the string 'question' or 'answer'.

  Args:
    cache_dir: (Optional) directory to cache the downloaded file. If `None`,
      caches in Keras' default cache directory.

  Returns:
    Tuple of (train, held_out, test) where the tuple elements are
    `tff.simulation.ClientData` objects.
  """
  path = tf.keras.utils.get_file(
      'stackoverflow.tar.bz2',
      origin='https://storage.googleapis.com/tff-datasets-public/stackoverflow.tar.bz2',
      file_hash='99eca2f8b8327a09e5fc123979df2d237acbc5e52322f6d86bf523ee47b961a2',
      hash_algorithm='sha256',
      extract=True,
      archive_format='tar',
      cache_dir=cache_dir)

  dir_path = os.path.dirname(path)
  train_client_data = hdf5_client_data.HDF5ClientData(
      os.path.join(dir_path, 'stackoverflow_train.h5'))
  held_out_client_data = hdf5_client_data.HDF5ClientData(
      os.path.join(dir_path, 'stackoverflow_held_out.h5'))
  test_client_data = hdf5_client_data.HDF5ClientData(
      os.path.join(dir_path, 'stackoverflow_test.h5'))

  return train_client_data, held_out_client_data, test_client_data


def load_word_counts(cache_dir=None):
  """Loads the word counts for the Stack Overflow dataset.

  Args:
    cache_dir: (Optional) directory to cache the downloaded file. If `None`,
      caches in Keras' default cache directory.

  Returns:
    A collections.OrderedDict where the keys are string tokens, and the values
    are the counts of unique users who have at least one example in the training
    set containing that token in the body text.
  """
  path = tf.keras.utils.get_file(
      'stackoverflow.word_count.tar.bz2',
      origin='https://storage.googleapis.com/tff-datasets-public/stackoverflow.word_count.tar.bz2',
      file_hash='1dc00256d6e527c54b9756d968118378ae14e6692c0b3b6cad470cdd3f0c519c',
      hash_algorithm='sha256',
      extract=True,
      archive_format='tar',
      cache_dir=cache_dir)

  word_counts = collections.OrderedDict()
  dir_path = os.path.dirname(path)
  file_path = os.path.join(dir_path, 'stackoverflow.word_count')
  with open(file_path) as f:
    for line in f:
      word, count = line.split()
      word_counts[word] = int(count)
  return word_counts


def load_tag_counts(cache_dir=None):
  """Loads the tag counts for the Stack Overflow dataset.

  Args:
    cache_dir: (Optional) directory to cache the downloaded file. If `None`,
      caches in Keras' default cache directory.

  Returns:
    A collections.OrderedDict where the keys are string tags, and the values
    are the counts of unique users who have at least one example in the training
    set containing with that tag. The dictionary items are in decreasing order
    of tag frequency.
  """
  path = tf.keras.utils.get_file(
      'stackoverflow.tag_count.tar.bz2',
      origin='https://storage.googleapis.com/tff-datasets-public/stackoverflow.tag_count.tar.bz2',
      file_hash='6fe281cec490d9384a290d560072438e7e2b377bbb823876ce7bd6f82696772d',
      hash_algorithm='sha256',
      extract=True,
      archive_format='tar',
      cache_dir=cache_dir)

  dir_path = os.path.dirname(path)
  file_path = os.path.join(dir_path, 'stackoverflow.tag_count')
  with open(file_path) as f:
    tag_counts = json.load(f)
  return collections.OrderedDict(
      sorted(tag_counts.items(), key=lambda item: item[1], reverse=True))
