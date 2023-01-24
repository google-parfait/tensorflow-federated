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
"""Utility methods for vision simulation datasets: gldv2 and iNaturalist."""

import collections
import csv
import logging
import os
from typing import ByteString

import tensorflow as tf

from tensorflow_federated.python.simulation.datasets.client_data import ClientData
from tensorflow_federated.python.simulation.datasets.file_per_user_client_data import FilePerUserClientData

KEY_IMAGE_BYTES = 'image/encoded_jpeg'
KEY_IMAGE_DECODED = 'image/decoded'
KEY_CLASS = 'class'


def read_csv(path: str) -> list[dict[str, str]]:
  """Reads a csv file, and returns the content inside a list of dictionaries.

  Args:
    path: The path to the csv file.

  Returns:
    A list of dictionaries. Each row in the csv file will be a list entry. The
    dictionary is keyed by the column names.
  """
  with open(path, 'r') as f:
    return list(csv.DictReader(f))


def load_tfrecord(fname: str, logger_tag: str) -> tf.data.Dataset:
  """Reads a `tf.data.Dataset` from a TFRecord file.

  Parse each element into a `tf.train.Example`.

  Args:
    fname: The file name of the TFRecord file.
    logger_tag: The tag for the logger.

  Returns:
    `tf.data.Dataset`.
  """
  logger = logging.getLogger(logger_tag)
  logger.info('Start loading dataset for file %s', fname)
  raw_dataset = tf.data.TFRecordDataset([fname])

  def _parse(example_proto):
    feature_description = {
        KEY_IMAGE_BYTES: tf.io.FixedLenFeature([], tf.string, default_value=''),
        KEY_CLASS: tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }
    return collections.OrderedDict(
        tf.io.parse_single_example(example_proto, feature_description)
    )

  ds = raw_dataset.map(_parse)

  def _transform(item):
    return collections.OrderedDict([
        (KEY_IMAGE_DECODED, tf.io.decode_jpeg(item[KEY_IMAGE_BYTES])),
        (KEY_CLASS, tf.reshape(item[KEY_CLASS], [1])),
    ])

  ds = ds.map(_transform)
  logger.info('Finished loading dataset for file %s', fname)
  return ds


def load_data_from_cache(
    cache_dir: str, train_sub_dir: str, test_file_name: str, logger_tag: str
) -> tuple[ClientData, tf.data.Dataset]:
  """Load train and test data from the TFRecord files.

  Args:
    cache_dir: The directory containing the TFRecord files.
    train_sub_dir: The sub-directory for keeping the training data files.
    test_file_name: The file name for the test data.
    logger_tag: The tag for the logger.

  Returns:
    A tuple of `ClientData`, `tf.data.Dataset`.
  """
  logger = logging.getLogger(logger_tag)
  train_dir = os.path.join(cache_dir, train_sub_dir)
  logger.info('Start to load train data from cache directory: %s', train_dir)
  train = FilePerUserClientData.create_from_dir(train_dir, load_tfrecord)
  logger.info('Finish loading train data from cache directory: %s', train_dir)
  test_file = os.path.join(cache_dir, test_file_name)
  logger.info('Start to load test data from file: %s', test_file)
  test = load_tfrecord(test_file, logger_tag)
  logger.info('Finish loading test data from file: %s', test_file)
  return train, test


def create_example(image_bytes: bytes, label: int) -> tf.train.Example:
  return tf.train.Example(
      features=tf.train.Features(
          feature={
              KEY_IMAGE_BYTES: tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[image_bytes])
              ),
              KEY_CLASS: tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[label])
              ),
          }
      )
  )


def decode_bytes(
    key_bytes: ByteString, serialized_value_bytes: ByteString
) -> dict[str, tf.Tensor]:
  """Convert a serialized `tf.train.Example` to a feature dict."""
  del key_bytes  # Unused.
  feature_description = {
      KEY_IMAGE_BYTES: tf.io.FixedLenFeature([], tf.string, default_value=''),
      KEY_CLASS: tf.io.FixedLenFeature([], tf.int64, default_value=-1),
  }
  example = tf.io.parse_single_example(
      serialized_value_bytes, feature_description
  )
  return collections.OrderedDict([
      (KEY_IMAGE_DECODED, tf.io.decode_jpeg(example[KEY_IMAGE_BYTES])),
      (KEY_CLASS, tf.reshape(example[KEY_CLASS], [1])),
  ])
