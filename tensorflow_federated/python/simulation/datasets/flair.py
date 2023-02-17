# Copyright 2023, The TensorFlow Federated Authors.
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
"""Libraries for loading the FLAIR dataset.

For details on the FLAIR dataset, please see the associated paper:

FLAIR: Federated Learning Annotated Image Repository
    Congzheng Song, Filip Granqvist, Kunal Talwar
    https://arxiv.org/abs/2207.08869

To load the dataset, first call `download_data`, followed by `write_data`. Once
this is done, you can call `load_data` without having to re-download or re-write
the data.

NOTE: Please ensure that you use the same `data_dir` argument in both
`download_data` and `write_data`. Similarly, please ensure that you use the same
`cache_dir` in both `write_data` and `load_data`.
"""

import collections
import json
import multiprocessing
import os.path
from typing import Any
import urllib.parse

from absl import logging
import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import client_data
from tensorflow_federated.python.simulation.datasets import file_per_user_client_data


_CLIENT_DATA = client_data.ClientData

# URL and file paths
_METADATA_FILENAME = 'flair_metadata.json'
_TAR_PREFIX = 'small_images'
_IMAGE_SUBDIR = 'images'

# Dataset features
_IMAGE_FEATURE = 'image'
_LABEL_FEATURE = 'label'
_FINE_GRAINED_LABEL_FEATURE = 'fine_grained_label'

# Dataset information
_IMAGE_SHAPE = (256, 256, 3)
_NUM_LABELS = 17
_NUM_FINEGRAINED_LABELS = 1628


def download_data(data_dir: str) -> None:
  """Downloads and unpacks the raw FLAIR data into a given directory."""
  if not tf.io.gfile.exists(data_dir):
    tf.io.gfile.makedirs(data_dir)
  base_url = (
      'https://docs-assets.developer.apple.com/ml-research/datasets/flair/'
  )
  metadata_url = urllib.parse.urljoin(
      base_url, 'labels/labels_and_metadata.json'
  )
  metadata_path = os.path.join(data_dir, _METADATA_FILENAME)
  logging.info('Downloading metadata...')
  tf.keras.utils.get_file(
      fname=metadata_path, origin=metadata_url, extract=False
  )
  logging.info('Finished downloading metadata.')

  def download_image_bucket(image_bucket_url: str) -> None:
    tf.keras.utils.get_file(
        origin=image_bucket_url,
        extract=True,
        cache_dir=data_dir,
        cache_subdir=_IMAGE_SUBDIR,
    )

  image_buckets = [
      f'images/small/{_TAR_PREFIX}-{x:02d}.tar.gz' for x in range(43)
  ]
  image_bucket_urls = [
      urllib.parse.urljoin(base_url, bucket) for bucket in image_buckets
  ]
  logging.info('Downloading and unpacking images...')
  with multiprocessing.pool.ThreadPool(multiprocessing.cpu_count()) as pool:
    pool.map(download_image_bucket, image_bucket_urls)
  logging.info('Finished downloading images.')


def _create_label_index(
    label_set: set[str],
) -> collections.OrderedDict[str, int]:
  """Creates a mapping from a set of keys to their sorted order."""
  label_index_pairs = [
      (label, index) for index, label in enumerate(sorted(label_set))
  ]
  return collections.OrderedDict(label_index_pairs)


def _process_metadata(
    metadata: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, int], dict[str, int]]:
  """Extracts dataset client and label metadata from FLAIR metadata entries.

  This method takes the example `metadata` for flair. This is a list of
  dictionaries. Each dictionary represents a single example, and holds key value
  pairs for `user_id` (a string id for the client), `labels` (a list of strings
  corresponding to the example's labels) and `fine_grained_labels` (a list of
  strings corresponding to the example's fine-grained labels).

  In order to write the FLAIR dataset, this method processes the metadata in
  three ways. First, it creates `client_metadata`, a `user_id`-keyed dictionary
  whose values are lists of all example metadata for a given client. Second,
  it creates a mapping from the labels to their index in sorted alphabetical
  order. Last, it creates the same mapping from fine-grained labels to their
  index in sorted alphabetical order. These last two maps are for the purposes
  of associating integer-valued labels with each example.

  Args:
    metadata: A list of metadata entries for the FLAIR dataset.

  Returns:
    A tuple `client_metadata, labels_to_index, fine_grained_labels_to_index`.
    Here, `client_metadata` is a dictionary mapping (string) client ids to
    a list of their example metadta, while `labels_to_index` and
    `fine_grained_labels_to_index` are dictionaries that map label strings to
    their index in sorted alphabetical order.
  """
  client_metadata = collections.defaultdict(list)
  label_set = set([])
  fine_grained_label_set = set([])
  for example_entry in metadata:
    client_metadata[example_entry['user_id']].append(example_entry)
    label_set.update(example_entry['labels'])
    fine_grained_label_set.update(example_entry['fine_grained_labels'])

  labels_to_index = _create_label_index(label_set)
  fine_grained_labels_to_index = _create_label_index(fine_grained_label_set)
  return client_metadata, labels_to_index, fine_grained_labels_to_index


def _create_example(
    image: bytes, labels: list[int], fine_grained_labels: list[int]
) -> tf.train.Example:
  """Create a `tf.train.Example` for a given image and labels."""
  image_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
  label_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
  fine_grained_label_feature = tf.train.Feature(
      int64_list=tf.train.Int64List(value=fine_grained_labels)
  )
  feature_dict = collections.OrderedDict([
      (_IMAGE_FEATURE, image_feature),
      (_LABEL_FEATURE, label_feature),
      (_FINE_GRAINED_LABEL_FEATURE, fine_grained_label_feature),
  ])
  return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def _load_examples(
    example_metadata: list[dict[str, Any]],
    labels_to_index: dict[str, int],
    fine_grained_labels_to_index: dict[str, int],
    image_dir: str,
) -> list[tf.train.Example]:
  """Loads a list of `tf.train.Example`s from a given directory."""
  examples = []
  for metadata in example_metadata:
    image_id = metadata['image_id']
    image_path = os.path.join(image_dir, f'{image_id}.jpg')
    with tf.io.gfile.GFile(image_path, 'rb') as f:
      image = f.read()
    labels = [labels_to_index[label] for label in metadata['labels']]
    fine_grained_labels = [
        fine_grained_labels_to_index[fg_label]
        for fg_label in metadata['fine_grained_labels']
    ]
    example = _create_example(image, labels, fine_grained_labels)
    examples.append(example)

  return examples


def _write_examples(
    client_metadata: dict[str, Any],
    labels_to_index: dict[str, int],
    fine_grained_labels_to_index: dict[str, int],
    image_dir: str,
    cache_dir: str,
) -> None:
  """Writes examples for each client to a TFRecords file."""
  logging.info('Writing TFRecords files...')
  num_clients = len(client_metadata)
  for client_index, client_id in enumerate(client_metadata):
    example_metadata = client_metadata[client_id]
    client_examples = _load_examples(
        example_metadata,
        labels_to_index,
        fine_grained_labels_to_index,
        image_dir,
    )

    # This should be one of 'train', 'val', or 'test'. Note that each client in
    # the FLAIR dataset has a *unique* split (ie. if one of their examples has
    # the split 'val', then so too do all of the client's examples). In other
    # words, the dataset splits have non-overlapping client sets.
    split = client_metadata[client_id][0]['partition']

    # We now write the examples to a TFRecords file under a subfolder for the
    # given split. This will make it easier to create separate
    # `tff.simulation.datasets.ClientData` objects for each split.
    split_dir = os.path.join(cache_dir, split)
    tf.io.gfile.makedirs(split_dir)
    client_example_file = os.path.join(split_dir, f'{client_id}.tfrecords')
    with tf.io.TFRecordWriter(client_example_file) as writer:
      for example in client_examples:
        writer.write(example.SerializeToString())

    if (client_index + 1) % 500 == 0:
      logging.info(
          'Processed %s out of %s clients.', client_index + 1, num_clients
      )

  logging.info('Finished writing all examples.')


def write_data(data_dir: str, cache_dir: str) -> None:
  """Writes the raw FLAIR images to a processed dataset format.

  Args:
    data_dir: The dataset to load FLAIR images from. Note that `write_data` must
      only be called after `download_data` (and with the same directory).
    cache_dir: The directory used to write the processed FLAIR dataset.
  """
  if not tf.io.gfile.exists(data_dir):
    raise ValueError(
        f'The directory {data_dir} does not exist. Please ensure '
        'that you run `download_data` first.'
    )
  if not tf.io.gfile.exists(cache_dir):
    tf.io.gfile.makedirs(cache_dir)

  raw_metadata_file = os.path.join(data_dir, _METADATA_FILENAME)
  with tf.io.gfile.GFile(raw_metadata_file, 'r') as f:
    raw_metadata = json.load(f)

  client_metadata, labels_to_index, fine_grained_labels_to_index = (
      _process_metadata(raw_metadata)
  )
  image_dir = os.path.join(data_dir, _IMAGE_SUBDIR, _TAR_PREFIX)
  _write_examples(
      client_metadata,
      labels_to_index,
      fine_grained_labels_to_index,
      image_dir,
      cache_dir,
  )


def _parse_example(example_proto) -> collections.OrderedDict[str, tf.Tensor]:
  """Parse an example to image and label in tensorflow tensor format."""
  feature_description = {
      _IMAGE_FEATURE: tf.io.FixedLenFeature([], tf.string),
      _LABEL_FEATURE: tf.io.VarLenFeature(tf.int64),
      _FINE_GRAINED_LABEL_FEATURE: tf.io.VarLenFeature(tf.int64),
  }
  example = tf.io.parse_single_example(example_proto, feature_description)
  image = tf.io.decode_jpeg(example[_IMAGE_FEATURE])
  # We reshape the image to the correct size in order to get static typing.
  image = tf.reshape(image, _IMAGE_SHAPE)
  one_hot_labels = tf.one_hot(
      example[_LABEL_FEATURE].values, depth=_NUM_LABELS, dtype=tf.int32
  )
  labels = tf.reduce_sum(one_hot_labels, axis=0)
  one_hot_fine_grained_labels = tf.one_hot(
      example[_FINE_GRAINED_LABEL_FEATURE].values,
      depth=_NUM_FINEGRAINED_LABELS,
      dtype=tf.int32,
  )
  fine_grained_labels = tf.reduce_sum(one_hot_fine_grained_labels, axis=0)
  return collections.OrderedDict([
      (_IMAGE_FEATURE, image),
      (_LABEL_FEATURE, labels),
      (_FINE_GRAINED_LABEL_FEATURE, fine_grained_labels),
  ])


def _load_tfrecords(filename: str) -> tf.data.Dataset:
  """Load tfrecords from `filename` and return a `tf.data.Dataset`."""
  dataset = tf.data.TFRecordDataset([filename])
  return dataset.map(_parse_example, tf.data.AUTOTUNE)


def _get_client_ids_to_files(cache_dir: str, split: str) -> dict[str, str]:
  """Get the dataset filename for a each client in a given split."""
  split_dir = os.path.join(cache_dir, split)
  split_files = tf.io.gfile.listdir(split_dir)
  client_id_file_pairs = []
  for file in split_files:
    client_id = os.path.splitext(file)[0]
    client_file = os.path.join(split_dir, file)
    client_id_file_pairs.append((client_id, client_file))
  return collections.OrderedDict(client_id_file_pairs)


def load_data(
    cache_dir: str,
) -> tuple[_CLIENT_DATA, _CLIENT_DATA, _CLIENT_DATA]:
  """Loads the FLAIR dataset.

  Args:
    cache_dir: A string representing the directory containing the output of
      `write_data`.

  Returns:
    A three-tuple of `tff.simulation.datasets.ClientData` representing the
    train, validation, and test split of the FLAIR dataset.
  """
  datasets = []
  for split in ['train', 'val', 'test']:
    client_ids_to_files = _get_client_ids_to_files(cache_dir, split)
    split_client_data = file_per_user_client_data.FilePerUserClientData(
        client_ids_to_files=client_ids_to_files,
        dataset_fn=_load_tfrecords,
    )
    datasets.append(split_client_data)
  return tuple(datasets)
