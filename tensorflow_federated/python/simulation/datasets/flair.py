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
"""Libraries for loading the federated FLAIR dataset."""

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
# Logging constants
_LOG_INTERVAL = 500

# Image and URL constants
_DATA_URL = (
    'https://docs-assets.developer.apple.com/ml-research/datasets/flair/'
)
_NUM_IMAGE_BATCHES = 43
_TAR_PREFIX = 'small_images'
_SMALL_IMAGE_URLS = [
    urllib.parse.urljoin(
        _DATA_URL, f'images/small/{_TAR_PREFIX}-{str(i).zfill(2)}.tar.gz'
    )
    for i in range(_NUM_IMAGE_BATCHES)
]
_LABELS_AND_META_DATA_URL = urllib.parse.urljoin(
    _DATA_URL, 'labels/labels_and_metadata.json'
)
_LABEL_RELATIONSHIP_URL = urllib.parse.urljoin(
    _DATA_URL, 'labels/label_relationship.txt'
)
_LABEL_INDEX_FILE = 'label_to_index.json'
_IMAGE_SUBDIR = 'images'
_PARTITIONS = ['train', 'val', 'test']

# TFRecords constants
_KEY_IMAGE_BYTES = 'image/encoded_jpeg'
_KEY_IMAGE = 'image'
_KEY_LABELS = 'labels'
_KEY_FINE_GRAINED_LABELS = 'fine_grained_labels'
_IMAGE_SHAPE = (256, 256, 3)
_NUM_LABELS = 17
_NUM_FINEGRAINED_LABELS = 1628


def download_data(dataset_dir: str):
  """Downloads the FLAIR metadata and images."""
  if not tf.io.gfile.exists(dataset_dir):
    tf.io.gfile.makedirs(dataset_dir)
  logging.info('Downloading label metadata...')
  label_metadata_path = os.path.join(
      dataset_dir, os.path.basename(_LABELS_AND_META_DATA_URL)
  )
  tf.keras.utils.get_file(
      fname=label_metadata_path, origin=_LABELS_AND_META_DATA_URL, extract=False
  )
  label_relation_path = os.path.join(
      dataset_dir, os.path.basename(_LABEL_RELATIONSHIP_URL)
  )
  tf.keras.utils.get_file(
      fname=label_relation_path, origin=_LABEL_RELATIONSHIP_URL, extract=False
  )

  logging.info('Finished downloading label metadata.')
  logging.info('Downloading and unpacking images...')

  def download_image_batch(image_url):
    tf.keras.utils.get_file(
        origin=image_url,
        extract=True,
        cache_dir=dataset_dir,
        cache_subdir=_IMAGE_SUBDIR,
    )

  with multiprocessing.pool.ThreadPool(multiprocessing.cpu_count()) as pool:
    pool.map(download_image_batch, _SMALL_IMAGE_URLS)
  logging.info('Finished downloading images.')


def _load_client_metadata(
    metadata_file: str,
) -> tuple[dict[str, Any], set[str], set[str]]:
  """Loads client metadata for the FLAIR dataset."""
  client_metadata = collections.defaultdict(list)
  with tf.io.gfile.GFile(metadata_file, 'r') as f:
    metadata_list = json.load(f)
  label_set = set([])
  fine_grained_label_set = set([])

  for metadata in metadata_list:
    client_metadata[metadata['user_id']].append(metadata)
    label_set.update(metadata['labels'])
    fine_grained_label_set.update(metadata['fine_grained_labels'])
  return client_metadata, label_set, fine_grained_label_set


def _create_example(
    image_bytes: bytes, labels: list[int], fine_grained_labels: list[int]
) -> tf.train.Example:
  """Create a `tf.train.Example` for a given image and labels."""
  features = collections.OrderedDict([
      (
          _KEY_IMAGE_BYTES,
          tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
      ),
      (
          _KEY_LABELS,
          tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
      ),
      (
          _KEY_FINE_GRAINED_LABELS,
          tf.train.Feature(
              int64_list=tf.train.Int64List(value=fine_grained_labels)
          ),
      ),
  ])
  return tf.train.Example(features=tf.train.Features(feature=features))


def _write_tfrecords(
    client_metadata: dict[str, Any],
    labels_to_index: dict[str, int],
    fine_grained_labels_to_index: dict[str, int],
    image_dir: str,
    tfrecords_dir: str,
) -> None:
  """Writes TFRecords files for the FLAIR dataset."""
  logging.info('Writing label index file...')
  label_index_file = os.path.join(tfrecords_dir, _LABEL_INDEX_FILE)
  with tf.io.gfile.GFile(label_index_file, 'w') as f:
    json.dump(
        {
            'labels': labels_to_index,
            'fine_grained_labels': fine_grained_labels_to_index,
        },
        f,
        indent=4,
    )

  logging.info('Finished writing label index file.')
  logging.info('Preparing TFRecords files...')
  for i, client_id in enumerate(client_metadata):
    partition = client_metadata[client_id][0]['partition']

    # Load and concatenate all images and labels of a user.
    client_examples = []
    for metadata in client_metadata[client_id]:
      image_id = metadata['image_id']
      image_path = os.path.join(image_dir, f'{image_id}.jpg')
      with tf.io.gfile.GFile(image_path, 'rb') as f:
        image_bytes = f.read()
      example = _create_example(
          image_bytes=image_bytes,
          labels=[labels_to_index[label] for label in metadata['labels']],
          fine_grained_labels=[
              fine_grained_labels_to_index[label]
              for label in metadata['fine_grained_labels']
          ],
      )
      client_examples.append(example)

    partition_dir = os.path.join(tfrecords_dir, partition)
    tf.io.gfile.makedirs(partition_dir)
    client_example_file = os.path.join(partition_dir, f'{client_id}.tfrecords')
    with tf.io.TFRecordWriter(client_example_file) as writer:
      for example in client_examples:
        writer.write(example.SerializeToString())

    if (i + 1) % _LOG_INTERVAL == 0:
      logging.info('Processed %s/%s clients', i + 1, len(client_metadata))

  logging.info('Finished preparing TFRecords files.')


def _create_label_index(
    label_set: set[str],
) -> collections.OrderedDict[str, int]:
  """Creates a mapping from a set of keys to their sorted order."""
  label_index_pairs = [
      (label, index) for index, label in enumerate(sorted(label_set))
  ]
  return collections.OrderedDict(label_index_pairs)


def prepare_data(dataset_dir: str, tfrecords_dir: str) -> None:
  """Prepares the FLAIR dataset in TFRecords format."""
  if not tf.io.gfile.exists(dataset_dir):
    raise ValueError(f'image_dir {dataset_dir} does not exist')
  if not tf.io.gfile.exists(tfrecords_dir):
    tf.io.gfile.makedirs(tfrecords_dir)

  metadata_file_basename = os.path.basename(_LABELS_AND_META_DATA_URL)
  metadata_file = os.path.join(dataset_dir, metadata_file_basename)
  client_metadata, label_set, fine_grained_label_set = _load_client_metadata(
      metadata_file
  )
  labels_to_index = _create_label_index(label_set)
  fine_grained_labels_to_index = _create_label_index(fine_grained_label_set)
  image_dir = os.path.join(dataset_dir, _IMAGE_SUBDIR, _TAR_PREFIX)
  _write_tfrecords(
      client_metadata=client_metadata,
      labels_to_index=labels_to_index,
      fine_grained_labels_to_index=fine_grained_labels_to_index,
      image_dir=image_dir,
      tfrecords_dir=tfrecords_dir,
  )


def _parse_example(example_proto) -> collections.OrderedDict[str, tf.Tensor]:
  """Parse an example to image and label in tensorflow tensor format."""
  feature_description = {
      _KEY_IMAGE_BYTES: tf.io.FixedLenFeature([], tf.string),
      _KEY_LABELS: tf.io.VarLenFeature(tf.int64),
      _KEY_FINE_GRAINED_LABELS: tf.io.VarLenFeature(tf.int64),
  }
  example = tf.io.parse_single_example(example_proto, feature_description)
  image = tf.io.decode_jpeg(example[_KEY_IMAGE_BYTES])
  image = tf.reshape(image, _IMAGE_SHAPE)
  labels = tf.reduce_sum(
      tf.one_hot(
          example[_KEY_LABELS].values, depth=_NUM_LABELS, dtype=tf.int32
      ),
      axis=0,
  )
  fine_grained_labels = tf.reduce_sum(
      tf.one_hot(
          example[_KEY_FINE_GRAINED_LABELS].values,
          depth=_NUM_FINEGRAINED_LABELS,
          dtype=tf.int32,
      ),
      axis=0,
  )
  return collections.OrderedDict([
      (_KEY_IMAGE, image),
      (_KEY_LABELS, labels),
      (_KEY_FINE_GRAINED_LABELS, fine_grained_labels),
  ])


def _load_tfrecords(filename: str) -> tf.data.Dataset:
  """Load tfrecords from `filename` and return a `tf.data.Dataset`."""
  dataset = tf.data.TFRecordDataset([filename])
  return dataset.map(_parse_example, tf.data.AUTOTUNE)


def _get_client_ids_to_files(
    tfrecords_dir: str, partition: str
) -> dict[str, str]:
  """Get the tfrecords filenames for a dataset partition."""
  partition_dir = os.path.join(tfrecords_dir, partition)
  partition_client_files = os.listdir(partition_dir)
  return {
      client_file.split('.tfrecords')[0]: os.path.join(
          partition_dir, client_file
      )
      for client_file in partition_client_files
  }


def load_data(
    tfrecords_dir: str,
) -> tuple[_CLIENT_DATA, _CLIENT_DATA, _CLIENT_DATA]:
  """Loads the FLAIR dataset."""
  datasets = []
  for partition in _PARTITIONS:
    client_ids_to_files = _get_client_ids_to_files(tfrecords_dir, partition)
    partition_client_data = file_per_user_client_data.FilePerUserClientData(
        client_ids_to_files=client_ids_to_files,
        dataset_fn=_load_tfrecords,
    )
    datasets.append(partition_client_data)
  return tuple(datasets)
