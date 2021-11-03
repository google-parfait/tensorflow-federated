# Copyright 2020, The TensorFlow Federated Authors.
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
"""Libraries for the federated Google Landmark v2 dataset for simulation."""

import collections
import logging
import logging.handlers
import multiprocessing.pool
import os
import shutil
import sys
import tempfile
import traceback

from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import vision_datasets_utils
from tensorflow_federated.python.simulation.datasets.client_data import ClientData

FED_GLD_SPLIT_FILE_BUNDLE = 'landmarks-user-160k'
FED_GLD_SPLIT_FILE_DOWNLOAD_URL = 'http://storage.googleapis.com/gresearch/federated-vision-datasets/%s.zip' % FED_GLD_SPLIT_FILE_BUNDLE
FED_GLD_SPLIT_FILE_BUNDLE_MD5_CHECKSUM = '53c36bd7d5fc12f927af2820b7e4a57c'
FED_GLD_TRAIN_SPLIT_FILE = 'federated_train.csv'
FED_GLD_TEST_SPLIT_FILE = 'test.csv'
GLD_SHARD_BASE_URL = 'https://s3.amazonaws.com/google-landmark'
NUM_SHARD_TRAIN = 500
MINI_GLD_TRAIN_DOWNLOAD_URL = 'https://storage.googleapis.com/tff-datasets-public/mini_gld_train_split.csv'
MINI_GLD_TRAIN_SPLIT_FILE = 'mini_gld_train_split.csv'
MINI_GLD_TEST_DOWNLOAD_URL = 'https://storage.googleapis.com/tff-datasets-public/mini_gld_test.csv'
MINI_GLD_TEST_SPLIT_FILE = 'mini_gld_test.csv'
MINI_GLD_TRAIN_SPLIT_FILE_MD5_CHECKSUM = '9fd62cf79a67046fdd673d3a0ac52841'
MINI_GLD_TEST_SPLIT_FILE_MD5_CHECKSUM = '298e9d19d66357236f66fe8e22920933'
FED_GLD_CACHE = 'gld160k'
MINI_GLD_CACHE = 'gld23k'
TRAIN_SUB_DIR = 'train'
TEST_FILE_NAME = 'test.tfRecord'
LOGGER = 'gldv2'


def _listener_process(queue: multiprocessing.Queue, log_file: str):
  """Sets up a separate process for handling logging messages.

  This setup is required because without it, the logging messages will be
  duplicated when multiple processes are created for downloading GLD dataset.

  Args:
    queue: The queue to receive logging messages.
    log_file: The file which the messages will be written to.
  """
  root = logging.getLogger()
  h = logging.FileHandler(log_file)
  fmt = logging.Formatter(
      fmt='%(asctime)s %(levelname)-8s %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S')
  h.setFormatter(fmt)
  root.addHandler(h)
  while True:
    try:
      record = queue.get()
      # We send None as signal to stop
      if record is None:
        break
      logger = logging.getLogger(record.name)
      logger.handle(record)
    except Exception:  # pylint: disable=broad-except
      print('Something went wrong:', file=sys.stderr)
      traceback.print_exc(file=sys.stderr)


def _create_dataset_with_mapping(
    image_dir: str, mapping: List[Dict[str, str]]) -> List[tf.train.Example]:
  """Builds a dataset based on the mapping file and the images in the image dir.

  Args:
    image_dir: The directory contains the image files.
    mapping: A list of dictionaries. Each dictionary contains 'image_id' and
      'class' columns.

  Returns:
    A list of `tf.train.Example`.
  """
  logger = logging.getLogger(LOGGER)
  examples = []
  for row in mapping:
    img_path = os.path.join(image_dir, '%s.jpg' % row['image_id'])
    try:
      with open(img_path, 'rb') as f:
        img_bytes = f.read()
        examples.append(
            vision_datasets_utils.create_example(img_bytes, int(row['class'])))
    except IOError as e:
      logger.warning('Image %s is not found. Exception: %s', img_path, e)
      continue
  return examples


def _create_train_data_files(cache_dir: str, image_dir: str, mapping_file: str):
  """Create the train data and persist it into a separate file per user.

  Args:
    cache_dir: The directory caching the intermediate results.
    image_dir: The directory containing all the downloaded images.
    mapping_file: The file containing 'image_id' to 'class' mappings.
  """
  logger = logging.getLogger(LOGGER)
  if not os.path.isdir(image_dir):
    logger.error('Image directory %s does not exist', image_dir)
    raise ValueError('%s does not exist or is not a directory' % image_dir)

  mapping_table = vision_datasets_utils.read_csv(mapping_file)
  expected_cols = ['user_id', 'image_id', 'class']
  if not all(col in mapping_table[0].keys() for col in expected_cols):
    logger.error('%s has wrong format.', mapping_file)
    raise ValueError(
        'The mapping file must contain user_id, image_id and class columns. '
        'The existing columns are %s' % ','.join(mapping_table[0].keys()))
  if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
  mapping_per_user = collections.defaultdict(list)
  for row in mapping_table:
    user_id = row['user_id']
    mapping_per_user[user_id].append(row)
  for user_id, data in mapping_per_user.items():
    examples = _create_dataset_with_mapping(image_dir, data)
    with tf.io.TFRecordWriter(os.path.join(cache_dir, str(user_id))) as writer:
      for example in examples:
        writer.write(example.SerializeToString())
      logger.info('Created tfrecord file for user %s with %d examples, at %s',
                  user_id, len(examples), cache_dir)


def _create_test_data_file(cache_dir: str, image_dir: str, mapping_file: str):
  """Create the test data and persist it into a file.

  Args:
    cache_dir: The directory caching the intermediate results.
    image_dir: The directory containing all the downloaded images.
    mapping_file: The file containing 'image_id' to 'class' mappings.
  """
  logger = logging.getLogger(LOGGER)
  if not os.path.isdir(image_dir):
    logger.error('Image directory %s does not exist', image_dir)
    raise ValueError('%s does not exist or is not a directory' % image_dir)
  mapping_table = vision_datasets_utils.read_csv(mapping_file)
  expected_cols = ['image_id', 'class']
  if not all(col in mapping_table[0].keys() for col in expected_cols):
    logger.error('%s has wrong format.', mapping_file)
    raise ValueError(
        'The mapping file must contain image_id and class columns. The existing'
        ' columns are %s' % ','.join(mapping_table[0].keys()))
  if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
  examples = _create_dataset_with_mapping(image_dir, mapping_table)
  with tf.io.TFRecordWriter(os.path.join(cache_dir, TEST_FILE_NAME)) as writer:
    for example in examples:
      writer.write(example.SerializeToString())
    logger.info('Created tfrecord file at %s', cache_dir)


def _create_federated_gld_dataset(
    cache_dir: str, image_dir: str, train_mapping_file: str,
    test_mapping_file: str) -> Tuple[ClientData, tf.data.Dataset]:
  """Generate fedreated GLDv2 dataset with the downloaded images.

  Args:
    cache_dir: The directory for caching the intermediate results.
    image_dir: The directory that contains the filtered images.
    train_mapping_file: The mapping file for the train set.
    test_mapping_file: The mapping file for the test set.

  Returns:
    A tuple of `(ClientData, tf.data.Dataset)`.
  """

  _create_train_data_files(
      cache_dir=os.path.join(cache_dir, FED_GLD_CACHE, TRAIN_SUB_DIR),
      image_dir=image_dir,
      mapping_file=train_mapping_file)
  _create_test_data_file(
      cache_dir=os.path.join(cache_dir, FED_GLD_CACHE),
      image_dir=image_dir,
      mapping_file=test_mapping_file)
  return vision_datasets_utils.load_data_from_cache(
      cache_dir=os.path.join(cache_dir, FED_GLD_CACHE),
      train_sub_dir=TRAIN_SUB_DIR,
      test_file_name=TEST_FILE_NAME,
      logger_tag=LOGGER)


def _create_mini_gld_dataset(
    cache_dir: str, image_dir: str) -> Tuple[ClientData, tf.data.Dataset]:
  """Generate mini federated GLDv2 dataset with the downloaded images.

  Args:
    cache_dir: The directory for caching the intermediate results.
    image_dir: The directory that contains the filtered images.

  Returns:
    A tuple of `ClientData`, `tf.data.Dataset`.
  """
  train_path = tf.keras.utils.get_file(
      MINI_GLD_TRAIN_SPLIT_FILE,
      origin=MINI_GLD_TRAIN_DOWNLOAD_URL,
      file_hash=MINI_GLD_TRAIN_SPLIT_FILE_MD5_CHECKSUM,
      hash_algorithm='md5',
      cache_dir=cache_dir)
  test_path = tf.keras.utils.get_file(
      MINI_GLD_TEST_SPLIT_FILE,
      origin=MINI_GLD_TEST_DOWNLOAD_URL,
      file_hash=MINI_GLD_TEST_SPLIT_FILE_MD5_CHECKSUM,
      hash_algorithm='md5',
      cache_dir=cache_dir)
  _create_train_data_files(
      cache_dir=os.path.join(cache_dir, MINI_GLD_CACHE, TRAIN_SUB_DIR),
      image_dir=image_dir,
      mapping_file=train_path)
  _create_test_data_file(
      cache_dir=os.path.join(cache_dir, MINI_GLD_CACHE),
      image_dir=image_dir,
      mapping_file=test_path)
  return vision_datasets_utils.load_data_from_cache(
      cache_dir=os.path.join(cache_dir, MINI_GLD_CACHE),
      train_sub_dir=TRAIN_SUB_DIR,
      test_file_name=TEST_FILE_NAME,
      logger_tag=LOGGER)


def _filter_images(shard: int, all_images: Set[str], image_dir: str,
                   base_url: str):
  """Download full GLDv2 dataset, only keep images that are included in the federated gld v2 dataset.

  Args:
    shard: The shard of the GLDv2 dataset.
    all_images: A set which contains all images included in the federated GLD
      dataset.
    image_dir: The directory to keep all filtered images.
    base_url: The base url for downloading GLD v2 dataset images.

  Raises:
    IOError: when failed to download checksum.
  """
  shard_str = '%03d' % shard
  images_tar_url = '%s/train/images_%s.tar' % (base_url, shard_str)
  images_md5_url = '%s/md5sum/train/md5.images_%s.txt' % (base_url, shard_str)
  with tempfile.TemporaryDirectory() as tmp_dir:
    logger = logging.getLogger(LOGGER)
    logger.info('Start to download checksum for shard %s', shard_str)
    md5_path = tf.keras.utils.get_file(
        'images_md5_%s.txt' % shard_str,
        origin=images_md5_url,
        cache_dir=tmp_dir)
    with open(md5_path, 'r') as f:
      md5_hash = f.read()
    if not md5_hash:
      msg = 'Failed to download checksum for shard %s.' % shard_str
      logger.info(msg)
      raise IOError(msg)
    logger.info('Downloaded checksum for shard %s successfully.', shard_str)
    logger.info('Start to download data for shard %s', shard_str)
    tf.keras.utils.get_file(
        'images_%s.tar' % shard_str,
        origin=images_tar_url,
        file_hash=md5_hash,
        hash_algorithm='md5',
        extract=True,
        cache_dir=tmp_dir)
    logger.info('Data for shard %s was downloaded successfully.', shard_str)
    count = 0
    for root, _, files in os.walk(tmp_dir):
      for filename in files:
        name, extension = os.path.splitext(filename)
        if extension == '.jpg' and name in all_images:
          count += 1
          shutil.copyfile(
              os.path.join(root, filename), os.path.join(image_dir, filename))
    logger.info('Moved %d images from shard %s to %s', count, shard_str,
                image_dir)


def _download_data(
    num_worker: int, cache_dir: str, base_url: str
) -> Tuple[ClientData, tf.data.Dataset, ClientData, tf.data.Dataset]:
  """Create a `tff.simulation.datasets.ClientData` for the chosen data split.

  Download the entire GLD v2 dataset, subset the dataset to only include the
  images in the federated GLD v2 dataset, and create both gld23k and gld160k
  datasets.

  Args:
    num_worker: The number of threads for downloading the GLD v2 dataset.
    cache_dir: The directory for caching temporary results.
    base_url: The base url for downloading GLD images.

  Returns:
    A tuple of `tff.simulation.datasets.ClientData`, `tf.data.Dataset`.
  """
  logger = logging.getLogger(LOGGER)
  logger.info('Start to download fed gldv2 mapping files')
  path = tf.keras.utils.get_file(
      '%s.zip' % FED_GLD_SPLIT_FILE_BUNDLE,
      origin=FED_GLD_SPLIT_FILE_DOWNLOAD_URL,
      file_hash=FED_GLD_SPLIT_FILE_BUNDLE_MD5_CHECKSUM,
      hash_algorithm='md5',
      extract=True,
      archive_format='zip',
      cache_dir=cache_dir)
  logger.info('Fed gldv2 mapping files are downloaded successfully.')
  base_path = os.path.dirname(path)
  train_path = os.path.join(base_path, FED_GLD_SPLIT_FILE_BUNDLE,
                            FED_GLD_TRAIN_SPLIT_FILE)
  test_path = os.path.join(base_path, FED_GLD_SPLIT_FILE_BUNDLE,
                           FED_GLD_TEST_SPLIT_FILE)
  train_mapping = vision_datasets_utils.read_csv(train_path)
  test_mapping = vision_datasets_utils.read_csv(test_path)
  all_images = set()
  all_images.update([row['image_id'] for row in train_mapping],
                    [row['image_id'] for row in test_mapping])
  image_dir = os.path.join(cache_dir, 'images')
  if not os.path.exists(image_dir):
    os.mkdir(image_dir)
  logger.info('Start to download GLDv2 dataset.')
  with multiprocessing.pool.ThreadPool(num_worker) as pool:
    train_args = [
        (i, all_images, image_dir, base_url) for i in range(NUM_SHARD_TRAIN)
    ]
    pool.starmap(_filter_images, train_args)

  logger.info('Finish downloading GLDv2 dataset.')
  fed_gld_train, fed_gld_test = _create_federated_gld_dataset(
      cache_dir, image_dir, train_path, test_path)
  mini_gld_train, mini_gld_test = _create_mini_gld_dataset(cache_dir, image_dir)

  return fed_gld_train, fed_gld_test, mini_gld_train, mini_gld_test


def load_data(num_worker: int = 1,
              cache_dir: str = 'cache',
              gld23k: bool = False,
              base_url: str = GLD_SHARD_BASE_URL):
  """Loads a federated version of the Google Landmark v2 dataset.

  The dataset consists of photos of various world landmarks, with images
  grouped by photographer to achieve a federated partitioning of the data.
  The dataset is downloaded and cached locally. If previously downloaded, it
  tries to load the dataset from cache.

  The `tf.data.Datasets` returned by
  `tff.simulation.datasets.ClientData.create_tf_dataset_for_client` will yield
  `collections.OrderedDict` objects at each iteration, with the following keys
  and values:

    -   `'image/decoded'`: A `tf.Tensor` with `dtype=tf.uint8` that
        corresponds to the pixels of the landmark images.
    -   `'class'`: A `tf.Tensor` with `dtype=tf.int64` and shape [1],
        corresponding to the class label of the landmark ([0, 203) for gld23k,
        [0, 2028) for gld160k).

  Two flavors of GLD datasets are available. When gld23k is true, a minimum
  version of the federated Google landmark dataset will be provided for faster
  iterations. The gld23k dataset contains 203 classes, 233 clients and 23080
  images.  When gld23k is false, the gld160k dataset
  (https://arxiv.org/abs/2003.08082) will be provided.  The gld160k dataset
  contains 2,028 classes, 1262 clients and 164,172 images.

  Args:
    num_worker: (Optional) The number of threads for downloading the GLD v2
      dataset.
    cache_dir: (Optional) The directory to cache the downloaded file. If `None`,
      caches in Keras' default cache directory.
    gld23k: (Optional) When true, a smaller version of the federated Google
      Landmark v2 dataset will be loaded. This gld23k dataset is used for faster
      prototyping.
    base_url: (Optional) The base url to download GLD v2 image shards.

  Returns:
    Tuple of (train, test) where the tuple elements are
    a `tff.simulation.datasets.ClientData` and a  `tf.data.Dataset`.
  """
  if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)
  q = multiprocessing.Queue(-1)
  listener = multiprocessing.Process(
      target=_listener_process,
      args=(q, os.path.join(cache_dir, 'load_data.log')))
  listener.start()
  logger = logging.getLogger(LOGGER)
  qh = logging.handlers.QueueHandler(q)
  logger.addHandler(qh)
  logger.info('Start to load data.')
  if gld23k:
    existing_data_cache = os.path.join(cache_dir, MINI_GLD_CACHE)
  else:
    existing_data_cache = os.path.join(cache_dir, FED_GLD_CACHE)
  try:
    logger.info('Try loading dataset from cache')
    return vision_datasets_utils.load_data_from_cache(existing_data_cache,
                                                      TRAIN_SUB_DIR,
                                                      TEST_FILE_NAME, LOGGER)
  except Exception:  # pylint: disable=broad-except
    logger.info('Loading from cache failed, start to download the data.')
    fed_gld_train, fed_gld_test, mini_gld_train, mini_gld_test = _download_data(
        num_worker, cache_dir, base_url)
  finally:
    q.put_nowait(None)
    listener.join()
  if gld23k:
    return mini_gld_train, mini_gld_test
  else:
    return fed_gld_train, fed_gld_test
