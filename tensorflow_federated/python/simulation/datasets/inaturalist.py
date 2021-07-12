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
"""Libraries for the federated iNaturalist dataset for simulation."""
import collections
import enum
import logging
import os

from typing import Dict
from typing import List
from typing import Tuple

import tensorflow as tf

from tensorflow_federated.python.simulation.datasets import vision_datasets_utils as utils
from tensorflow_federated.python.simulation.datasets.client_data import ClientData

LOGGER = 'iNat2017'
TRAIN_SUB_DIR = 'train'
TEST_FILE_NAME = 'test.tfRecord'
INAT_TRAIN_DOWNLOAD_URL = 'https://storage.googleapis.com/tff-datasets-public/iNaturalist_train.csv'
INAT_TRAIN_SPLIT_FILE = 'iNaturalist_train.csv'
INAT_TRAIN_IMAGE_URL = 'https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val_images.tar.gz'
INAT_TRAIN_IMAGE_MD5_CHECKSUM = '7c784ea5e424efaec655bd392f87301f'
INAT_TEST_DOWNLOAD_URL = 'https://storage.googleapis.com/tff-datasets-public/iNaturalist_test.csv'
INAT_TEST_SPLIT_FILE = 'iNaturalist_test.csv'
INAT_TEST_IMAGE_URL = 'https://ml-inat-competition-datasets.s3.amazonaws.com/2017/test2017.tar.gz'
INAT_TEST_IMAGE_MD5_CHECKSUM = '7d9b096fa1cd94d67a0fa779ea301234'
INAT_TRAIN_SPLIT_FILE_MD5_CHECKSUM = '63d5a41699434bc2b13f5853f3764c18'
INAT_TEST_SPLIT_FILE_MD5_CHECKSUM = 'bcf08a5a740ee76dac4e5af668081891'


class INaturalistSplit(enum.Enum):
  """The different split for the iNaturalist dataset."""
  USER_120K = enum.auto()
  GEO_100 = enum.auto()
  GEO_300 = enum.auto()
  GEO_1K = enum.auto()
  GEO_3K = enum.auto()
  GEO_10K = enum.auto()
  GEO_30K = enum.auto()

  def __repr__(self):
    return '<%s.%s>' % (self.__class__.__name__, self.name)


def _load_data_from_cache(
    cache_dir: str,
    split: INaturalistSplit) -> Tuple[ClientData, tf.data.Dataset]:
  """Load train and test data from the TFRecord files.

  Args:
    cache_dir: The directory containing the TFRecord files.
    split: The split of the federated iNaturalist 2017 dataset.

  Returns:
    A tuple of `ClientData`, `tf.data.Dataset`.
  """
  cache_dir = os.path.join(cache_dir, split.name)
  return utils.load_data_from_cache(cache_dir, TRAIN_SUB_DIR, TEST_FILE_NAME,
                                    LOGGER)


def _generate_image_map(image_dir: str) -> Dict[str, str]:
  """Create an dictionary with key as image id, value as path to the image file.

  Args:
    image_dir: The directory containing all the images.

  Returns:
    The dictionary containing the image id to image file path mapping.
  """
  image_map = {}
  for root, _, files in os.walk(image_dir):
    for f in files:
      if f.endswith('.jpg'):
        image_id = f.rstrip('.jpg')
        image_path = os.path.join(root, f)
        image_map[image_id] = image_path
  return image_map


def _create_dataset_with_mapping(
    image_path_map: Dict[str, str],
    image_class_list: List[Dict[str, str]]) -> List[tf.train.Example]:
  """Builds a dataset based on the mapping file and the images in the image dir.

  Args:
    image_path_map: The directory contains the image id to image path mapping.
    image_class_list: A list of dictionaries. Each dictionary contains
      'image_id' and 'class' keys.

  Returns:
    A list of `tf.train.Example`.
  """
  logger = logging.getLogger(LOGGER)
  examples = []
  for image_class in image_class_list:
    image_id = image_class['image_id']
    if image_id not in image_path_map:
      logger.warning('Image %s is not found.', image_class['image_id'])
      continue
    with open(image_path_map[image_id], 'rb') as f:
      img_bytes = f.read()
      examples.append(
          utils.create_example(img_bytes, int(image_class['class'])))
  return examples


def _create_train_data_files(image_path_map: Dict[str, str], cache_dir: str,
                             split: INaturalistSplit, train_path: str):
  """Create the train data and persist it into a separate file per user.

  Args:
    image_path_map: The dictionary containing the image id to image path
      mapping.
    cache_dir: The directory containing the created datasets.
    split: The split of the federated iNaturalist 2017 dataset.
    train_path: The path to the mapping file for training data.
  """
  logger = logging.getLogger(LOGGER)

  mapping_table = utils.read_csv(train_path)
  user_id_col = split.name.lower()
  expected_cols = [user_id_col, 'image_id', 'class']
  if not all(col in mapping_table[0].keys() for col in expected_cols):
    logger.error('%s has wrong format.', train_path)
    raise ValueError(
        'The mapping file must contain the user_id for the chosen split, image_id and class columns. '
        'The existing columns are %s' % ','.join(mapping_table[0].keys()))
  cache_dir = os.path.join(cache_dir, split.name.lower(), TRAIN_SUB_DIR)
  if not os.path.exists(cache_dir):
    logger.info('Creating cache directory for training data.')
    os.makedirs(cache_dir)
  mapping_per_user = collections.defaultdict(list)
  for row in mapping_table:
    user_id = row[user_id_col]
    if user_id != 'NA':
      mapping_per_user[user_id].append(row)
  for user_id, data in mapping_per_user.items():
    examples = _create_dataset_with_mapping(image_path_map, data)
    with tf.io.TFRecordWriter(os.path.join(cache_dir, str(user_id))) as writer:
      for example in examples:
        writer.write(example.SerializeToString())
      logger.info('Created tfrecord file for user %s with %d examples, at %s',
                  user_id, len(examples), cache_dir)


def _create_test_data_file(image_path_map: Dict[str, str], cache_dir: str,
                           split: INaturalistSplit, mapping_file: str):
  """Create the test data and persist it into a file.

  Args:
    image_path_map: The dictionary containing the image id to image path
      mapping.
    cache_dir: The directory caching the intermediate results.
    split: The split of the federated iNaturalist 2017 dataset.
    mapping_file: The file containing 'image_id' to 'class' mappings.
  """
  logger = logging.getLogger(LOGGER)
  mapping_table = utils.read_csv(mapping_file)
  expected_cols = ['image_id', 'class']
  if not all(col in mapping_table[0].keys() for col in expected_cols):
    logger.error('%s has wrong format.', mapping_file)
    raise ValueError(
        'The mapping file must contain image_id and class columns. The existing'
        ' columns are %s' % ','.join(mapping_table[0].keys()))
  cache_dir = os.path.join(cache_dir, split.name.lower())
  examples = _create_dataset_with_mapping(image_path_map, mapping_table)
  with tf.io.TFRecordWriter(os.path.join(cache_dir, TEST_FILE_NAME)) as writer:
    for example in examples:
      writer.write(example.SerializeToString())
    logger.info('Created test tfrecord file at %s', cache_dir)


def _generate_data_from_image_dir(
    image_dir: str, cache_dir: str,
    split: INaturalistSplit) -> Tuple[ClientData, tf.data.Dataset]:
  """Generate dataset from the images.

  Args:
    image_dir: The directory containing the images.
    cache_dir: The directory keeping the created datasets.
    split: The split of the federated iNaturalist 2017 dataset.

  Returns:
    A tuple of `ClientData`, `tf.data.Dataset`.
  """
  logger = logging.getLogger(LOGGER)
  logger.info('Start to download Fed iNatualist 2017 mapping files')
  train_path = tf.keras.utils.get_file(
      INAT_TRAIN_SPLIT_FILE,
      origin=INAT_TRAIN_DOWNLOAD_URL,
      file_hash=INAT_TRAIN_SPLIT_FILE_MD5_CHECKSUM,
      hash_algorithm='md5',
      cache_dir=cache_dir)
  test_path = tf.keras.utils.get_file(
      INAT_TEST_SPLIT_FILE,
      origin=INAT_TEST_DOWNLOAD_URL,
      file_hash=INAT_TEST_SPLIT_FILE_MD5_CHECKSUM,
      hash_algorithm='md5',
      cache_dir=cache_dir)
  logger.info('Fed iNaturalist 2017 mapping files are downloaded successfully.')
  image_map = _generate_image_map(image_dir)
  _create_train_data_files(image_map, cache_dir, split, train_path)
  _create_test_data_file(image_map, cache_dir, split, test_path)
  return _load_data_from_cache(cache_dir, split)


def load_data(
    image_dir: str = 'images',
    cache_dir: str = 'cache',
    split: INaturalistSplit = INaturalistSplit.USER_120K
) -> Tuple[ClientData, tf.data.Dataset]:
  """Loads a federated version of the iNaturalist 2017 dataset.

  If the dataset is loaded for the first time, the images for the entire
  iNaturalist 2017 dataset will be downloaded from AWS Open Data Program.

  The dataset is created from the images stored inside the image_dir. Once the
  dataset is created, it will be cached inside the cache directory.

  The `tf.data.Datasets` returned by
  `tff.simulation.datasets.ClientData.create_tf_dataset_for_client` will yield
  `collections.OrderedDict` objects at each iteration, with the following keys
  and values:

    -   `'image/decoded'`: A `tf.Tensor` with `dtype=tf.uint8` that
        corresponds to the pixels of the images.
    -   `'class'`: A `tf.Tensor` with `dtype=tf.int64` and shape [1],
        corresponding to the class label.

  Seven splits of iNaturalist datasets are available. The details of each
  different dataset split can be found in https://arxiv.org/abs/2003.08082.
  For the USER_120K dataset, the images are split by the user id.
  The number of clients for USER_120K is 9275. The training set contains 120.300
  images of 1203 species, and test set contains 35641 images.
  For the GEO_* datasets, the images are splitted by the geo location.
  The number of clients for the GEO_* datasets:
    1. GEO_100: 3607.
    2. GEO_300: 1209.
    3. GEO_1K: 369.
    4: GEO_3K: 136.
    5. GEO_10K: 39.
    6. GEO_30K: 12.

  Args:
    image_dir: (Optional) The directory containing the images downloaded from
              https://github.com/visipedia/inat_comp/tree/master/2017
    cache_dir: (Optional) The directory to cache the created datasets.
    split: (Optional) The split of the dataset, default to be split by users.

  Returns:
    Tuple of (train, test) where the tuple elements are
    a `tff.simulation.datasets.ClientData` and a  `tf.data.Dataset`.

  """
  logging.basicConfig(filename='load_data.log', level=logging.INFO)
  logger = logging.getLogger(LOGGER)
  logger.info('Start to load data.')
  if not os.path.exists(cache_dir):
    logger.info('Creating cache directory.')
    os.mkdir(cache_dir)
  try:
    return _load_data_from_cache(cache_dir, split)
  except Exception:  # pylint: disable=broad-except:
    if not image_dir:
      raise ValueError('image_dir cannot be empty or none.')
    if not os.path.isdir(image_dir):
      logger.error('Image directory %s does not exist', image_dir)
      raise ValueError('%s does not exist or is not a directory' % image_dir)
    logger.info('Start to download the images for the training set.')
    tf.keras.utils.get_file(
        'train_val_images.tar.gz',
        origin=INAT_TRAIN_IMAGE_URL,
        file_hash=INAT_TRAIN_IMAGE_MD5_CHECKSUM,
        hash_algorithm='md5',
        extract=True,
        cache_dir=image_dir)
    logger.info('Finish to download the images for the training set.')
    logger.info('Start to download the images for the testing set.')
    tf.keras.utils.get_file(
        'test2017.tar.gz',
        origin=INAT_TEST_IMAGE_URL,
        file_hash=INAT_TEST_IMAGE_MD5_CHECKSUM,
        hash_algorithm='md5',
        extract=True,
        cache_dir=image_dir)
    logger.info('Finish to download the images for the testing set.')
    return _generate_data_from_image_dir(image_dir, cache_dir, split)
