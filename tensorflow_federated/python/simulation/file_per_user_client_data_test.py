# Copyright 2018, The TensorFlow Federated Authors.
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
"""Tests for tensorflow_federated.python.simulation.file_per_user_client_data.

Demonstrates how to take a columnar dataset (e.g. a CSV) and split it into
per-user files and build a ClientData object that can be used for Federated
Learning simulations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import os.path
import tempfile

from absl.testing import absltest
import numpy as np
import six
import tensorflow as tf

from tensorflow_federated.python.simulation import file_per_user_client_data

# A fake columnar dataset of (user id, value 1, value 2, value 3), roughly
# resembling a CSV file.
#
# See `FilePerUserClientDataTest._setup_fake_per_user_data` for how this split
# up per-user.
FAKE_TEST_DATA = [
    ('ClientA', 3, 4.0, [5., 6.5]),
    ('ClientB', 1, 4.2, [1., 6.1]),
    ('ClientB', 2, 5.3, [5., 6.3]),
    ('ClientA', 5, 4.7, [3., 6.8]),
    ('ClientC', 3, 1.0, [5., 6.4]),
    ('ClientA', 2, 7.5, [7., 6.2]),
    ('ClientA', 3, 4.0, [9., 6.9]),
]


def _create_example(features):
  """Convert a tuple of features to a tf.Example."""
  output_features = collections.OrderedDict()
  for i, feature in enumerate(features):
    if isinstance(feature, int):
      output_features[str(i)] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[feature]))
    elif isinstance(feature, float):
      output_features[str(i)] = tf.train.Feature(
          float_list=tf.train.FloatList(value=[feature]))
    elif isinstance(feature, list):
      output_features[str(i)] = tf.train.Feature(
          float_list=tf.train.FloatList(value=feature))
    else:
      # This is hit if the unittest is updated with unknown types, not an error
      # in the object under test. Extend the unittest capabilities to fix.
      raise NotImplementedError('Cannot handle feature type [%s]' %
                                type(feature))
  return tf.train.Example(features=tf.train.Features(
      feature=output_features)).SerializeToString()


class FakeUserData(object):
  """Container object that creates fake per-user data.

  Using the fake test data, create temporary per-user TFRecord files used for
  the test. Convert each feature-tuple to a `tf.Example` protocol buffer message
  and serialize it to the per-user file.
  """

  def __init__(self, test_data, temp_dir):
    """Construct a FakePerUseData object.

    Args:
      test_data: A list of tuples whose first element is the client ID and all
        subsequent elements are training example features.
      temp_dir: The path to the directory to store temporary per-user files.

    Returns:
      A dict of client IDs to string file paths to TFRecord files.
    """
    writers = {}
    client_file_dict = {}
    for example in test_data:
      client_id, features = example[0], example[1:]
      writer = writers.get(client_id)
      if writer is None:
        fd, path = tempfile.mkstemp(suffix=client_id, dir=temp_dir)
        # close the pre-opened file descriptor immediately to avoid leaking.
        os.close(fd)
        client_file_dict[client_id] = path
        writer = tf.python_io.TFRecordWriter(path=path)
        writers[client_id] = writer
      writer.write(_create_example(features))
    for writer in six.itervalues(writers):
      writer.close()
    self._client_data_file_dict = client_file_dict

  def create_test_dataset_fn(self, client_id):
    client_path = self._client_data_file_dict[client_id]
    features = {
        '0': tf.FixedLenFeature(shape=[], dtype=tf.int64),
        '1': tf.FixedLenFeature(shape=[], dtype=tf.float32),
        '2': tf.FixedLenFeature(shape=[2], dtype=tf.float32),
    }

    def parse_example(e):
      feature_dict = tf.parse_single_example(serialized=e, features=features)
      return tuple(feature_dict[k] for k in sorted(six.iterkeys(feature_dict)))

    return tf.data.TFRecordDataset(client_path).map(parse_example)

  @property
  def client_ids(self):
    return list(self._client_data_file_dict.keys())


class FilePerUserClientDataTest(tf.test.TestCase, absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(FilePerUserClientDataTest, cls).setUpClass()
    cls.temp_dir = tempfile.mkdtemp()
    cls.fake_user_data = FakeUserData(FAKE_TEST_DATA, cls.temp_dir)

  @classmethod
  def tearDownClass(cls):
    for test_data_path in os.listdir(cls.temp_dir):
      os.remove(os.path.join(cls.temp_dir, test_data_path))
    super(FilePerUserClientDataTest, cls).tearDownClass()

  def _create_fake_client_data(self):
    fake_user_data = FilePerUserClientDataTest.fake_user_data
    return file_per_user_client_data.FilePerUserClientData(
        client_ids=fake_user_data.client_ids,
        create_tf_dataset_fn=fake_user_data.create_test_dataset_fn)

  def test_construct_with_non_callable(self):
    with six.assertRaisesRegex(self, TypeError, r'found non-callable'):
      file_per_user_client_data.FilePerUserClientData(
          client_ids=FilePerUserClientDataTest.fake_user_data.client_ids,
          create_tf_dataset_fn=None)

  def test_construct_with_non_list(self):
    with six.assertRaisesRegex(self, TypeError, r'Expected list, found dict'):
      file_per_user_client_data.FilePerUserClientData(
          client_ids={},  # Not a list.
          create_tf_dataset_fn=tf.data.TFRecordDataset)

  def test_client_ids_property(self):
    data = self._create_fake_client_data()
    expected_client_ids = sorted(set(example[0] for example in FAKE_TEST_DATA))
    self.assertEqual(data.client_ids, expected_client_ids)

  def test_output_shapes_property(self):
    expected_shapes = (
        tf.TensorShape([]),
        tf.TensorShape([]),
        tf.TensorShape([2]),
    )
    actual_shapes = self._create_fake_client_data().output_shapes
    self.assertEqual(expected_shapes, actual_shapes)

  def test_output_types_property(self):
    expected_types = (tf.int64, tf.float32, tf.float32)
    actual_types = self._create_fake_client_data().output_types
    self.assertEqual(expected_types, actual_types)

  def test_create_tf_dataset_for_client(self):
    data = self._create_fake_client_data()
    # Iterate over each client, ensuring we received a tf.data.Dataset with the
    # correct data.
    client_id_counters = collections.Counter(
        example[0] for example in FAKE_TEST_DATA)
    for client_id, expected_num_examples in six.iteritems(client_id_counters):
      tf_dataset = data.create_tf_dataset_for_client(client_id)
      self.assertIsInstance(tf_dataset, tf.data.Dataset)

      actual_num_examples = tf_dataset.reduce(np.int32(0), lambda x, _: x + 1)
      self.assertEqual(actual_num_examples.numpy(), expected_num_examples)

      # Assert the actual examples provided are the same.
      expected_examples = [
          example[1:] for example in FAKE_TEST_DATA if example[0] == client_id
      ]
      for actual in tf_dataset:
        expected = expected_examples.pop(0)
        self.assertLen(actual, len(expected))
        for i, e in enumerate(expected):
          if isinstance(e, list):
            self.assertSequenceAlmostEqual(actual[i].numpy(), e, places=4)
          else:
            self.assertAlmostEqual(actual[i].numpy(), e, places=4)
      self.assertEmpty(expected_examples)

  def test_create_tf_dataset_from_all_clients(self):
    data = self._create_fake_client_data()
    expected_examples = [
        example[1:] for example in sorted(FAKE_TEST_DATA, key=lambda x: x[0])
    ]
    tf_dataset = data.create_tf_dataset_from_all_clients()
    # Assert the actual examples provided are the same.
    for actual in tf_dataset:
      expected = expected_examples.pop(0)
      self.assertLen(actual, len(expected))
      for i, e in enumerate(expected):
        if isinstance(e, list):
          self.assertSequenceAlmostEqual(actual[i].numpy(), e, places=4)
        else:
          self.assertAlmostEqual(actual[i].numpy(), e, places=4)
    self.assertEmpty(expected_examples)

  def test_build_client_file_dict(self):
    temp_dir = FilePerUserClientDataTest.temp_dir
    data = file_per_user_client_data.FilePerUserClientData.create_from_dir(
        path=temp_dir, create_tf_dataset_fn=tf.data.TFRecordDataset)
    expected_client_ids = set(example[0] for example in FAKE_TEST_DATA)
    self.assertLen(data.client_ids, len(expected_client_ids))

  def test_build_client_file_dict_default_create_fn(self):
    temp_dir = FilePerUserClientDataTest.temp_dir
    data = file_per_user_client_data.FilePerUserClientData.create_from_dir(
        path=temp_dir)
    expected_client_ids = set(example[0] for example in FAKE_TEST_DATA)
    self.assertLen(data.client_ids, len(expected_client_ids))


if __name__ == '__main__':
  # Need eager_mode to iterate over tf.data.Dataset.
  tf.enable_v2_behavior()
  tf.test.main()
