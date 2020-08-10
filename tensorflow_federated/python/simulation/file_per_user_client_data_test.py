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

Demonstrates how to take a column dataset (e.g. a CSV) and split it into
per-user files and build a ClientData object that can be used for Federated
Learning simulations.
"""

import collections
import os
import os.path
import tempfile

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.backends.native import execution_contexts
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
      raise NotImplementedError('Cannot handle feature type [{}]'.format(
          type(feature)))
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
        writer = tf.io.TFRecordWriter(path=path)
        writers[client_id] = writer
      writer.write(_create_example(features))
    for writer in writers.values():
      writer.close()
    self._client_data_file_dict = client_file_dict

  def create_test_dataset_fn(self, client_path):
    features = {
        '0': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        '1': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
        '2': tf.io.FixedLenFeature(shape=[2], dtype=tf.float32),
    }

    def parse_example(e):
      feature_dict = tf.io.parse_single_example(serialized=e, features=features)
      return tuple(feature_dict[k] for k in sorted(feature_dict.keys()))

    return tf.data.TFRecordDataset(client_path).map(parse_example)

  @property
  def client_data_file_dict(self):
    return self._client_data_file_dict


class FilePerUserClientDataTest(tf.test.TestCase, absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.temp_dir = tempfile.mkdtemp()
    cls.fake_user_data = FakeUserData(FAKE_TEST_DATA, cls.temp_dir)

  @classmethod
  def tearDownClass(cls):
    for test_data_path in os.listdir(cls.temp_dir):
      os.remove(os.path.join(cls.temp_dir, test_data_path))
    super().tearDownClass()

  def _create_fake_client_data(self):
    fake_user_data = FilePerUserClientDataTest.fake_user_data
    return file_per_user_client_data.FilePerUserClientData(
        fake_user_data.client_data_file_dict,
        fake_user_data.create_test_dataset_fn,
    )

  def test_construct_with_non_callable(self):
    fake_user_data = FilePerUserClientDataTest.fake_user_data
    with self.assertRaisesRegex(TypeError, r'found non-callable'):
      file_per_user_client_data.FilePerUserClientData(
          client_ids_to_files=fake_user_data.client_data_file_dict,
          dataset_fn=None,
      )

  def test_construct_with_non_dict(self):
    with self.assertRaisesRegex(TypeError, r'Expected collections.abc.Mapping'):
      file_per_user_client_data.FilePerUserClientData(
          client_ids_to_files=[],  # Not a dict.
          dataset_fn=tf.data.TFRecordDataset,
      )

  def test_client_ids_property(self):
    data = self._create_fake_client_data()
    expected_client_ids = sorted(set(example[0] for example in FAKE_TEST_DATA))
    self.assertEqual(data.client_ids, expected_client_ids)

  def test_element_type_structure(self):
    expected_structure = (tf.TensorSpec(shape=[], dtype=tf.int64),
                          tf.TensorSpec(shape=[], dtype=tf.float32),
                          tf.TensorSpec(shape=[2], dtype=tf.float32))
    actual_structure = self._create_fake_client_data().element_type_structure
    self.assertEqual(expected_structure, actual_structure)

  def test_create_tf_dataset_for_client(self):
    data = self._create_fake_client_data()
    # Iterate over each client, ensuring we received a tf.data.Dataset with the
    # correct data.
    client_id_counters = collections.Counter(
        example[0] for example in FAKE_TEST_DATA)
    for client_id, expected_num_examples in client_id_counters.items():
      tf_dataset = data.create_tf_dataset_for_client(client_id)
      self.assertIsInstance(tf_dataset, tf.data.Dataset)

      actual_num_examples = tf_dataset.reduce(np.int32(0), lambda x, _: x + 1)
      self.assertEqual(
          self.evaluate(actual_num_examples), expected_num_examples)

      # Assert the actual examples provided are the same.
      expected_examples = [
          example[1:] for example in FAKE_TEST_DATA if example[0] == client_id
      ]
      for actual in tf_dataset:
        expected = expected_examples.pop(0)
        actual = self.evaluate(actual)
        self.assertLen(actual, len(expected))
        for i, e in enumerate(expected):
          if isinstance(e, list):
            self.assertSequenceAlmostEqual(actual[i], e, places=4)
          else:
            self.assertAlmostEqual(actual[i], e, places=4)
      self.assertEmpty(expected_examples)

  def test_dataset_computation(self):
    data = self._create_fake_client_data()
    self.assertIsInstance(data.dataset_computation,
                          computation_base.Computation)
    # Iterate over each client, ensuring we received a tf.data.Dataset with the
    # correct data.
    client_id_counters = collections.Counter(
        example[0] for example in FAKE_TEST_DATA)
    for client_id, expected_num_examples in client_id_counters.items():
      tf_dataset = data.dataset_computation(client_id)
      self.assertIsInstance(tf_dataset, tf.data.Dataset)

      actual_num_examples = tf_dataset.reduce(np.int32(0), lambda x, _: x + 1)
      self.assertEqual(
          self.evaluate(actual_num_examples), expected_num_examples)

      # Assert the actual examples provided are the same.
      expected_examples = [
          example[1:] for example in FAKE_TEST_DATA if example[0] == client_id
      ]
      for actual in tf_dataset:
        expected = expected_examples.pop(0)
        actual = self.evaluate(actual)
        self.assertLen(actual, len(expected))
        for i, e in enumerate(expected):
          if isinstance(e, list):
            self.assertSequenceAlmostEqual(actual[i], e, places=4)
          else:
            self.assertAlmostEqual(actual[i], e, places=4)
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
  execution_contexts.set_local_execution_context()
  tf.test.main()
