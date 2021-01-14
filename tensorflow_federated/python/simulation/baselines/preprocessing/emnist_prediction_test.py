# Copyright 2019, Google LLC.
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

import collections
from unittest import mock

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.simulation import client_data
from tensorflow_federated.python.simulation.baselines.preprocessing import emnist_prediction


NUM_ONLY_DIGITS_CLIENTS = 3383
TOTAL_NUM_CLIENTS = 3400


TEST_DATA = collections.OrderedDict(
    pixels=([tf.zeros((28, 28), dtype=tf.float32)]),
    label=([tf.constant(0, dtype=tf.int32)]),
)


def _compute_length_of_dataset(ds):
  return ds.reduce(0, lambda x, _: x + 1)


class PreprocessFnTest(tf.test.TestCase, parameterized.TestCase):

  def test_preprocess_fn_with_negative_epochs_raises(self):
    with self.assertRaisesRegex(ValueError,
                                'num_epochs must be a positive integer'):
      emnist_prediction.create_preprocess_fn(
          num_epochs=-2, batch_size=1, shuffle_buffer_size=1)

  def test_non_supported_task_raises(self):
    with self.assertRaisesRegex(
        ValueError,
        'emnist_task must be one of "digit_recognition" or "autoencoder".'):
      emnist_prediction.create_preprocess_fn(
          num_epochs=1,
          batch_size=1,
          shuffle_buffer_size=1,
          emnist_task='bad_task')

  @parameterized.named_parameters(
      ('paream1', 1, 1),
      ('param2', 4, 2),
      ('param3', 9, 3),
      ('param4', 12, 1),
      ('param5', 5, 3),
      ('param6', 7, 2),
  )
  def test_ds_length_is_ceil_num_epochs_over_batch_size(self, num_epochs,
                                                        batch_size):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = emnist_prediction.create_preprocess_fn(
        num_epochs=num_epochs,
        batch_size=batch_size,
        shuffle_buffer_size=1)
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(
        _compute_length_of_dataset(preprocessed_ds),
        tf.cast(tf.math.ceil(num_epochs / batch_size), tf.int32))

  def test_digit_recognition_preprocess_returns_correct_elements(self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = emnist_prediction.create_preprocess_fn(
        num_epochs=1,
        batch_size=20,
        shuffle_buffer_size=1,
        emnist_task='digit_recognition')
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(preprocessed_ds.element_spec,
                     (tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
                      tf.TensorSpec(shape=(None,), dtype=tf.int32)))

    element = next(iter(preprocessed_ds))
    expected_element = (tf.zeros(shape=(1, 28, 28, 1), dtype=tf.float32),
                        tf.zeros(shape=(1,), dtype=tf.int32))
    self.assertAllClose(self.evaluate(element), expected_element)

  def test_autoencoder_preprocess_returns_correct_elements(self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = emnist_prediction.create_preprocess_fn(
        num_epochs=1,
        batch_size=20,
        shuffle_buffer_size=1,
        emnist_task='autoencoder')
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(preprocessed_ds.element_spec,
                     (tf.TensorSpec(shape=(None, 784), dtype=tf.float32),
                      tf.TensorSpec(shape=(None, 784), dtype=tf.float32)))

    element = next(iter(preprocessed_ds))
    expected_element = (tf.ones(shape=(1, 784), dtype=tf.float32),
                        tf.ones(shape=(1, 784), dtype=tf.float32))
    self.assertAllClose(self.evaluate(element), expected_element)


EMNIST_LOAD_DATA = 'tensorflow_federated.python.simulation.datasets.emnist.load_data'


class FederatedDatasetTest(tf.test.TestCase, parameterized.TestCase):

  @mock.patch(EMNIST_LOAD_DATA)
  def test_preprocess_applied(self, mock_load_data):
    # Mock out the actual data loading from disk. Assert that the preprocessing
    # function is applied to the client data, and that only the ClientData
    # objects we desired are used.
    #
    # The correctness of the preprocessing function is tested in other tests.
    mock_train = mock.create_autospec(client_data.ClientData)
    mock_test = mock.create_autospec(client_data.ClientData)
    mock_load_data.return_value = (mock_train, mock_test)

    _, _ = emnist_prediction.get_federated_datasets()

    mock_load_data.assert_called_once()

    # Assert the training and testing data are preprocessed.
    self.assertEqual(mock_train.mock_calls,
                     mock.call.preprocess(mock.ANY).call_list())
    self.assertEqual(mock_test.mock_calls,
                     mock.call.preprocess(mock.ANY).call_list())

  @parameterized.named_parameters(
      ('task1', 'digit_recognition'),
      ('task2', 'autoencoder'),
  )
  def test_negative_train_client_epochs_raises(self, task):
    with self.assertRaisesRegex(
        ValueError,
        'train_client_epochs_per_round must be a positive integer.'):
      emnist_prediction.get_federated_datasets(
          train_client_epochs_per_round=-10,
          test_client_epochs_per_round=1,
          emnist_task=task)

  @parameterized.named_parameters(
      ('task1', 'digit_recognition'),
      ('task2', 'autoencoder'),
  )
  def test_negative_client_epochs_raises(self, task):
    with self.assertRaisesRegex(
        ValueError,
        'test_client_epochs_per_round must be a positive integer.'):
      emnist_prediction.get_federated_datasets(
          train_client_epochs_per_round=1,
          test_client_epochs_per_round=-10,
          emnist_task=task)


class CentralizedDatasetTest(tf.test.TestCase):

  @mock.patch(EMNIST_LOAD_DATA)
  def test_preprocess_applied(self, mock_load_data):
    # Mock out the actual data loading from disk. Assert that the preprocessing
    # function is applied to the client data, and that only the ClientData
    # objects we desired are used.
    #
    # The correctness of the preprocessing function is tested in other tests.
    sample_ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)

    mock_train = mock.create_autospec(client_data.ClientData)
    mock_train.create_tf_dataset_from_all_clients = mock.Mock(
        return_value=sample_ds)

    mock_test = mock.create_autospec(client_data.ClientData)
    mock_test.create_tf_dataset_from_all_clients = mock.Mock(
        return_value=sample_ds)

    mock_load_data.return_value = (mock_train, mock_test)

    _, _ = emnist_prediction.get_centralized_datasets()

    mock_load_data.assert_called_once()

    # Assert the validation ClientData isn't used, and the train and test
    # are amalgamated into datasets single datasets over all clients.
    self.assertEqual(mock_train.mock_calls,
                     mock.call.create_tf_dataset_from_all_clients().call_list())
    self.assertEqual(mock_test.mock_calls,
                     mock.call.create_tf_dataset_from_all_clients().call_list())


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  tf.test.main()
