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

from unittest import mock

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.simulation.baselines import baseline_task
from tensorflow_federated.python.simulation.baselines import client_spec
from tensorflow_federated.python.simulation.baselines.emnist import char_recognition_tasks


class CreateCharacterRecognitionModelTest(tf.test.TestCase,
                                          parameterized.TestCase):

  @parameterized.named_parameters(
      ('emnist_10', True),
      ('emnist_62', False),
  )
  @mock.patch('tensorflow_federated.python.simulation.'
              'baselines.emnist.emnist_models.create_conv_dropout_model')
  def test_get_character_recognition_model_constructs_cnn_dropout(
      self, only_digits, mock_model_builder):
    char_recognition_tasks._get_character_recognition_model(
        model_id='cnn_dropout', only_digits=only_digits)
    mock_model_builder.assert_called_once_with(only_digits=only_digits)

  @parameterized.named_parameters(
      ('emnist_10', True),
      ('emnist_62', False),
  )
  @mock.patch('tensorflow_federated.python.simulation.'
              'baselines.emnist.emnist_models.create_original_fedavg_cnn_model')
  def test_get_character_recognition_model_constructs_cnn(
      self, only_digits, mock_model_builder):
    char_recognition_tasks._get_character_recognition_model(
        model_id='cnn', only_digits=only_digits)
    mock_model_builder.assert_called_once_with(only_digits=only_digits)

  @parameterized.named_parameters(
      ('emnist_10', True),
      ('emnist_62', False),
  )
  @mock.patch('tensorflow_federated.python.simulation.'
              'baselines.emnist.emnist_models.create_two_hidden_layer_model')
  def test_get_character_recognition_model_constructs_2nn(
      self, only_digits, mock_model_builder):
    char_recognition_tasks._get_character_recognition_model(
        model_id='2nn', only_digits=only_digits)
    mock_model_builder.assert_called_once_with(only_digits=only_digits)

  @parameterized.named_parameters(
      ('emnist_10', True),
      ('emnist_62', False),
  )
  def test_raises_on_unsupported_model(self, only_digits):
    with self.assertRaises(ValueError):
      char_recognition_tasks._get_character_recognition_model(
          model_id='unsupported_model', only_digits=only_digits)


class CreateCharacterRecognitionTaskTest(tf.test.TestCase,
                                         parameterized.TestCase):

  @parameterized.named_parameters(
      ('emnist_10_cnn', True, 'cnn'),
      ('emnist_62_cnn', False, 'cnn'),
      ('emnist_10_cnn_dropout', True, 'cnn_dropout'),
      ('emnist_62_cnn_dropout', False, 'cnn_dropout'),
      ('emnist_10_2nn', True, '2nn'),
      ('emnist_62_2nn', False, '2nn'),
  )
  def test_constructs_with_eval_client_spec(self, only_digits, model_id):
    train_client_spec = client_spec.ClientSpec(
        num_epochs=2, batch_size=10, max_elements=3, shuffle_buffer_size=5)
    eval_client_spec = client_spec.ClientSpec(
        num_epochs=1, batch_size=2, max_elements=5, shuffle_buffer_size=10)
    baseline_task_spec = char_recognition_tasks.create_character_recognition_task(
        train_client_spec,
        eval_client_spec=eval_client_spec,
        model_id=model_id,
        only_digits=only_digits,
        use_synthetic_data=True)
    self.assertIsInstance(baseline_task_spec, baseline_task.BaselineTask)

  @parameterized.named_parameters(
      ('emnist_10_cnn', True, 'cnn'),
      ('emnist_62_cnn', False, 'cnn'),
      ('emnist_10_cnn_dropout', True, 'cnn_dropout'),
      ('emnist_62_cnn_dropout', False, 'cnn_dropout'),
      ('emnist_10_2nn', True, '2nn'),
      ('emnist_62_2nn', False, '2nn'),
  )
  def test_constructs_with_no_eval_client_spec(self, only_digits, model_id):
    train_client_spec = client_spec.ClientSpec(
        num_epochs=2, batch_size=10, max_elements=3, shuffle_buffer_size=5)
    baseline_task_spec = char_recognition_tasks.create_character_recognition_task(
        train_client_spec,
        model_id=model_id,
        only_digits=only_digits,
        use_synthetic_data=True)
    self.assertIsInstance(baseline_task_spec, baseline_task.BaselineTask)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  tf.test.main()
