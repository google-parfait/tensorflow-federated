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
from tensorflow_federated.python.simulation.baselines.cifar import cifar100_tasks


class CreateResnetModelTest(tf.test.TestCase, parameterized.TestCase):

  @mock.patch('tensorflow_federated.python.simulation.'
              'baselines.cifar.resnet_models.create_resnet18')
  def test_get_resnet18_model(self, mock_model_builder):
    input_shape = (32, 32, 3)
    cifar100_tasks._get_resnet_model(
        model_id='resnet18', input_shape=input_shape)
    mock_model_builder.assert_called_once_with(
        input_shape=input_shape, num_classes=cifar100_tasks._NUM_CLASSES)

  @mock.patch('tensorflow_federated.python.simulation.'
              'baselines.cifar.resnet_models.create_resnet34')
  def test_get_resnet34_model(self, mock_model_builder):
    input_shape = (24, 24, 3)
    cifar100_tasks._get_resnet_model(
        model_id='resnet34', input_shape=input_shape)
    mock_model_builder.assert_called_once_with(
        input_shape=input_shape, num_classes=cifar100_tasks._NUM_CLASSES)

  @mock.patch('tensorflow_federated.python.simulation.'
              'baselines.cifar.resnet_models.create_resnet50')
  def test_get_resnet50_model(self, mock_model_builder):
    input_shape = (24, 1, 3)
    cifar100_tasks._get_resnet_model(
        model_id='resnet50', input_shape=input_shape)
    mock_model_builder.assert_called_once_with(
        input_shape=input_shape, num_classes=cifar100_tasks._NUM_CLASSES)

  @mock.patch('tensorflow_federated.python.simulation.'
              'baselines.cifar.resnet_models.create_resnet101')
  def test_get_resnet101_model(self, mock_model_builder):
    input_shape = (1, 32, 3)
    cifar100_tasks._get_resnet_model(
        model_id='resnet101', input_shape=input_shape)
    mock_model_builder.assert_called_once_with(
        input_shape=input_shape, num_classes=cifar100_tasks._NUM_CLASSES)

  @mock.patch('tensorflow_federated.python.simulation.'
              'baselines.cifar.resnet_models.create_resnet152')
  def test_get_resnet152_model(self, mock_model_builder):
    input_shape = (2, 5, 3)
    cifar100_tasks._get_resnet_model(
        model_id='resnet152', input_shape=input_shape)
    mock_model_builder.assert_called_once_with(
        input_shape=input_shape, num_classes=cifar100_tasks._NUM_CLASSES)


class ImageClassificationTaskTest(tf.test.TestCase, parameterized.TestCase):

  def test_constructs_with_eval_client_spec(self):
    train_client_spec = client_spec.ClientSpec(
        num_epochs=2, batch_size=10, max_elements=3, shuffle_buffer_size=5)
    eval_client_spec = client_spec.ClientSpec(
        num_epochs=1, batch_size=2, max_elements=5, shuffle_buffer_size=10)
    baseline_task_spec = cifar100_tasks.create_image_classification_task(
        train_client_spec,
        eval_client_spec=eval_client_spec,
        model_id='resnet18',
        use_synthetic_data=True)
    self.assertIsInstance(baseline_task_spec, baseline_task.BaselineTask)

  def test_constructs_with_no_eval_client_spec(self):
    train_client_spec = client_spec.ClientSpec(
        num_epochs=2, batch_size=10, max_elements=3, shuffle_buffer_size=5)
    baseline_task_spec = cifar100_tasks.create_image_classification_task(
        train_client_spec, model_id='resnet18', use_synthetic_data=True)
    self.assertIsInstance(baseline_task_spec, baseline_task.BaselineTask)

  @parameterized.named_parameters(
      ('crop1', 32, 32),
      ('crop2', 24, 24),
      ('crop3', 32, 1),
      ('crop4', 1, 32),
  )
  def test_constructs_with_different_crop_sizes(self, crop_height, crop_width):
    train_client_spec = client_spec.ClientSpec(
        num_epochs=2, batch_size=10, max_elements=3, shuffle_buffer_size=5)
    baseline_task_spec = cifar100_tasks.create_image_classification_task(
        train_client_spec,
        model_id='resnet18',
        crop_height=crop_height,
        crop_width=crop_width,
        use_synthetic_data=True)
    self.assertIsInstance(baseline_task_spec, baseline_task.BaselineTask)

  @parameterized.named_parameters(
      ('crop1', -20, 32),
      ('crop2', 50, 24),
      ('crop3', 24, -10),
      ('crop4', 26, 35),
      ('crop5', 33, 33),
  )
  def test_raises_on_bad_crop_sizes(self, crop_height, crop_width):
    train_client_spec = client_spec.ClientSpec(
        num_epochs=2, batch_size=10, max_elements=3, shuffle_buffer_size=5)
    with self.assertRaisesRegex(
        ValueError, 'The crop_height and crop_width '
        'must be between 1 and 32.'):
      cifar100_tasks.create_image_classification_task(
          train_client_spec,
          model_id='resnet18',
          crop_height=crop_height,
          crop_width=crop_width,
          use_synthetic_data=True)

  @parameterized.named_parameters(
      ('resnet18', 'resnet18'),
      ('resnet34', 'resnet34'),
      ('resnet50', 'resnet50'),
      ('resnet101', 'resnet101'),
      ('resnet152', 'resnet152'),
  )
  def test_constructs_with_different_models(self, model_id):
    train_client_spec = client_spec.ClientSpec(
        num_epochs=2, batch_size=10, max_elements=3, shuffle_buffer_size=5)
    baseline_task_spec = cifar100_tasks.create_image_classification_task(
        train_client_spec,
        model_id=model_id,
        crop_height=3,
        crop_width=3,
        use_synthetic_data=True)
    self.assertIsInstance(baseline_task_spec, baseline_task.BaselineTask)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  tf.test.main()
