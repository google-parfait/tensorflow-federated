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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.simulation.baselines import baseline_task
from tensorflow_federated.python.simulation.baselines import client_spec
from tensorflow_federated.python.simulation.baselines.stackoverflow import tag_prediction_tasks


class LogisticRegressionModelTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('input2_output5', 2, 5),
      ('input1_output10', 1, 10),
      ('input7_output3', 7, 3),
      ('input100_output1', 100, 1),
  )
  def test_constructs_keras_model_with_correct_dimensions(
      self, input_size, output_size):
    model = tag_prediction_tasks._build_logistic_regression_model(
        input_size=input_size, output_size=output_size)
    self.assertIsInstance(model, tf.keras.Model)

    model_weights = model.weights
    self.assertLen(model_weights, 2)
    self.assertEqual(model_weights[0].shape, (input_size, output_size))
    self.assertEqual(model_weights[1].shape, (output_size,))


class TagPredictionTasksTest(tf.test.TestCase, parameterized.TestCase):

  def test_constructs_with_eval_client_spec(self):
    train_client_spec = client_spec.ClientSpec(
        num_epochs=2, batch_size=10, max_elements=3, shuffle_buffer_size=5)
    eval_client_spec = client_spec.ClientSpec(
        num_epochs=1, batch_size=2, max_elements=5, shuffle_buffer_size=10)
    baseline_task_spec = tag_prediction_tasks.create_tag_prediction_task(
        train_client_spec,
        eval_client_spec=eval_client_spec,
        use_synthetic_data=True)
    self.assertIsInstance(baseline_task_spec, baseline_task.BaselineTask)

  def test_constructs_without_eval_client_spec(self):
    train_client_spec = client_spec.ClientSpec(
        num_epochs=2, batch_size=10, max_elements=3, shuffle_buffer_size=5)
    baseline_task_spec = tag_prediction_tasks.create_tag_prediction_task(
        train_client_spec, use_synthetic_data=True)
    self.assertIsInstance(baseline_task_spec, baseline_task.BaselineTask)

  @parameterized.named_parameters(
      ('word_vocab_size1', 1),
      ('word_vocab_size8', 8),
      ('word_vocab_size10', 10),
      ('word_vocab_size50', 50),
  )
  def test_constructs_with_different_word_vocab_sizes(self, word_vocab_size):
    train_client_spec = client_spec.ClientSpec(
        num_epochs=2, batch_size=10, max_elements=3, shuffle_buffer_size=5)
    baseline_task_spec = tag_prediction_tasks.create_tag_prediction_task(
        train_client_spec,
        word_vocab_size=word_vocab_size,
        use_synthetic_data=True)
    self.assertIsInstance(baseline_task_spec, baseline_task.BaselineTask)

  @parameterized.named_parameters(
      ('word_vocab_size0', 0),
      ('word_vocab_size_minus1', -1),
      ('word_vocab_size_minus5', -5),
  )
  def test_raises_on_bad_word_vocab_size(self, word_vocab_size):
    train_client_spec = client_spec.ClientSpec(
        num_epochs=2, batch_size=10, max_elements=3, shuffle_buffer_size=5)
    with self.assertRaises(ValueError):
      tag_prediction_tasks.create_tag_prediction_task(
          train_client_spec,
          word_vocab_size=word_vocab_size,
          use_synthetic_data=True)

#   @parameterized.named_parameters(
#       ('tag_vocab_size1', 1),
#       ('tag_vocab_size8', 8),
#       ('tag_vocab_size10', 10),
#       ('tag_vocab_size50', 50),
#   )
#   def test_constructs_with_different_tag_vocab_sizes(self, tag_vocab_size):
#     train_client_spec = client_spec.ClientSpec(
#         num_epochs=2, batch_size=10, max_elements=3, shuffle_buffer_size=5)
#     baseline_task_spec = tag_prediction_tasks.create_tag_prediction_task(
#         train_client_spec,
#         tag_vocab_size=tag_vocab_size,
#         use_synthetic_data=True)
#     self.assertIsInstance(baseline_task_spec, baseline_task.BaselineTask)

#   @parameterized.named_parameters(
#       ('tag_vocab_size0', 0),
#       ('tag_vocab_size_minus1', -1),
#       ('tag_vocab_size_minus5', -5),
#   )
#   def test_raises_on_bad_tag_vocab_size(self, tag_vocab_size):
#     train_client_spec = client_spec.ClientSpec(
#         num_epochs=2, batch_size=10, max_elements=3, shuffle_buffer_size=5)
#     with self.assertRaises(ValueError):
#       tag_prediction_tasks.create_tag_prediction_task(
#           train_client_spec,
#           tag_vocab_size=tag_vocab_size,
#           use_synthetic_data=True)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  tf.test.main()
