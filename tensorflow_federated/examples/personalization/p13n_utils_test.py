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

import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.examples.personalization import p13n_utils

_INPUT_DIM = 2
_OUTPUT_DIM = 1


def _create_dataset():
  """Constructs an unbatched dataset with three datapoints."""
  return tf.data.Dataset.from_tensor_slices({
      'x': np.array([[-1.0, -1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
      'y': np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
  })


def _model_fn():
  """Constructs a linear model with weights initialized to be zeros."""
  inputs = tf.keras.Input(shape=(_INPUT_DIM,))
  outputs = tf.keras.layers.Dense(
      _OUTPUT_DIM, kernel_initializer='zeros')(
          inputs)
  keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
  input_spec = collections.OrderedDict([
      ('x', tf.TensorSpec([None, _INPUT_DIM], dtype=tf.float32)),
      ('y', tf.TensorSpec([None, _OUTPUT_DIM], dtype=tf.float32))
  ])
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.MeanAbsoluteError()])


class P13nUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.model = _model_fn()
    self.dataset = _create_dataset()

  def test_evaluate_fn_succeeds_with_valid_args(self):
    metrics = p13n_utils.evaluate_fn(model=self.model, dataset=self.dataset)

    # Model weights are zeros, so MeanSquaredError and MeanAbsoluteError are 1.
    self.assertDictEqual(metrics, {
        'num_test_examples': 3,
        'mean_absolute_error': 1.0,
        'loss': 1.0
    })

  def test_build_and_run_personalize_fn_succeeds_with_valid_args(self):
    p13n_fn = p13n_utils.build_personalize_fn(
        optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.5),
        batch_size=2,
        num_epochs=1,
        num_epochs_per_eval=1,
        shuffle=False)
    p13n_metrics = p13n_fn(
        model=self.model, train_data=self.dataset, test_data=self.dataset)

    # The model weights become [0, 0, 1] after training one epoch, so the
    # MeanSquaredError and MeanAbsoluteError are 0.
    self.assertDictEqual(
        p13n_metrics,
        {
            'epoch_1': {
                'num_test_examples': 3,
                'mean_absolute_error': 0.0,
                'loss': 0.0
            },
            'num_train_examples': 3  # Same dataset is used for train and test.
        })


if __name__ == '__main__':
  tf.test.main()
