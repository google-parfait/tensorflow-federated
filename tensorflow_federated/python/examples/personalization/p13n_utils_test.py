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

from tensorflow_federated.python.examples.personalization import p13n_utils


def _create_dataset():
  """Constructs an unbatched dataset with three datapoints."""
  x = np.array([[-1.0, -1.0], [1.0, 1.0], [1.0, 1.0]])
  y = np.array([[1.0], [1.0], [1.0]])
  ds = collections.OrderedDict(x=x.astype(np.float32), y=y.astype(np.float32))
  return tf.data.Dataset.from_tensor_slices(ds)


def _model_fn():
  """Constructs a linear model with weights initialized to be zeros."""
  inputs = tf.keras.Input(shape=(2,))  # feature dim = 2
  outputs = tf.keras.layers.Dense(1, kernel_initializer='zeros')(inputs)
  keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
  input_spec = collections.OrderedDict([
      ('x', tf.TensorSpec([None, 2], dtype=tf.float32)),
      ('y', tf.TensorSpec([None, 1], dtype=tf.float32))
  ])
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.MeanAbsoluteError()])


class P13nUtilsTest(tf.test.TestCase):

  def test_evaluate_fn_succeeds_with_valid_args(self):
    model = _model_fn()
    dataset = _create_dataset()
    metrics = p13n_utils.evaluate_fn(model=model, dataset=dataset, batch_size=1)

    # Since the weights are all zeros, both MSE and MAE equal 1.0.
    self.assertDictContainsSubset({
        'loss': 1.0,
        'mean_absolute_error': 1.0
    }, metrics)

  def test_build_personalize_fn_succeeds_with_valid_args(self):
    model = _model_fn()
    dataset = _create_dataset()
    p13n_fn = p13n_utils.build_personalize_fn(
        optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.5),
        train_batch_size=2,
        max_num_epochs=1,
        num_epochs_per_eval=1,
        test_batch_size=1,
        shuffle=False)
    p13n_metrics = p13n_fn(model=model, train_data=dataset, test_data=dataset)

    # The model weights become [0, 0, 1] after training one epoch, which gives
    # an MSE 0.0 and MAE 0.0.
    self.assertDictContainsSubset({
        'loss': 0.0,
        'mean_absolute_error': 0.0
    }, p13n_metrics['epoch_1'])
    # The model is trained for one epoch, so the final model has the same
    # metrics as those in `epoch_1`.
    self.assertDictContainsSubset({
        'loss': 0.0,
        'mean_absolute_error': 0.0
    }, p13n_metrics['final_model'])
    # The total number of training examples is 3.
    self.assertEqual(p13n_metrics['num_examples'], 3)
    # The batch size is set to 2 in `p13n_fn`, so training data has 2 batches.
    self.assertEqual(p13n_metrics['num_batches'], 2)


if __name__ == '__main__':
  tf.test.main()
