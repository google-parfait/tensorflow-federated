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

import keras
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from examplespersonalization import p13n_utils

_INPUT_DIM = 2
_OUTPUT_DIM = 1


def _create_dataset():
  """Constructs an unbatched dataset with three datapoints."""
  return tf.data.Dataset.from_tensor_slices({
      'x': np.array([[-1.0, -1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
      'y': np.array([[1.0], [1.0], [1.0]], dtype=np.float32),
  })


def _model_fn():
  """Constructs a linear model with weights initialized to be zeros."""
  inputs = keras.Input(shape=(_INPUT_DIM,))
  outputs = keras.layers.Dense(_OUTPUT_DIM, kernel_initializer='zeros')(inputs)
  keras_model = keras.Model(inputs=inputs, outputs=outputs)
  input_spec = collections.OrderedDict([
      ('x', tf.TensorSpec([None, _INPUT_DIM], dtype=tf.float32)),
      ('y', tf.TensorSpec([None, _OUTPUT_DIM], dtype=tf.float32)),
  ])
  return tff.learning.models.functional_model_from_keras(
      keras_model=keras_model,
      loss_fn=keras.losses.MeanSquaredError(),
      input_spec=input_spec,
      metrics_constructor=collections.OrderedDict(
          loss=keras.metrics.MeanSquaredError,
          mean_absolute_error=keras.metrics.MeanAbsoluteError,
          num_examples=tff.learning.metrics.NumExamplesCounter,
          num_batches=tff.learning.metrics.NumBatchesCounter,
      ),
  )


class P13nUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.model = _model_fn()
    self.dataset = _create_dataset()

  def test_evaluate_fn_succeeds_with_valid_args(self):
    metrics = p13n_utils.evaluate_fn(
        model=self.model,
        model_weights=self.model.initial_weights,
        dataset=self.dataset,
    )

    # Model weights are zeros, so MeanSquaredError and MeanAbsoluteError are 1.
    self.assertDictEqual(
        metrics,
        {
            'num_test_examples': 3,
            'loss': 1.0,
            'mean_absolute_error': 1.0,
            'num_examples': 3,
            'num_batches': 3,
        },
    )

  def test_build_and_run_personalize_fn_succeeds_with_valid_args(self):
    p13n_fn = p13n_utils.build_personalize_fn(
        optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.5),
        batch_size=2,
        num_epochs=1,
        num_epochs_per_eval=1,
        shuffle=False,
    )
    p13n_metrics = p13n_fn(
        model=self.model,
        model_weights=self.model.initial_weights,
        train_data=self.dataset,
        test_data=self.dataset,
    )

    # The model weights become [0, 0, 1] after training one epoch, so the
    # MeanSquaredError and MeanAbsoluteError are 0.
    self.assertDictEqual(
        p13n_metrics,
        {
            'epoch_1': {
                'num_test_examples': 3,
                'loss': 0.0,
                'mean_absolute_error': 0.0,
                'num_examples': 3,
                'num_batches': 3,
            },
            'num_train_examples': 3,  # Same dataset is used for train and test.
        },
    )

  def test_personalize_fn_advances_dropout_seed_across_batches(self):
    feature_dim = 8
    row = np.arange(1, feature_dim + 1, dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices({
        'x': np.stack([row, row, row], axis=0),
        'y': np.zeros([3, 1], dtype=np.float32),
    })

    def create_keras_dropout_model():
      return keras.Sequential([
          keras.Input(shape=(feature_dim,)),
          keras.layers.Dropout(0.5, seed=42),
          keras.layers.Dense(
              1, kernel_initializer='ones', bias_initializer='zeros'
          ),
      ])

    input_spec = collections.OrderedDict([
        ('x', tf.TensorSpec([None, feature_dim], dtype=tf.float32)),
        ('y', tf.TensorSpec([None, 1], dtype=tf.float32)),
    ])
    dropout_model = tff.learning.models.functional_model_from_keras(
        keras_model=create_keras_dropout_model(),
        loss_fn=keras.losses.MeanSquaredError(),
        input_spec=input_spec,
        metrics_constructor=collections.OrderedDict(
            loss=keras.metrics.MeanSquaredError,
        ),
    )

    optimizer = tff.learning.optimizers.build_sgdm(learning_rate=0.05)
    p13n_fn = p13n_utils.build_personalize_fn(
        optimizer_fn=optimizer,
        batch_size=1,
        num_epochs=1,
        num_epochs_per_eval=1,
        shuffle=False,
    )
    p13n_metrics = p13n_fn(
        model=dropout_model,
        model_weights=dropout_model.initial_weights,
        train_data=dataset,
        test_data=dataset,
    )

    # Compare against stateful Keras execution, which advances `SeedGenerator`
    # state across batches in place.
    eager_keras_model = create_keras_dropout_model()
    loss_fn = keras.losses.MeanSquaredError()
    eager_trainable = tuple(
        tf.convert_to_tensor(v) for v in eager_keras_model.trainable_variables
    )
    eager_opt_state = optimizer.initialize(
        tf.nest.map_structure(tf.TensorSpec.from_tensor, eager_trainable)
    )
    for batch in dataset.batch(1):
      for var, val in zip(
          eager_keras_model.trainable_variables, eager_trainable
      ):
        var.assign(val)
      with tf.GradientTape() as tape:
        preds = eager_keras_model(batch['x'], training=True)
        batch_loss = loss_fn(batch['y'], preds)
      grads = tape.gradient(
          batch_loss,
          tuple(v.value for v in eager_keras_model.trainable_variables),
      )
      eager_opt_state, eager_trainable = optimizer.next(
          eager_opt_state, eager_trainable, grads
      )
    expected_eval = p13n_utils.evaluate_fn(
        model=dropout_model,
        model_weights=(
            eager_trainable,
            tuple(
                tf.convert_to_tensor(v)
                for v in eager_keras_model.non_trainable_variables
            ),
        ),
        dataset=dataset,
    )
    self.assertAllClose(p13n_metrics['epoch_1']['loss'], expected_eval['loss'])

    # Verify the result differs from keeping the initial dropout seed frozen
    # across all batches.
    frozen_trainable = tuple(
        tf.convert_to_tensor(w) for w in dropout_model.initial_weights[0]
    )
    frozen_opt_state = optimizer.initialize(
        tf.nest.map_structure(tf.TensorSpec.from_tensor, frozen_trainable)
    )
    for batch in dataset.batch(1):
      with tf.GradientTape() as tape:
        tape.watch(frozen_trainable)
        preds = dropout_model.predict_on_batch(
            model_weights=(frozen_trainable, dropout_model.initial_weights[1]),
            x=batch['x'],
            training=True,
        )
        batch_loss = dropout_model.loss(output=preds, label=batch['y'])
      grads = tape.gradient(batch_loss, frozen_trainable)
      frozen_opt_state, frozen_trainable = optimizer.next(
          frozen_opt_state, frozen_trainable, grads
      )
    frozen_eval = p13n_utils.evaluate_fn(
        model=dropout_model,
        model_weights=(frozen_trainable, dropout_model.initial_weights[1]),
        dataset=dataset,
    )
    self.assertNotAllClose(p13n_metrics['epoch_1']['loss'], frozen_eval['loss'])


if __name__ == '__main__':
  tf.test.main()
