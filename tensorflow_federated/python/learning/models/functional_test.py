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
"""Tests for FunctionModel."""

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning.models import functional


def initial_weights():
  """Returns lists of trainable variables and non-trainable variables."""
  trainable_variables = (np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
                         np.asarray([0.0], dtype=np.float32))
  non_trainable_variables = ()
  return (trainable_variables, non_trainable_variables)


def predict_on_batch(model_weights, x, training):
  """Test predict_on_batch implementing linear regression."""
  trainable = model_weights[0]
  w, b = trainable
  # For the sake of testing, only add the bias term when training so that
  # we get different outputs.
  if training:
    return tf.matmul(x, w, transpose_b=True) + b
  else:
    return tf.matmul(x, w, transpose_b=True)


def forward_pass(model_weights, batch_input, training):
  """Test forward_pass implementing linear regression on MSE."""
  x, y = batch_input
  predictions = predict_on_batch(model_weights, x, training)
  residuals = predictions - y
  num_examples = tf.shape(predictions)[0]
  total_loss = tf.reduce_sum(tf.pow(residuals, 2.))
  average_loss = total_loss / tf.cast(num_examples, tf.float32)
  return model_lib.BatchOutput(
      loss=average_loss, predictions=predictions, num_examples=num_examples)


def create_test_dataset():
  """Create a test dataset."""

  def preprocess(ds):

    def generate_example(i, t):
      del t  # Unused.
      features = tf.random.stateless_uniform(shape=[3], seed=(0, i))
      label = tf.expand_dims(
          tf.reduce_sum(features * tf.constant([1.0, 2.0, 3.0])), axis=-1) + 5.0
      return (features, label)

    return ds.map(generate_example).batch(5, drop_remainder=True)

  num_examples = 25
  return preprocess(tf.data.Dataset.range(num_examples).enumerate())


class FunctionalTest(tf.test.TestCase):

  def test_fail_construction_on_tf_value(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    with self.assertRaisesRegex(TypeError, 'initial_weights may not contain'):
      functional.FunctionalModel((tf.constant(1.0), ()), forward_pass,
                                 predict_on_batch, input_spec)
    with self.assertRaisesRegex(TypeError, 'initial_weights may not contain'):
      functional.FunctionalModel((tf.Variable(1.0), ()), forward_pass,
                                 predict_on_batch, input_spec)

  def test_predict_on_batch(self):
    dataset = create_test_dataset()
    example_batch = next(iter(dataset))
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(initial_weights(),
                                                  forward_pass,
                                                  predict_on_batch, input_spec)
    self.assertAllClose(
        functional_model.predict_on_batch(functional_model.initial_weights,
                                          example_batch[0]), [[0.]] * 5)

  def test_forward_pass(self):
    dataset = create_test_dataset()
    example_batch = next(iter(dataset))
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(initial_weights(),
                                                  forward_pass,
                                                  predict_on_batch, input_spec)
    self.assertAllClose(
        functional_model.predict_on_batch(functional_model.initial_weights,
                                          example_batch[0]), [[0.]] * 5)

  def test_tff_model_from_functional_same_result(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(initial_weights(),
                                                  forward_pass,
                                                  predict_on_batch, input_spec)
    tff_model = functional.model_from_functional(functional_model)

    for training in [True, False]:
      for batch in dataset:
        self.assertAllClose(
            tff_model.predict_on_batch(batch[0], training),
            functional_model.predict_on_batch(functional_model.initial_weights,
                                              batch[0], training))
        tf.nest.map_structure(
            self.assertAllClose, tff_model.forward_pass(batch, training),
            functional_model.forward_pass(functional_model.initial_weights,
                                          batch, training))

  def test_functional_model_converges(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(initial_weights(),
                                                  forward_pass,
                                                  predict_on_batch, input_spec)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)
    variables = tf.nest.map_structure(tf.Variable,
                                      functional_model.initial_weights)
    trainable = variables[0]
    loss = None
    num_epochs = 50
    for batch in dataset.repeat(num_epochs):
      with tf.GradientTape() as tape:
        batch_output = functional_model.forward_pass(
            variables, batch, training=True)
      gradients = tape.gradient(batch_output.loss, trainable)
      optimizer.apply_gradients(zip(gradients, trainable))
      loss = batch_output.loss
    # Expect some amount of convergence after a few epochs of the dataset.
    self.assertLess(loss, 0.1)
    self.assertAllClose(trainable, ([[1.0, 2.0, 3.0]], [5.0]), atol=0.5)

  def test_tff_model_from_functional_converges(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(initial_weights(),
                                                  forward_pass,
                                                  predict_on_batch, input_spec)
    tff_model = functional.model_from_functional(functional_model)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)
    loss = None
    num_epochs = 50
    for batch in dataset.repeat(num_epochs):
      with tf.GradientTape() as tape:
        batch_output = tff_model.forward_pass(batch, training=True)
      gradients = tape.gradient(batch_output.loss,
                                tff_model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, tff_model.trainable_variables))
      loss = batch_output.loss
    # Expect some amount of convergence after a few epochs of the dataset.
    self.assertLess(loss, 0.1)
    self.assertAllClose(
        tff_model.trainable_variables, ([[1.0, 2.0, 3.0]], [5.0]), atol=0.5)


if __name__ == '__main__':
  tf.test.main()
