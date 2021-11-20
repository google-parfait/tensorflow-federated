# Copyright 2021, The TensorFlow Federated Authors.  #
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

import collections

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.tensorflow_libs import variable_utils


def initial_weights():
  """Returns lists of trainable variables and non-trainable variables."""
  trainable_variables = (np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
                         np.asarray([0.0], dtype=np.float32))
  non_trainable_variables = ()
  return (trainable_variables, non_trainable_variables)


@tf.function
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


@tf.function
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


def create_test_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(1,)),
      tf.keras.layers.Dense(
          1, kernel_initializer='zeros', bias_initializer='zeros')
  ])


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
    with self.assertRaisesRegex(functional.ValueMustNotBeTFError,
                                'initial_weights may not contain'):
      functional.FunctionalModel((tf.constant(1.0), ()), forward_pass,
                                 predict_on_batch, input_spec)
    with self.assertRaisesRegex(functional.ValueMustNotBeTFError,
                                'initial_weights may not contain'):
      functional.FunctionalModel((tf.Variable(1.0), ()), forward_pass,
                                 predict_on_batch, input_spec)

  def test_fail_non_tf_function(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    with self.assertRaisesRegex(
        functional.CallableMustBeTFFunctionError,
        'forward_pass_fn does not have a `get_concrete_function`'):
      functional.FunctionalModel((), forward_pass.python_function,
                                 predict_on_batch, input_spec)
    with self.assertRaisesRegex(
        functional.CallableMustBeTFFunctionError,
        'predict_on_batch_fn does not have a `get_concrete_function`'):
      functional.FunctionalModel((), forward_pass,
                                 predict_on_batch.python_function, input_spec)

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

  def test_predict_on_batch_keras_outside_graph_fails(self):
    dataset = create_test_dataset()
    example_batch = next(iter(dataset))
    functional_model = functional.functional_model_from_keras(
        keras_model=create_test_keras_model(),
        loss_fn=tf.keras.losses.MeanSquaredError(),
        input_spec=(tf.TensorSpec([None, 1], dtype=tf.float32),
                    tf.TensorSpec([None, 1], dtype=tf.int32)))
    with self.assertRaisesRegex(functional.KerasFunctionalModelError,
                                'only usable inside a tff.tf_computation'):
      functional_model.predict_on_batch(functional_model.initial_weights,
                                        example_batch[0])

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

  def test_construct_from_keras(self):
    keras_model = create_test_keras_model()
    # Assign some variables after initialization so we can assert that they
    # were cloned into the FunctionalModel.
    tf.nest.map_structure(lambda v: v.assign(tf.ones_like(v)),
                          keras_model.variables)
    functional_model = functional.functional_model_from_keras(
        keras_model=keras_model,
        loss_fn=tf.keras.losses.MeanSquaredError(),
        input_spec=(tf.TensorSpec([None, 1], dtype=tf.float32),
                    tf.TensorSpec([None, 1], dtype=tf.int32)))
    self.assertIsInstance(functional_model, functional.FunctionalModel)
    # Assert all ones, instead of zeros from a newly initial model.
    tf.nest.map_structure(lambda v: self.assertAllClose(v, tf.ones_like(v)),
                          functional_model.initial_weights)

  def test_construct_from_keras_converges(self):
    functional_model = functional.functional_model_from_keras(
        keras_model=create_test_keras_model(),
        loss_fn=tf.keras.losses.MeanSquaredError(),
        input_spec=(tf.TensorSpec([None, 1], dtype=tf.float32),
                    tf.TensorSpec([None, 1], dtype=tf.int32)))
    with tf.Graph().as_default() as test_graph:
      # Capture all the variables for later initialization in the session,
      # otherwise it's hard to get our hands on the Keras-owned variables.
      with variable_utils.record_variable_creation_scope(
      ) as captured_variables:
        # Create data satisfying y = 2*x + 1
        dataset = tf.data.Dataset.from_tensor_slices((
            # Features
            [[1.0], [2.0], [3.0]],
            # Labels.
            [[3.0], [5.0], [7.0]],
        )).batch(1)
        variables = tf.nest.map_structure(tf.Variable,
                                          functional_model.initial_weights)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

        @tf.function
        def train():
          weights = tf.nest.map_structure(lambda v: v.read_value(), variables)
          initial_loss = loss = functional_model.forward_pass(
              weights, next(iter(dataset)), training=True).loss
          trainable = variables[0]
          for batch in dataset.repeat(30):
            with tf.GradientTape() as tape:
              weights = tf.nest.map_structure(lambda v: v.read_value(),
                                              variables)
              tape.watch(weights[0])
              batch_output = functional_model.forward_pass(
                  weights, batch, training=True)
            gradients = tape.gradient(batch_output.loss, weights[0])
            optimizer.apply_gradients(zip(gradients, trainable))
            loss = batch_output.loss
          return initial_loss, loss

        initial_loss, final_loss = train()
    with tf.compat.v1.Session(graph=test_graph) as sess:
      sess.run(tf.compat.v1.initializers.variables(captured_variables))
      initial_loss, final_loss = sess.run([initial_loss, final_loss])
    # Expect some amount of convergence after a few epochs of the dataset.
    self.assertGreater(initial_loss, 2.0)
    self.assertLess(final_loss, 0.2)

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
    self.assertAllClose(tff_model.report_local_outputs(),
                        collections.OrderedDict(loss=[1066.19628, 1250.0]))
    self.assertAllClose(tff_model.report_local_unfinalized_metrics(),
                        collections.OrderedDict(loss=[1066.19628, 1250.0]))

  def test_tff_model_from_functional_fails_with_repeated_metric_names(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(initial_weights(),
                                                  forward_pass,
                                                  predict_on_batch, input_spec)
    metric_constructors = [
        lambda: tf.keras.metrics.MeanSquaredError(name='same_name'),
        lambda: tf.keras.metrics.RootMeanSquaredError(name='same_name')
    ]
    with self.assertRaisesRegex(ValueError,
                                'each metric should have a unique name'):
      functional.model_from_functional(functional_model, metric_constructors)

  def test_tff_model_from_functional_binding_metrics_succeeds(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(initial_weights(),
                                                  forward_pass,
                                                  predict_on_batch, input_spec)
    metric_constructors = [
        tf.keras.metrics.MeanSquaredError, tf.keras.metrics.RootMeanSquaredError
    ]
    tff_model = functional.model_from_functional(functional_model,
                                                 metric_constructors)
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
    local_outputs = tff_model.report_local_outputs()
    self.assertAllClose(
        local_outputs,
        collections.OrderedDict(
            # The model uses mean squred error as `loss`, so the other two
            # metrics (`mean_squared_error` and `root_mean_squared_error`)
            # should have the same state as `loss`.
            loss=[1066.19628, 1250.0],
            mean_squared_error=[1066.19628, 1250.0],
            root_mean_squared_error=[1066.19628, 1250.0]))
    self.assertAllClose(
        tff_model.report_local_unfinalized_metrics(),
        collections.OrderedDict(
            # The model uses mean squred error as `loss`, so the other two
            # metrics (`mean_squared_error` and `root_mean_squared_error`)
            # should have the same state as `loss`.
            loss=[1066.19628, 1250.0],
            mean_squared_error=[1066.19628, 1250.0],
            root_mean_squared_error=[1066.19628, 1250.0]))

  def test_tff_model_from_functional_federated_aggregate_metrics_succeeds(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(initial_weights(),
                                                  forward_pass,
                                                  predict_on_batch, input_spec)
    metric_constructors = [
        lambda: tf.keras.metrics.MeanSquaredError(name='mse'),
        lambda: tf.keras.metrics.MeanAbsoluteError(name='mae')
    ]
    tff_model = functional.model_from_functional(functional_model,
                                                 metric_constructors)
    client_1_local_outputs = collections.OrderedDict(
        loss=[1.0, 2.0], mse=[1.0, 2.0], mae=[1.0, 2.0])
    client_2_local_outputs = collections.OrderedDict(
        loss=[2.0, 4.0], mse=[2.0, 2.0], mae=[1.0, 6.0])
    aggregated_metrics = tff_model.federated_output_computation(
        [client_1_local_outputs, client_2_local_outputs])
    self.assertAllClose(
        aggregated_metrics,
        # loss = (1.0+2.0)/(2.0+4.0) = 0.5
        # mse = (1.0+2.0)/(2.0+2.0) = 0.75
        # mae = (1.0+1.0)/(2.0+6.0) = 0.25
        collections.OrderedDict(loss=0.5, mse=0.75, mae=0.25))

  def test_keras_model_with_non_trainable_variables_fails(self):
    inputs = tf.keras.layers.Input(shape=[1])
    d = tf.keras.layers.Dense(1)
    d.trainable = False
    outputs = d(inputs)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    with self.assertRaisesRegex(functional.KerasFunctionalModelError,
                                'non-trainable variables'):
      functional.functional_model_from_keras(
          keras_model,
          tf.keras.losses.MeanSquaredError(),
          input_spec=(tf.TensorSpec(shape=[None, 1]),
                      tf.TensorSpec(shape=[None, 1])))

  def test_keras_model_with_batch_normalization_fails(self):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=[10]),
        tf.keras.layers.BatchNormalization(),
    ])
    with self.assertRaisesRegex(functional.KerasFunctionalModelError,
                                'batch normalization'):
      functional.functional_model_from_keras(
          model,
          tf.keras.losses.MeanSquaredError(),
          input_spec=(tf.TensorSpec(shape=[None, 10]),
                      tf.TensorSpec(shape=[None, 1])))

  def test_keras_model_with_shared_variables_fails(self):

    class SharedLayer(tf.keras.layers.Layer):

      def __init__(self, dense_layer: tf.keras.layers.Dense, **kwargs):
        super().__init__()
        self._dense_layer = dense_layer
        self.kernel = dense_layer.kernel
        self.bias = dense_layer.bias

      def call(self, inputs):
        return inputs

      def get_config(self):
        config = super().get_config()
        config['dense_layer'] = self._dense_layer
        return config

    inputs = tf.keras.layers.Input(shape=[1])
    layer1 = tf.keras.layers.Dense(1)
    y = layer1(inputs)
    layer2 = SharedLayer(layer1)
    outputs = layer2(y)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    with self.assertRaisesRegex(functional.KerasFunctionalModelError,
                                'sharing variables across layers'):
      functional.functional_model_from_keras(
          keras_model,
          tf.keras.losses.MeanSquaredError(),
          input_spec=(tf.TensorSpec(shape=[None, 1]),
                      tf.TensorSpec(shape=[None, 1])))


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  tf.test.main()
