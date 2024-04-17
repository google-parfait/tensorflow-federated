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
import itertools
from typing import Any, Optional

import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.learning.metrics import aggregator
from tensorflow_federated.python.learning.metrics import types
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.models import variable
from tensorflow_federated.python.tensorflow_libs import variable_utils


def initial_weights():
  """Returns lists of trainable variables and non-trainable variables."""
  trainable_variables = (
      np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
      np.asarray([0.0], dtype=np.float32),
  )
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


def loss(output, label, sample_weight=None) -> float:
  del sample_weight
  return tf.math.reduce_mean(tf.math.pow(output - label, 2.0))


@tf.function
def predict_on_batch_with_structure(model_weights, x, training):
  """Test predict_on_batch returning a structure."""
  trainable = model_weights[0]
  w, b = trainable
  # Return a tuple whose first element is our prediction.
  # For the sake of testing, only add the bias term when training so that
  # we get different outputs.
  if training:
    return (tf.matmul(x, w, transpose_b=True) + b, x)
  else:
    return (tf.matmul(x, w, transpose_b=True), x)


def loss_with_structure(output, label, sample_weight=None) -> float:
  del sample_weight
  predictions = output[0]  # Since we return a tuple above.
  return tf.math.reduce_mean(tf.math.pow(predictions - label, 2.0))


@tf.function
def initialize_metrics() -> types.MetricsState:
  return collections.OrderedDict(num_examples=(0.0,), accuracy=(0.0, 0.0))


@tf.function
def update_metrics_state(
    state: types.MetricsState,
    labels: Any,
    batch_output: variable.BatchOutput,
    sample_weight: Optional[Any] = None,
) -> types.MetricsState:
  del sample_weight  # Unused.
  batch_size = tf.cast(tf.shape(labels)[0], tf.float32)

  def update_accuracy(variables, labels, predictions):
    accuracy, num_examples = variables
    num_matches = tf.reduce_sum(
        tf.cast(tf.equal(labels, predictions), tf.float32)
    )
    accuracy += num_matches
    num_examples += batch_size
    return accuracy, num_examples

  new_dict = collections.OrderedDict(
      num_examples=(state['num_examples'][0] + batch_size,),
      accuracy=update_accuracy(
          state['accuracy'], labels, batch_output.predictions
      ),
  )
  return new_dict


@tf.function
def finalize_metrics(state: types.MetricsState) -> Any:
  accuracy, num_examples = state['accuracy']
  return collections.OrderedDict(
      num_examples=state['num_examples'][0],
      accuracy=tf.math.divide_no_nan(accuracy, num_examples),
  )


def create_test_keras_model():
  return tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(1,)),
      tf.keras.layers.Dense(
          1, kernel_initializer='zeros', bias_initializer='zeros'
      ),
  ])


def create_test_dataset():
  """Create a test dataset."""

  def preprocess(ds):
    def generate_example(i, t):
      del t  # Unused.
      features = tf.random.stateless_uniform(shape=[3], seed=(0, i))
      label = (
          tf.expand_dims(
              tf.reduce_sum(features * tf.constant([1.0, 2.0, 3.0])), axis=-1
          )
          + 5.0
      )
      return (features, label)

    return ds.map(generate_example).batch(5, drop_remainder=True)

  num_examples = 25
  return preprocess(tf.data.Dataset.range(num_examples).enumerate())


class FunctionalModelErrorsTest(tf.test.TestCase):

  def test_fail_construction_on_tf_value(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    with self.assertRaisesRegex(
        functional.ValueMustNotBeTFError, 'initial_weights may not contain'
    ):
      functional.FunctionalModel(
          initial_weights=(tf.constant(1.0), ()),
          predict_on_batch_fn=predict_on_batch,
          loss_fn=loss,
          input_spec=input_spec,
      )
    with self.assertRaisesRegex(
        functional.ValueMustNotBeTFError, 'initial_weights may not contain'
    ):
      functional.FunctionalModel(
          initial_weights=(tf.Variable(1.0), ()),
          predict_on_batch_fn=predict_on_batch,
          loss_fn=loss,
          input_spec=input_spec,
      )

  def test_fail_non_tf_function(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    with self.assertRaisesRegex(
        functional.CallableNotTensorFlowFunctionError,
        'predict_on_batch_fn does not have a `get_concrete_function`',
    ):
      functional.FunctionalModel(
          initial_weights=(),
          predict_on_batch_fn=predict_on_batch.python_function,
          loss_fn=loss,
          input_spec=input_spec,
      )


class FunctionalModelTest(tf.test.TestCase):

  def test_predict_on_batch(self):
    dataset = create_test_dataset()
    example_batch = next(iter(dataset))
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(
        initial_weights=initial_weights(),
        predict_on_batch_fn=predict_on_batch,
        loss_fn=loss,
        input_spec=input_spec,
    )
    self.assertAllClose(
        functional_model.predict_on_batch(
            functional_model.initial_weights, example_batch[0]
        ),
        [[0.0]] * 5,
    )

  def test_loss(self):
    dataset = create_test_dataset()
    example_batch = next(iter(dataset))
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(
        initial_weights=initial_weights(),
        predict_on_batch_fn=predict_on_batch,
        loss_fn=loss,
        input_spec=input_spec,
    )
    batch_output = functional_model.predict_on_batch(
        functional_model.initial_weights, example_batch[0]
    )
    batch_loss = functional_model.loss(batch_output, example_batch[1])
    self.assertAllClose(batch_loss, 74.250, rtol=1e-03, atol=1e-03)

  def test_predict_on_batch_with_structure(self):
    dataset = create_test_dataset()
    example_batch = next(iter(dataset))
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(
        initial_weights=initial_weights(),
        predict_on_batch_fn=predict_on_batch_with_structure,
        loss_fn=loss_with_structure,
        input_spec=input_spec,
    )
    self.assertAllClose(
        functional_model.predict_on_batch(
            functional_model.initial_weights, example_batch[0]
        ),
        ([[0.0]] * 5, example_batch[0]),  # our test returns predictions and x
    )

  def test_loss_with_structure(self):
    dataset = create_test_dataset()
    example_batch = next(iter(dataset))
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(
        initial_weights=initial_weights(),
        predict_on_batch_fn=predict_on_batch_with_structure,
        loss_fn=loss_with_structure,
        input_spec=input_spec,
    )
    batch_output = functional_model.predict_on_batch(
        functional_model.initial_weights, example_batch[0]
    )
    batch_loss = functional_model.loss(batch_output, example_batch[1])
    self.assertAllClose(batch_loss, 74.250, rtol=1e-03, atol=1e-03)

  def test_metrics_eager(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(
        initial_weights=initial_weights(),
        predict_on_batch_fn=predict_on_batch,
        loss_fn=loss,
        metrics_fns=(
            initialize_metrics,
            update_metrics_state,
            finalize_metrics,
        ),
        input_spec=input_spec,
    )
    with self.subTest('initialize'):
      state = functional_model.initialize_metrics_state()
      self.assertAllClose(
          state,
          collections.OrderedDict(num_examples=(0.0,), accuracy=(0.0, 0.0)),
      )
    with self.subTest('update'):
      batch_output = variable.BatchOutput(predictions=np.asarray([0, 1, 1]))
      labels = np.asarray([0, 0, 1])
      updated_state = functional_model.update_metrics_state(
          state, labels=labels, batch_output=batch_output
      )
      self.assertAllClose(
          updated_state,
          collections.OrderedDict(num_examples=(3.0,), accuracy=(2.0, 3.0)),
      )
    with self.subTest('finalize'):
      final_metrics = functional_model.finalize_metrics(updated_state)
      self.assertAllClose(
          final_metrics,
          collections.OrderedDict(num_examples=3.0, accuracy=2.0 / 3.0),
          msg=final_metrics,
      )

  def test_metrics_graph(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    with tf.Graph().as_default() as g:
      with tf.compat.v1.Session(graph=g) as sess:
        functional_model = functional.FunctionalModel(
            initial_weights=initial_weights(),
            predict_on_batch_fn=predict_on_batch,
            loss_fn=loss,
            metrics_fns=(
                initialize_metrics,
                update_metrics_state,
                finalize_metrics,
            ),
            input_spec=input_spec,
        )
        with self.subTest('initialize'):
          initial_state = sess.run(functional_model.initialize_metrics_state())
          self.assertAllClose(
              initial_state,
              collections.OrderedDict(num_examples=(0.0,), accuracy=(0.0, 0.0)),
          )
        with self.subTest('update'):
          state_placeholder = tf.nest.map_structure(
              lambda t: tf.compat.v1.placeholder(tf.float32), initial_state
          )
          predictions = tf.compat.v1.placeholder(tf.int32)
          labels = tf.compat.v1.placeholder(tf.int32)
          updated_state = sess.run(
              fetches=functional_model.update_metrics_state(
                  state_placeholder,
                  labels=labels,
                  batch_output=variable.BatchOutput(predictions=predictions),
              ),
              feed_dict={
                  predictions: np.asarray([0, 1, 1]),
                  labels: np.asarray([0, 0, 1]),
                  **{
                      placeholder: value
                      for placeholder, value in zip(
                          tf.nest.flatten(state_placeholder),
                          tf.nest.flatten(initial_state),
                      )
                  },
              },
          )
          self.assertAllClose(
              updated_state,
              collections.OrderedDict(num_examples=(3.0,), accuracy=(2.0, 3.0)),
              msg=updated_state,
          )
        with self.subTest('finalize'):
          final_metrics = sess.run(
              fetches=functional_model.finalize_metrics(state_placeholder),
              feed_dict={
                  placeholder: value
                  for placeholder, value in zip(
                      tf.nest.flatten(state_placeholder),
                      tf.nest.flatten(updated_state),
                  )
              },
          )
          self.assertAllClose(
              final_metrics,
              collections.OrderedDict(num_examples=3.0, accuracy=2.0 / 3.0),
              msg=final_metrics,
          )

  def test_functional_model_converges(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(
        initial_weights=initial_weights(),
        predict_on_batch_fn=predict_on_batch,
        loss_fn=loss,
        input_spec=input_spec,
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)
    variables = tf.nest.map_structure(
        tf.Variable, functional_model.initial_weights
    )
    trainable = variables[0]
    loss_value = None
    num_epochs = 50
    for batch in dataset.repeat(num_epochs):
      with tf.GradientTape() as tape:
        batch_output = functional_model.predict_on_batch(
            model_weights=variables, x=batch[0], training=True
        )
        batch_loss = functional_model.loss(output=batch_output, label=batch[1])
      gradients = tape.gradient(batch_loss, trainable)
      optimizer.apply_gradients(zip(gradients, trainable))
      loss_value = batch_loss
    # Expect some amount of convergence after a few epochs of the dataset.
    self.assertLess(loss_value, 0.1)
    self.assertAllClose(trainable, ([[1.0, 2.0, 3.0]], [5.0]), atol=0.5)


class ModelFromFunctionalModelTest(tf.test.TestCase):

  def test_tff_model_from_functional_same_result(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(
        initial_weights=initial_weights(),
        predict_on_batch_fn=predict_on_batch,
        loss_fn=loss,
        input_spec=input_spec,
    )
    self.assert_variable_model_from_functional_same_result(
        dataset, functional_model
    )

  def test_tff_model_from_functional_with_structure_same_result(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(
        initial_weights=initial_weights(),
        predict_on_batch_fn=predict_on_batch_with_structure,
        loss_fn=loss_with_structure,
        input_spec=input_spec,
    )
    self.assert_variable_model_from_functional_same_result(
        dataset, functional_model
    )

  def assert_variable_model_from_functional_same_result(
      self, dataset, functional_model
  ):
    tff_model = functional.model_from_functional(functional_model)

    for training in [True, False]:
      for batch in dataset:
        tff_model_batch_output = tff_model.predict_on_batch(batch[0], training)
        tff_model_batch_loss, _, tff_model_batch_num_examples = (
            tff_model.forward_pass(batch, training)
        )
        functional_model_batch_output = functional_model.predict_on_batch(
            functional_model.initial_weights, batch[0], training
        )
        functional_model_batch_loss = functional_model.loss(
            output=functional_model_batch_output, label=batch[1]
        )
        functional_model_batch_num_examples = tf.shape(
            tf.nest.flatten(functional_model_batch_output)[0]
        )[0]

        self.assertAllClose(
            tff_model_batch_output, functional_model_batch_output
        )
        self.assertAllClose(tff_model_batch_loss, functional_model_batch_loss)
        self.assertAllClose(
            tff_model_batch_num_examples, functional_model_batch_num_examples
        )

  def test_tff_model_from_functional_converges(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(
        initial_weights=initial_weights(),
        predict_on_batch_fn=predict_on_batch,
        loss_fn=loss,
        input_spec=input_spec,
    )
    tff_model = functional.model_from_functional(functional_model)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)
    loss_value = None
    num_epochs = 50
    for batch in dataset.repeat(num_epochs):
      with tf.GradientTape() as tape:
        batch_output = tff_model.forward_pass(batch, training=True)
      gradients = tape.gradient(
          batch_output.loss, tff_model.trainable_variables
      )
      optimizer.apply_gradients(zip(gradients, tff_model.trainable_variables))
      loss_value = batch_output.loss
    # Expect some amount of convergence after a few epochs of the dataset.
    self.assertLess(loss_value, 0.1)
    self.assertAllClose(
        tff_model.trainable_variables, ([[1.0, 2.0, 3.0]], [5.0]), atol=0.5
    )
    self.assertAllClose(
        tff_model.report_local_unfinalized_metrics(),
        collections.OrderedDict(loss=[1066.19628, 1250.0]),
    )

  def test_tff_model_from_functional_fails_with_repeated_metric_names(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(
        initial_weights=initial_weights(),
        predict_on_batch_fn=predict_on_batch,
        loss_fn=loss,
        input_spec=input_spec,
    )
    metric_constructors = [
        lambda: tf.keras.metrics.MeanSquaredError(name='same_name'),
        lambda: tf.keras.metrics.RootMeanSquaredError(name='same_name'),
    ]
    with self.assertRaisesRegex(
        ValueError, 'each metric should have a unique name'
    ):
      functional.model_from_functional(functional_model, metric_constructors)

  def test_tff_model_from_functional_binding_metrics_succeeds(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(
        initial_weights=initial_weights(),
        predict_on_batch_fn=predict_on_batch,
        loss_fn=loss,
        input_spec=input_spec,
    )
    metric_constructors = [
        tf.keras.metrics.MeanSquaredError,
        tf.keras.metrics.RootMeanSquaredError,
    ]
    tff_model = functional.model_from_functional(
        functional_model, metric_constructors
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)
    loss_value = None
    num_epochs = 50
    for batch in dataset.repeat(num_epochs):
      with tf.GradientTape() as tape:
        batch_output = tff_model.forward_pass(batch, training=True)
      gradients = tape.gradient(
          batch_output.loss, tff_model.trainable_variables
      )
      optimizer.apply_gradients(zip(gradients, tff_model.trainable_variables))
      loss_value = batch_output.loss
    # Expect some amount of convergence after a few epochs of the dataset.
    self.assertLess(loss_value, 0.1)
    self.assertAllClose(
        tff_model.trainable_variables, ([[1.0, 2.0, 3.0]], [5.0]), atol=0.5
    )
    self.assertAllClose(
        tff_model.report_local_unfinalized_metrics(),
        collections.OrderedDict(
            # The model uses mean squred error as `loss`, so the other two
            # metrics (`mean_squared_error` and `root_mean_squared_error`)
            # should have the same state as `loss`.
            loss=[1066.19628, 1250.0],
            mean_squared_error=[1066.19628, 1250.0],
            root_mean_squared_error=[1066.19628, 1250.0],
        ),
    )

  def test_tff_model_from_functional_overwrites_metrics(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(
        initial_weights=initial_weights(),
        predict_on_batch_fn=predict_on_batch,
        loss_fn=loss,
        metrics_fns=(
            initialize_metrics,
            update_metrics_state,
            finalize_metrics,
        ),
        input_spec=input_spec,
    )
    metric_constructors = [
        tf.keras.metrics.MeanSquaredError,
        tf.keras.metrics.RootMeanSquaredError,
    ]
    tff_model = functional.model_from_functional(
        functional_model, metric_constructors
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)
    loss_value = None
    num_epochs = 50
    for batch in dataset.repeat(num_epochs):
      with tf.GradientTape() as tape:
        batch_output = tff_model.forward_pass(batch, training=True)
      gradients = tape.gradient(
          batch_output.loss, tff_model.trainable_variables
      )
      optimizer.apply_gradients(zip(gradients, tff_model.trainable_variables))
      loss_value = batch_output.loss
    # Expect some amount of convergence after a few epochs of the dataset.
    self.assertLess(loss_value, 0.1)
    self.assertAllClose(
        tff_model.trainable_variables, ([[1.0, 2.0, 3.0]], [5.0]), atol=0.5
    )
    self.assertAllClose(
        tff_model.report_local_unfinalized_metrics(),
        collections.OrderedDict(
            # The model uses mean squred error as `loss`, so the other two
            # metrics (`mean_squared_error` and `root_mean_squared_error`)
            # should have the same state as `loss`.
            loss=[1066.19628, 1250.0],
            mean_squared_error=[1066.19628, 1250.0],
            root_mean_squared_error=[1066.19628, 1250.0],
        ),
    )

  def test_tff_model_from_functional_federated_aggregate_metrics_succeeds(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(
        initial_weights=initial_weights(),
        predict_on_batch_fn=predict_on_batch,
        loss_fn=loss,
        input_spec=input_spec,
    )
    metric_constructors = [
        lambda: tf.keras.metrics.MeanSquaredError(name='mse'),
        lambda: tf.keras.metrics.MeanAbsoluteError(name='mae'),
    ]
    tff_model = functional.model_from_functional(
        functional_model, metric_constructors
    )
    client_1_local_outputs = collections.OrderedDict(
        loss=[1.0, 2.0], mse=[1.0, 2.0], mae=[1.0, 2.0]
    )
    client_2_local_outputs = collections.OrderedDict(
        loss=[2.0, 4.0], mse=[2.0, 2.0], mae=[1.0, 6.0]
    )
    metrics_aggregator = aggregator.sum_then_finalize
    unfinalized_metrics_type = computation_types.to_type(
        collections.OrderedDict(
            loss=[np.float32, np.float32],
            mse=[np.float32, np.float32],
            mae=[np.float32, np.float32],
        )
    )

    metrics_aggregation_computation = metrics_aggregator(
        tff_model.metric_finalizers(), unfinalized_metrics_type
    )
    # Tell TFF that the argument is CLIENTS placed, so that when this
    # computation is later invoked on a list of values, TFF will teach each
    # element of the list as a single client value. This cannot be inferred from
    # the value itself.
    @federated_computation.federated_computation(
        computation_types.FederatedType(
            unfinalized_metrics_type, placements.CLIENTS
        )
    )
    def aggregate_metrics(metrics):
      return metrics_aggregation_computation(metrics)

    aggregated_metrics = aggregate_metrics(
        [client_1_local_outputs, client_2_local_outputs]
    )
    self.assertAllClose(
        aggregated_metrics,
        # loss = (1.0+2.0)/(2.0+4.0) = 0.5
        # mse = (1.0+2.0)/(2.0+2.0) = 0.75
        # mae = (1.0+1.0)/(2.0+6.0) = 0.25
        collections.OrderedDict(loss=0.5, mse=0.75, mae=0.25),
    )

  def test_tff_model_from_functional_resets_metrics(self):
    dataset = create_test_dataset()
    input_spec = dataset.element_spec
    functional_model = functional.FunctionalModel(
        initial_weights=initial_weights(),
        predict_on_batch_fn=predict_on_batch,
        loss_fn=loss,
        input_spec=input_spec,
    )
    metric_constructors = [
        tf.keras.metrics.MeanSquaredError,
        tf.keras.metrics.RootMeanSquaredError,
    ]
    tff_model = functional.model_from_functional(
        functional_model, metric_constructors
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)

    expected_initial_local_variables = [0.0, 0, 0.0, 0.0, 0.0, 0.0]
    self.assertSequenceEqual(
        tff_model.local_variables, expected_initial_local_variables
    )

    # Execute forward pass on the dataset, assert metrics are not zero.
    for batch in dataset:
      with tf.GradientTape() as tape:
        batch_output = tff_model.forward_pass(batch, training=True)
      gradients = tape.gradient(
          batch_output.loss, tff_model.trainable_variables
      )
      optimizer.apply_gradients(zip(gradients, tff_model.trainable_variables))
      loss_value = batch_output.loss
    self.assertGreater(loss_value, 0)
    self.assertAllClose(
        tff_model.report_local_unfinalized_metrics(),
        collections.OrderedDict(
            loss=[876.11523, 25.0],
            mean_squared_error=[876.11523, 25.0],
            root_mean_squared_error=[876.11523, 25.0],
        ),
    )

    # Reset metrics variables.
    tff_model.reset_metrics()
    self.assertSequenceEqual(
        tff_model.local_variables, expected_initial_local_variables
    )
    self.assertEqual(
        tff_model.report_local_unfinalized_metrics(),
        collections.OrderedDict(
            loss=[0, 0],
            mean_squared_error=[0, 0],
            root_mean_squared_error=[0, 0],
        ),
    )


class FunctionalModelFromKerasTest(tf.test.TestCase):

  def assert_variableless_function(self, fn, *args, **kwargs):
    graph_def = fn.get_concrete_function(*args, **kwargs).graph.as_graph_def()
    all_nodes = itertools.chain(
        graph_def.node, *[f.node_def for f in graph_def.library.function]
    )
    self.assertEmpty([node.op for node in all_nodes if 'Variable' in node.op])

  def test_construct_from_keras(self):
    keras_model = create_test_keras_model()
    # Assign some variables after initialization so we can assert that they
    # were cloned into the FunctionalModel.
    tf.nest.map_structure(
        lambda v: v.assign(tf.ones_like(v)), keras_model.variables
    )
    x_spec = tf.TensorSpec([None, 1], dtype=tf.float32)
    y_spec = tf.TensorSpec([None, 1], dtype=tf.int32)
    functional_model = functional.functional_model_from_keras(
        keras_model=keras_model,
        loss_fn=tf.keras.losses.MeanSquaredError(),
        input_spec=(x_spec, y_spec),
        metrics_constructor=collections.OrderedDict(
            accuracy=tf.keras.metrics.MeanSquaredError
        ),
    )
    self.assertIsInstance(functional_model, functional.FunctionalModel)

    model_weights_tensor_spec = tf.nest.map_structure(
        lambda n: tf.TensorSpec(n.shape, tf.dtypes.as_dtype(n.dtype)),
        functional_model.initial_weights,
    )
    with tf.Graph().as_default():
      metrics_state_tensor_spec = tf.nest.map_structure(
          tf.TensorSpec.from_tensor, functional_model.initialize_metrics_state()
      )
      predict_on_batch_concrete_structured_fn = tf.function(
          functional_model.predict_on_batch
      ).get_concrete_function(
          model_weights=model_weights_tensor_spec,
          x=x_spec,
          training=True,
      )
      batch_output_tensor_spec_structure = tf.nest.map_structure(
          tf.TensorSpec.from_tensor,
          predict_on_batch_concrete_structured_fn.structured_outputs,
      )
      self.assert_variableless_function(
          functional_model.predict_on_batch,
          model_weights=model_weights_tensor_spec,
          x=x_spec,
      )
      self.assert_variableless_function(
          tf.function(functional_model.loss),
          output=batch_output_tensor_spec_structure,
          label=y_spec,
      )
      self.assert_variableless_function(
          functional_model.initialize_metrics_state
      )
      self.assert_variableless_function(
          functional_model.update_metrics_state,
          state=metrics_state_tensor_spec,
          labels=tf.TensorSpec([1], tf.float32),
          batch_output=variable.BatchOutput(
              predictions=tf.TensorSpec([1], tf.float32)
          ),
      )
      self.assert_variableless_function(
          functional_model.finalize_metrics, state=metrics_state_tensor_spec
      )
    # Assert all ones, instead of zeros from a newly initial model.
    tf.nest.map_structure(
        lambda v: self.assertAllClose(v, tf.ones_like(v)),
        functional_model.initial_weights,
    )

  def test_predict_on_batch_keras_outside_graph_fails(self):
    dataset = create_test_dataset()
    example_batch = next(iter(dataset))
    functional_model = functional.functional_model_from_keras(
        keras_model=create_test_keras_model(),
        loss_fn=tf.keras.losses.MeanSquaredError(),
        input_spec=(
            tf.TensorSpec([None, 1], dtype=tf.float32),
            tf.TensorSpec([None, 1], dtype=tf.int32),
        ),
    )
    with self.assertRaisesRegex(
        functional.KerasFunctionalModelError,
        'only usable inside a `tff.tensorflow.computation`',
    ):
      functional_model.predict_on_batch(
          functional_model.initial_weights, example_batch[0]
      )

  def test_keras_model_with_non_none_sample_weights_fails(self):
    keras_model = create_test_keras_model()

    def loss_fn_with_non_none_sample_weight(y_true, y_pred, sample_weight=2.0):
      return (
          tf.math.reduce_sum(tf.math.pow(y_pred - y_true, 2.0)) * sample_weight
      )

    with self.assertRaisesRegex(
        functional.KerasFunctionalModelError,
        'non-None model_weight',
    ):
      functional.functional_model_from_keras(
          keras_model=keras_model,
          loss_fn=loss_fn_with_non_none_sample_weight,
          input_spec=(
              tf.TensorSpec(shape=[None, 1]),
              tf.TensorSpec(shape=[None, 1]),
          ),
      )

    def loss_fn_with_none_sample_weight(y_true, y_pred, sample_weight=None):
      del sample_weight
      return tf.math.reduce_sum(tf.math.pow(y_pred - y_true, 2.0))

    self.assertIsInstance(
        functional.functional_model_from_keras(
            keras_model=keras_model,
            loss_fn=loss_fn_with_none_sample_weight,
            input_spec=(
                tf.TensorSpec(shape=[None, 1]),
                tf.TensorSpec(shape=[None, 1]),
            ),
        ),
        functional.FunctionalModel,
    )

  def test_construct_from_keras_converges(self):
    functional_model = functional.functional_model_from_keras(
        keras_model=create_test_keras_model(),
        loss_fn=tf.keras.losses.MeanSquaredError(),
        input_spec=(
            tf.TensorSpec([None, 1], dtype=tf.float32),
            tf.TensorSpec([None, 1], dtype=tf.int32),
        ),
    )
    with tf.Graph().as_default() as test_graph:
      # Capture all the variables for later initialization in the session,
      # otherwise it's hard to get our hands on the Keras-owned variables.
      variable_creation_scope = variable_utils.record_variable_creation_scope()
      with variable_creation_scope as captured_variables:
        # Create data satisfying y = 2*x + 1
        dataset = tf.data.Dataset.from_tensor_slices((
            # Features
            [[1.0], [2.0], [3.0]],
            # Labels.
            [[3.0], [5.0], [7.0]],
        )).batch(1)
        variables = tf.nest.map_structure(
            tf.Variable, functional_model.initial_weights
        )
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

        @tf.function
        def train():
          weights = tf.nest.map_structure(lambda v: v.read_value(), variables)
          batch_input = next(iter(dataset))
          batch_output = functional_model.predict_on_batch(
              weights, batch_input[0], training=True
          )
          initial_loss = loss_value = functional_model.loss(
              batch_output, batch_input[1]
          )
          trainable = variables[0]
          for batch in dataset.repeat(30):
            with tf.GradientTape() as tape:
              weights = tf.nest.map_structure(
                  lambda v: v.read_value(), variables
              )
              tape.watch(weights[0])
              batch_output = functional_model.predict_on_batch(
                  weights, batch[0], training=True
              )
              batch_loss = functional_model.loss(batch_output, batch[1])
            gradients = tape.gradient(batch_loss, weights[0])
            optimizer.apply_gradients(zip(gradients, trainable))
            loss_value = batch_loss
          return initial_loss, loss_value

        initial_loss, final_loss = train()
    with tf.compat.v1.Session(graph=test_graph) as sess:
      sess.run(tf.compat.v1.initializers.variables(captured_variables))
      initial_loss, final_loss = sess.run([initial_loss, final_loss])
    # Expect some amount of convergence after a few epochs of the dataset.
    self.assertGreater(initial_loss, 2.0)
    self.assertLess(final_loss, 0.2)

  def test_keras_model_with_non_trainable_variables_fails(self):
    inputs = tf.keras.layers.Input(shape=[1])
    d = tf.keras.layers.Dense(1)
    d.trainable = False
    outputs = d(inputs)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    with self.assertRaisesRegex(
        functional.KerasFunctionalModelError, 'non-trainable variables'
    ):
      functional.functional_model_from_keras(
          keras_model,
          tf.keras.losses.MeanSquaredError(),
          input_spec=(
              tf.TensorSpec(shape=[None, 1]),
              tf.TensorSpec(shape=[None, 1]),
          ),
      )

  def test_keras_model_with_batch_normalization_fails(self):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=[10]),
        tf.keras.layers.BatchNormalization(),
    ])
    with self.assertRaisesRegex(
        functional.KerasFunctionalModelError, 'batch normalization'
    ):
      functional.functional_model_from_keras(
          model,
          tf.keras.losses.MeanSquaredError(),
          input_spec=(
              tf.TensorSpec(shape=[None, 10]),
              tf.TensorSpec(shape=[None, 1]),
          ),
      )

  def test_keras_layer_capturing_other_layer_fails(self):
    class SharedLayer(tf.keras.layers.Layer):

      def __init__(self, dense_layer: tf.keras.layers.Dense, **kwargs):
        super().__init__(**kwargs)
        self._dense_layer = dense_layer
        self.kernel = dense_layer.kernel
        self.bias = dense_layer.bias

      def call(self, inputs):
        return inputs

      def get_config(self):
        config = super().get_config()
        config.update({'dense_layer': self._dense_layer})
        return config

    inputs = tf.keras.layers.Input(shape=[1])
    layer1 = tf.keras.layers.Dense(1)
    y = layer1(inputs)
    layer2 = SharedLayer(layer1)
    outputs = layer2(y)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    with self.assertRaisesRegex(
        functional.KerasFunctionalModelError, 'sharing variables across layers'
    ):
      functional.functional_model_from_keras(
          keras_model,
          tf.keras.losses.MeanSquaredError(),
          input_spec=(
              tf.TensorSpec(shape=[None, 1]),
              tf.TensorSpec(shape=[None, 1]),
          ),
      )

  def test_keras_layer_input_other_layer_fails(self):
    # A variant of test_keras_layer_capturing_other_layer_fails, but
    # instead of passing the layer in the construction, it takes the other
    # layer as an input to `call`.

    class SharedLayer(tf.keras.layers.Layer):

      def call(
          self, inputs: tf.Tensor, dense_layer: tf.keras.layers.Dense
      ) -> tf.Tensor:
        return inputs @ dense_layer.kernel + dense_layer.bias

    def create_test_model():
      inputs = tf.keras.layers.Input(shape=[1])
      layer1 = tf.keras.layers.Dense(1)
      y = layer1(inputs)
      layer2 = SharedLayer()
      outputs = layer2(y, layer1)
      return tf.keras.Model(inputs=inputs, outputs=outputs)

    with self.assertRaisesRegex(
        functional.KerasFunctionalModelError,
        'has a layer that receives inputs from other layers directly',
    ):
      functional.functional_model_from_keras(
          create_test_model(),
          tf.keras.losses.MeanSquaredError(),
          input_spec=(
              tf.TensorSpec(shape=[None, 1]),
              tf.TensorSpec(shape=[None, 1]),
          ),
      )
    functional.functional_model_from_keras(
        create_test_model,
        tf.keras.losses.MeanSquaredError(),
        input_spec=(
            tf.TensorSpec(shape=[None, 1]),
            tf.TensorSpec(shape=[None, 1]),
        ),
    )


class KerasModelFromFunctionalWeightsTest(tf.test.TestCase):

  def test_keras_model_created_in_graph(self):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=[3]),
        tf.keras.layers.BatchNormalization(
            beta_initializer='zeros',
            gamma_initializer='zeros',
            moving_mean_initializer='zeros',
            moving_variance_initializer='zeros',
        ),
        tf.keras.layers.Dense(
            1, kernel_initializer='zeros', bias_initializer='zeros'
        ),
    ])
    # Assert the initial variables are all zero.
    self.assertAllClose(
        model.variables, tf.nest.map_structure(tf.zeros_like, model.variables)
    )
    trainable_weights = (
        # Batch norm: gamma, beta
        np.ones(shape=[3]) * 1.0,
        np.ones(shape=[3]) * 1.0,
        # Dense: kernel, bias
        np.ones(shape=[3, 1]) * 2.0,
        np.ones(shape=[1]) * 2.0,
    )
    non_trainable_weights = (
        # Batch norm: moving_mean, moving_variance
        np.ones(shape=[3]) * 3.0,
        np.ones(shape=[3]) * 3.0,
    )
    weights = (trainable_weights, non_trainable_weights)
    # Now create a new keras model using the structure with model weights that
    # are all twos.
    with tf.Graph().as_default() as g:
      new_model = functional.keras_model_from_functional_weights(
          model_weights=weights, keras_model=model
      )
      initializer = tf.compat.v1.initializers.variables(new_model.variables)
    with tf.compat.v1.Session(graph=g) as sess:
      sess.run(initializer)
      self.assertAllClose(
          sess.run(new_model.variables),
          # Note: the order of variables is not he same as the creation order,
          # but rather in layer, then within-layer creation, order.
          (
              # Batch norm: gamma, beta, moving_mean, moving_variance
              np.ones(shape=[3]) * 1.0,
              np.ones(shape=[3]) * 1.0,
              np.ones(shape=[3]) * 3.0,
              np.ones(shape=[3]) * 3.0,
              # Dense: kernel, bias
              np.ones(shape=[3, 1]) * 2.0,
              np.ones(shape=[1]) * 2.0,
          ),
      )

  def test_keras_model_in_eager_fails(self):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=[3]),
        tf.keras.layers.Dense(
            1, kernel_initializer='zeros', bias_initializer='zeros'
        ),
    ])
    trainable_weights = (np.ones(shape=[3, 1]) * 2.0, np.ones(shape=[1]) * 2.0)
    non_trainable_weights = ()
    weights = (trainable_weights, non_trainable_weights)
    with self.assertRaisesRegex(
        ValueError, 'can only be called from within a graph context'
    ):
      functional.keras_model_from_functional_weights(
          model_weights=weights, keras_model=model
      )

  def test_keras_model_too_few_weights_fails(self):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=[3]),
        tf.keras.layers.Dense(
            1, kernel_initializer='zeros', bias_initializer='zeros'
        ),
    ])
    trainable_weights = (np.ones(shape=[3, 1]) * 2.0,)
    non_trainable_weights = ()
    weights = (trainable_weights, non_trainable_weights)
    with tf.Graph().as_default():
      with self.assertRaisesRegex(ValueError, 'contains fewer weights'):
        functional.keras_model_from_functional_weights(
            model_weights=weights, keras_model=model
        )

  def test_keras_model_too_many_weights_fails(self):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=[3]),
        tf.keras.layers.Dense(
            1, kernel_initializer='zeros', bias_initializer='zeros'
        ),
    ])
    trainable_weights = (
        np.ones(shape=[3, 1]) * 2.0,
        np.ones(shape=[1]) * 2.0,
        np.ones(shape=[3, 1]) * 2.0,
    )
    non_trainable_weights = ()
    weights = (trainable_weights, non_trainable_weights)
    with tf.Graph().as_default():
      with self.assertRaisesRegex(ValueError, 'contained more variables'):
        functional.keras_model_from_functional_weights(
            model_weights=weights, keras_model=model
        )


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
