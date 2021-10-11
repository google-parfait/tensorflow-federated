# Copyright 2018, The TensorFlow Federated Authors.
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
"""Simple examples implementing the Model interface."""

import collections
import functools
from typing import Callable, List, OrderedDict, Union

import tensorflow as tf

from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.learning import model


class LinearRegression(model.Model):
  """Example of a simple linear regression implemented directly."""

  def __init__(self, feature_dim: int = 2):
    # Define all the variables, similar to what Keras Layers and Models
    # do in build().
    self._feature_dim = feature_dim
    # TODO(b/124070381): Support for integers in num_examples, etc., is handled
    # here in learning, by adding an explicit cast to a float where necessary in
    # order to pass typechecking in the reference executor.
    self._num_examples = tf.Variable(0, trainable=False)
    self._num_batches = tf.Variable(0, trainable=False)
    self._loss_sum = tf.Variable(0.0, trainable=False)
    self._a = tf.Variable([[0.0]] * feature_dim, trainable=True)
    self._b = tf.Variable(0.0, trainable=True)
    # Define a non-trainable model variable (another bias term) for code
    # coverage in testing.
    self._c = tf.Variable(0.0, trainable=False)
    self._input_spec = collections.OrderedDict(
        x=tf.TensorSpec([None, self._feature_dim], tf.float32),
        y=tf.TensorSpec([None, 1], tf.float32))

  @property
  def trainable_variables(self) -> List[tf.Variable]:
    return [self._a, self._b]

  @property
  def non_trainable_variables(self) -> List[tf.Variable]:
    return [self._c]

  @property
  def local_variables(self) -> List[tf.Variable]:
    return [self._num_examples, self._num_batches, self._loss_sum]

  @property
  def input_spec(self):
    # Model expects batched input, but the batch dimension is unspecified.
    return self._input_spec

  @tf.function
  def predict_on_batch(self, x, training=True):
    del training  # Unused.
    return tf.matmul(x, self._a) + self._b + self._c

  @tf.function
  def forward_pass(self, batch_input, training=True) -> model.BatchOutput:
    if not self._input_spec['y'].is_compatible_with(batch_input['y']):
      raise ValueError("Expected batch_input['y'] to be compatible with "
                       f"{self._input_spec['y']} but found {batch_input['y']}")
    if not self._input_spec['x'].is_compatible_with(batch_input['x']):
      raise ValueError("Expected batch_input['x'] to be compatible with "
                       "{self._input_spec['x']} but found {batch_input['x']}")
    predictions = self.predict_on_batch(x=batch_input['x'], training=training)
    residuals = predictions - batch_input['y']
    num_examples = tf.gather(tf.shape(predictions), 0)
    total_loss = 0.5 * tf.reduce_sum(tf.pow(residuals, 2))

    self._loss_sum.assign_add(total_loss)
    self._num_examples.assign_add(num_examples)
    self._num_batches.assign_add(1)

    average_loss = total_loss / tf.cast(num_examples, tf.float32)
    return model.BatchOutput(
        loss=average_loss, predictions=predictions, num_examples=num_examples)

  @tf.function
  def report_local_outputs(self):
    return collections.OrderedDict(
        num_examples=self._num_examples,
        num_examples_float=tf.cast(self._num_examples, tf.float32),
        num_batches=self._num_batches,
        loss=self._loss_sum / tf.cast(self._num_examples, tf.float32))

  @property
  def federated_output_computation(self) -> computation_base.Computation:

    @computations.federated_computation(
        computation_types.at_clients(
            collections.OrderedDict(
                num_examples=tf.int32,
                num_examples_float=tf.float32,
                num_batches=tf.int32,
                loss=tf.float32)))
    def fed_output(local_outputs):
      # TODO(b/124070381): Remove need for using num_examples_float here.
      return collections.OrderedDict(
          loss=intrinsics.federated_mean(
              local_outputs.loss, weight=local_outputs.num_examples_float),
          num_examples=intrinsics.federated_sum(local_outputs.num_examples))

    return fed_output

  @tf.function
  def report_local_unfinalized_metrics(
      self) -> OrderedDict[str, Union[tf.Tensor, List[tf.Tensor]]]:
    """Creates an `OrderedDict` of metric names to unfinalized values.

    Returns:
      An `OrderedDict` of metric names to unfinalized values. The `OrderedDict`
      has the same keys (metric names) as the `OrderedDict` returned by the
      method `metric_finalizers()`, and can be used as input to the finalizers
      to get the finalized metric values. This method and `metric_finalizers()`
      method can be used together to build a cross-client metrics aggregator
      when defining the federated training processes or evaluation computations.
    """
    return collections.OrderedDict(
        num_examples=self._num_examples,
        loss=[self._loss_sum,
              tf.cast(self._num_examples, tf.float32)])

  def metric_finalizers(
      self
  ) -> OrderedDict[str, Callable[[Union[tf.Tensor, List[tf.Tensor]]],
                                 tf.Tensor]]:
    """Creates an `OrderedDict` of metric names to finalizers.

    Returns:
      An `OrderedDict` of metric names to finalizers. A finalizer is a
      `tf.function` decorated callable that takes in a metric's unfinalized
      values (returned by `report_local_unfinalized_metrics()`), and returns the
      finalized values. This method and the `report_local_unfinalized_metrics()`
      method can be used together to construct a cross-client metrics aggregator
      when defining the federated training processes or evaluation computations.
    """
    return collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x),
        loss=tf.function(func=lambda x: x[0] / x[1]))


def _dense_all_zeros_layer(input_dims=None, output_dim=1):
  """Create a layer that can be used in isolation for linear regression.

  Constructs a Keras dense layer with a single output, using biases and weights
  that are initialized to zero. No activation function is applied. When this is
  the only layer in a model, the model is effectively a linear regression model.

  Args:
    input_dims: The integer length of the input to this layers. Maybe None if
      the layer input size does not need to be specified.
    output_dim: The integer length of the flattened output tensor. Defaults to
      one, effectively making the layer perform linear regression.

  Returns:
    a `tf.keras.layers.Dense` object.
  """
  build_keras_dense_layer = functools.partial(
      tf.keras.layers.Dense,
      units=output_dim,
      use_bias=True,
      kernel_initializer='zeros',
      bias_initializer='zeros',
      activation=None)
  if input_dims is not None:
    return build_keras_dense_layer(input_shape=(input_dims,))
  return build_keras_dense_layer()


def _dense_all_zeros_regularized_layer(input_dims=None,
                                       output_dim=1,
                                       regularization_constant=0.01):
  """Create a layer that can be used in isolation for linear regression.

  Constructs a Keras dense layer with a single output, using biases and weights
  that are initialized to zero. No activation function is applied. When this is
  the only layer in a model, the model is effectively a linear regression model.
  The regularization constant is used to scale L2 regularization on the weights
  and bias.

  Args:
    input_dims: The integer length of the input to this layers. Maybe None if
      the layer input size does not need to be specified.
    output_dim: The integer length of the flattened output tensor. Defaults to
      one, effectively making the layer perform linear regression.
    regularization_constant: The float scaling magnitude (lambda) for L2
      regularization on the layer's weights and bias.

  Returns:
    a `tf.keras.layers.Dense` object.
  """
  regularizer = tf.keras.regularizers.l2(regularization_constant)
  build_keras_dense_layer = functools.partial(
      tf.keras.layers.Dense,
      units=output_dim,
      use_bias=True,
      kernel_initializer='zeros',
      bias_initializer='zeros',
      kernel_regularizer=regularizer,
      bias_regularizer=regularizer,
      activation=None)
  if input_dims is not None:
    return build_keras_dense_layer(input_shape=(input_dims,))
  return build_keras_dense_layer()


def _dense_all_ones_regularized_layer(input_dims=None,
                                      output_dim=1,
                                      regularization_constant=0.01):
  """Create a layer that can be used in isolation for linear regression.

  Constructs a Keras dense layer with a single output, using biases and weights
  that are initialized to ones. No activation function is applied. When this is
  the only layer in a model, the model is effectively a linear regression model.
  The regularization constant is used to scale L2 regularization on the weights
  and bias.

  Args:
    input_dims: The integer length of the input to this layers. Maybe None if
      the layer input size does not need to be specified.
    output_dim: The integer length of the flattened output tensor. Defaults to
      one, effectively making the layer perform linear regression.
    regularization_constant: The float scaling magnitude (lambda) for L2
      regularization on the layer's weights and bias.

  Returns:
    a `tf.keras.layers.Dense` object.
  """
  regularizer = tf.keras.regularizers.l2(regularization_constant)
  build_keras_dense_layer = functools.partial(
      tf.keras.layers.Dense,
      units=output_dim,
      use_bias=True,
      kernel_initializer='ones',
      bias_initializer='ones',
      kernel_regularizer=regularizer,
      bias_regularizer=regularizer,
      activation=None)
  if input_dims is not None:
    return build_keras_dense_layer(input_shape=(input_dims,))
  return build_keras_dense_layer()


def build_linear_regression_keras_sequential_model(feature_dims=2):
  """Build a linear regression `tf.keras.Model` using the Sequential API."""
  keras_model = tf.keras.models.Sequential()
  keras_model.add(_dense_all_zeros_layer(feature_dims))
  return keras_model


def build_linear_regression_regularized_keras_sequential_model(
    feature_dims=2, regularization_constant=0.01):
  """Build a linear regression `tf.keras.Model` using the Sequential API."""
  keras_model = tf.keras.models.Sequential()
  keras_model.add(
      _dense_all_zeros_regularized_layer(
          feature_dims, regularization_constant=regularization_constant))
  return keras_model


def build_linear_regression_ones_regularized_keras_sequential_model(
    feature_dims=2, regularization_constant=0.01):
  """Build a linear regression `tf.keras.Model` using the Sequential API."""
  keras_model = tf.keras.models.Sequential()
  keras_model.add(
      _dense_all_ones_regularized_layer(
          feature_dims, regularization_constant=regularization_constant))
  return keras_model


def build_linear_regression_keras_functional_model(feature_dims=2):
  """Build a linear regression `tf.keras.Model` using the functional API."""
  a = tf.keras.layers.Input(shape=(feature_dims,), dtype=tf.float32)
  b = _dense_all_zeros_layer()(a)
  return tf.keras.Model(inputs=a, outputs=b)


def build_linear_regression_keras_subclass_model(feature_dims=2):
  """Build a linear regression model by sub-classing `tf.keras.Model`."""
  del feature_dims  # Unused.

  class _KerasLinearRegression(tf.keras.Model):

    def __init__(self):
      super().__init__()
      self._weights = _dense_all_zeros_layer()

    def call(self, inputs, training=True):
      return self._weights(inputs)

  return _KerasLinearRegression()


def build_embedding_keras_model(vocab_size=10):
  """Builds a test model with an embedding initialized to one-hot vectors."""
  keras_model = tf.keras.models.Sequential()
  keras_model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=5))
  keras_model.add(tf.keras.layers.Softmax())
  return keras_model


def build_conv_batch_norm_keras_model():
  """Builds a test model with convolution and batch normalization."""
  # This is an example of a model that has trainable and non-trainable
  # variables.
  l = tf.keras.layers
  data_format = 'channels_last'
  max_pool = l.MaxPooling2D((2, 2), (2, 2),
                            padding='same',
                            data_format=data_format)
  keras_model = tf.keras.models.Sequential([
      l.Reshape(target_shape=[28, 28, 1], input_shape=(28 * 28,)),
      l.Conv2D(
          32,
          5,
          padding='same',
          data_format=data_format,
          activation=tf.nn.relu,
          kernel_initializer='zeros',
          bias_initializer='zeros'),
      max_pool,
      l.BatchNormalization(),
      l.Conv2D(
          64,
          5,
          padding='same',
          data_format=data_format,
          activation=tf.nn.relu,
          kernel_initializer='zeros',
          bias_initializer='zeros'),
      max_pool,
      l.BatchNormalization(),
      l.Flatten(),
      l.Dense(
          1024,
          activation=tf.nn.relu,
          kernel_initializer='zeros',
          bias_initializer='zeros'),
      l.Dropout(0.4),
      l.Dense(10, kernel_initializer='zeros', bias_initializer='zeros'),
  ])
  return keras_model


def build_multiple_inputs_keras_model():
  """Builds a test model with two inputs."""
  l = tf.keras.layers
  a = l.Input((1,), name='a')
  b = l.Input((1,), name='b')
  # Each input has a single, independent dense layer, which are combined into
  # a final dense layer.
  output = l.Dense(1)(
      l.concatenate([
          l.Dense(1)(a),
          l.Dense(1)(b),
      ]))
  return tf.keras.Model(inputs={'a': a, 'b': b}, outputs=[output])


def build_multiple_outputs_keras_model():
  """Builds a test model with three outputs."""
  l = tf.keras.layers
  a = l.Input((1,))
  b = l.Input((1,))

  output_a = l.Dense(1)(a)
  output_b = l.Dense(1)(b)
  output_c = l.Dense(1)(l.concatenate([l.Dense(1)(a), l.Dense(1)(b)]))

  return tf.keras.Model(inputs=[a, b], outputs=[output_a, output_b, output_c])


def build_multiple_outputs_regularized_keras_model(
    regularization_constant=0.01):
  """Builds a test model with three outputs.

  All weights are initialized to ones.

  Args:
    regularization_constant: L2 scaling constant (lambda) for all weights and
      biases.

  Returns:
    a `tf.keras.Model` object.
  """
  dense = functools.partial(
      _dense_all_ones_regularized_layer,
      output_dim=1,
      regularization_constant=regularization_constant)
  a = tf.keras.layers.Input((1,))
  b = tf.keras.layers.Input((1,))

  output_a = dense()(a)
  output_b = dense()(b)
  output_c = dense()(tf.keras.layers.concatenate([dense()(a), dense()(b)]))

  return tf.keras.Model(inputs=[a, b], outputs=[output_a, output_b, output_c])


def build_lookup_table_keras_model():
  """Builds a test model with embedding feature columns."""
  l = tf.keras.layers
  a = l.Input(shape=(1,), dtype=tf.string)
  embedded_lookup_feature = tf.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_vocabulary_list(
          key='colors', vocabulary_list=('R', 'G', 'B')),
      dimension=16)
  dense_features = l.DenseFeatures([embedded_lookup_feature])({'colors': a})
  output = l.Dense(1)(dense_features)
  return tf.keras.Model(inputs=[a], outputs=[output])


def build_preprocessing_lookup_keras_model():
  """Builds a test model using processing layers."""
  l = tf.keras.layers
  a = l.Input(shape=(1,), dtype=tf.string)
  encoded = l.experimental.preprocessing.StringLookup(vocabulary=['A', 'B'])(a)
  return tf.keras.Model(inputs=[a], outputs=[encoded])
