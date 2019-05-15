# Lint as: python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

import tensorflow as tf

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.learning import model


class LinearRegression(model.Model):
  """Example of a simple linear regression implemented directly."""

  # A tuple (x, y), where 'x' represent features, and 'y' represent labels.
  Batch = collections.namedtuple('Batch', ['x', 'y'])  # pylint: disable=invalid-name

  def __init__(self, feature_dim=2):
    # Define all the variables, similar to what Keras Layers and Models
    # do in build().
    self._feature_dim = feature_dim
    # TODO(b/124070381): Support for integers in num_examples, etc., is handled
    # here in learning, by adding an explicit cast to a float where necessary in
    # order to pass typechecking in the reference executor.
    self._num_examples = tf.Variable(0, name='num_examples', trainable=False)
    self._num_batches = tf.Variable(0, name='num_batches', trainable=False)
    self._loss_sum = tf.Variable(0.0, name='loss_sum', trainable=False)
    self._a = tf.Variable(
        # N.B. The lambda is needed for use in defuns, see ValueError
        # raised from resource_variable_ops.py.
        lambda: tf.zeros(shape=(feature_dim, 1)),
        name='a',
        trainable=True)
    self._b = tf.Variable(0.0, name='b', trainable=True)
    # Define a non-trainable model variable (another bias term)
    # for code coverage in testing.
    self._c = tf.Variable(0.0, name='c', trainable=False)
    self._input_spec = LinearRegression.make_batch(
        x=tf.TensorSpec([None, self._feature_dim], tf.float32),
        y=tf.TensorSpec([None, 1.0], tf.float32))

  @property
  def trainable_variables(self):
    return [self._a, self._b]

  @property
  def non_trainable_variables(self):
    return [self._c]

  @property
  def local_variables(self):
    return [self._num_examples, self._num_batches, self._loss_sum]

  @property
  def input_spec(self):
    # Model expects batched input, but the batch dimension is unspecified.
    return self._input_spec

  @tf.function
  def _predict(self, x):
    return tf.matmul(x, self._a) + self._b + self._c

  @tf.function
  def forward_pass(self, batch, training=True):
    del training  # Unused
    if isinstance(batch, dict):
      batch = self.make_batch(**batch)
    if not self._input_spec.y.is_compatible_with(batch.y):
      raise ValueError('Expected batch.y to be compatible with '
                       '{} but found {}'.format(self._input_spec.y, batch.y))
    if not self._input_spec.x.is_compatible_with(batch.x):
      raise ValueError('Expected batch.x to be compatible with '
                       '{} but found {}'.format(self._input_spec.x, batch.x))
    predictions = self._predict(batch.x)
    residuals = predictions - batch.y
    num_examples = tf.gather(tf.shape(predictions), 0)
    total_loss = 0.5 * tf.reduce_sum(tf.pow(residuals, 2))

    tf.assign_add(self._loss_sum, total_loss)
    tf.assign_add(self._num_examples, num_examples)
    tf.assign_add(self._num_batches, 1)

    average_loss = total_loss / tf.cast(num_examples, tf.float32)
    return model.BatchOutput(loss=average_loss, predictions=predictions)

  @tf.function
  def report_local_outputs(self):
    return collections.OrderedDict([
        ('num_examples', self._num_examples),
        ('num_examples_float', tf.cast(self._num_examples, tf.float32)),
        ('num_batches', self._num_batches),
        ('loss', self._loss_sum / tf.cast(self._num_examples, tf.float32)),
    ])

  @property
  def federated_output_computation(self):

    @tff.federated_computation
    def fed_output(local_outputs):
      # TODO(b/124070381): Remove need for using num_examples_float here.
      return {
          'num_examples':
              tff.federated_sum(local_outputs.num_examples),
          'loss':
              tff.federated_mean(
                  local_outputs.loss, weight=local_outputs.num_examples_float),
      }

    return fed_output

  @classmethod
  def make_batch(cls, x, y):
    """Returns a `Batch` to pass to the forward pass."""
    return cls.Batch(x, y)


class TrainableLinearRegression(LinearRegression, model.TrainableModel):
  """A LinearRegression with trivial SGD training."""

  @tf.function
  def train_on_batch(self, batch):
    # Most users won't implement this, and let us provide the optimizer.
    fp = self.forward_pass(batch)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer.minimize(fp.loss, var_list=self.trainable_variables)
    return fp


def _dense_all_zeros_layer(input_dims=None, output_dim=1):
  """Create a layer that can be used in isolation for linear regression.

  Constructs a Keras dense layer with a single output, using biases and weights
  that are initialized to zero. No activation function is applied. When this is
  the only layer in a model, the model is effectively a linear regression model.

  Args:
    input_dims: the integer length of the input to this layers. Maybe None if
      the layer input size does not need to be specified.
    output_dim: the integer length of the flattened output tensor. Defaults to
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


def build_linear_regresion_keras_sequential_model(feature_dims=2):
  """Build a linear regression `tf.keras.Model` using the Sequential API."""
  keras_model = tf.keras.models.Sequential()
  keras_model.add(_dense_all_zeros_layer(feature_dims))
  return keras_model


def build_linear_regresion_keras_functional_model(feature_dims=2):
  """Build a linear regression `tf.keras.Model` using the functional API."""
  a = tf.keras.layers.Input(shape=(feature_dims,))
  b = _dense_all_zeros_layer()(a)
  return tf.keras.Model(inputs=a, outputs=b)


def build_linear_regresion_keras_subclass_model(feature_dims=2):
  """Build a linear regression model by sub-classing `tf.keras.Model`."""
  del feature_dims  # unused.

  class _KerasLinearRegression(tf.keras.Model):

    def __init__(self):
      super(_KerasLinearRegression, self).__init__()
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
  a = l.Input((1,))
  b = l.Input((1,))
  # Each input has a single, independent dense layer, which are combined into
  # a final dense layer.
  output = l.Dense(1)(
      l.concatenate([
          l.Dense(1)(a),
          l.Dense(1)(b),
      ]))
  return tf.keras.Model(inputs=[a, b], outputs=[output])
