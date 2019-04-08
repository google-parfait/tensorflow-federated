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
"""Tests for tensorflow_federated.python.learning.model_utils.

These tests also serve as examples for users who are familiar with Keras.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

from absl.testing import parameterized
import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow as tf

from tensorflow.python.keras import metrics as keras_metrics
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils

nest = tf.contrib.framework.nest


# TODO(b/123578208): Remove this local Sum implementation once TFF's TF version
# is moved back to HEAD. This can be replaced with tf.keras.metrics.Sum.
class Sum(keras_metrics.Metric):
  """Encapsulates metrics that perform a sum operation on the values."""

  def __init__(self, name='sum', dtype=None):
    """Creates a `Sum` instance.

    Args:
      name: string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    super(Sum, self).__init__(name=name, dtype=dtype)
    self.total = self.add_weight('total', initializer=tf.zeros_initializer)

  def update_state(self, values, sample_weight=None):
    """Accumulates statistics for computing the reduction metric.

    Args:
      values: Per-example value.
      sample_weight: (Ignored) weighting of each example. Defaults to 1.

    Returns:
      Update op.
    """
    del sample_weight
    values = tf.cast(values, self._dtype)
    value_sum = tf.reduce_sum(values)
    with tf.control_dependencies([value_sum]):
      return self.total.assign_add(value_sum)

  def result(self):
    return tf.identity(self.total)


class NumBatchesCounter(Sum):
  """A `tf.keras.metrics.Metric` that counts the number of batches seen."""

  def __init__(self, name='num_batches', dtype=tf.int64):
    super(NumBatchesCounter, self).__init__(name, dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return super(NumBatchesCounter, self).update_state(1, sample_weight)


class NumExamplesCounter(Sum):
  """A `tf.keras.metrics.Metric` that counts the number of examples seen."""

  def __init__(self, name='num_examples', dtype=tf.int64):
    super(NumExamplesCounter, self).__init__(name, dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return super(NumExamplesCounter, self).update_state(
        tf.shape(y_pred)[0], sample_weight)


def _create_dummy_batch(feature_dims):
  """Creates a dummy batch of zeros."""
  return collections.OrderedDict([('x', tf.zeros([1, feature_dims])),
                                  ('y', tf.zeros([1]))])


class ModelUtilsTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    tf.keras.backend.clear_session()
    super(ModelUtilsTest, self).setUp()

  def test_model_initializer(self):
    with tf.Graph().as_default() as g:
      model = model_utils.enhance(model_examples.LinearRegression(2))
      init = model_utils.model_initializer(model)
      with self.session(graph=g) as sess:
        sess.run(init)
        # Make sure we can read all the variables
        try:
          sess.run(model.local_variables)
          sess.run(model.weights)
        except tf.errors.FailedPreconditionError:
          self.fail('Expected variables to be initialized, but got '
                    'tf.errors.FailedPreconditionError')

  def test_non_keras_model(self):
    with self.assertRaisesRegexp(TypeError, r'keras\..*\.Model'):
      model_utils.from_keras_model(
          keras_model=0,  # not a tf.keras.Model
          dummy_batch=_create_dummy_batch(1),
          loss=tf.keras.losses.MeanSquaredError())

  # Test class for batches using namedtuple.
  _make_test_batch = collections.namedtuple('TestBatch', ['x', 'y'])

  @parameterized.parameters(
      {
          'dummy_batch':
              collections.OrderedDict([('x', np.ones([1, 1], np.float32)),
                                       ('y', np.zeros([1, 1], np.float32))])
      },
      {
          'dummy_batch':
              collections.OrderedDict([('x', [[1.0]]), ('y', [[0.0]])])
      },
      {'dummy_batch': _make_test_batch(x=[[1.0]], y=[[0.0]])},
  )
  def test_dummy_batch_types(self, dummy_batch):
    keras_model = model_examples.build_linear_regresion_keras_functional_model(
        feature_dims=1)
    tff_model = model_utils.from_keras_model(
        keras_model=keras_model,
        dummy_batch=dummy_batch,
        loss=tf.keras.losses.MeanSquaredError())
    self.assertIsInstance(tff_model, model_utils.EnhancedModel)

  @parameterized.parameters(
      # Test cases for the cartesian product of all parameter values.
      itertools.product(
          [1, 3],
          [
              model_examples.build_linear_regresion_keras_functional_model,
              model_examples.build_linear_regresion_keras_sequential_model,
              model_examples.build_linear_regresion_keras_subclass_model,
          ],
      ))
  def test_tff_model_from_keras_model(self, feature_dims, model_fn):
    keras_model = model_fn(feature_dims)
    tff_model = model_utils.from_keras_model(
        keras_model=keras_model,
        dummy_batch=_create_dummy_batch(feature_dims),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[NumBatchesCounter(), NumExamplesCounter()])
    self.assertIsInstance(tff_model, model_utils.EnhancedModel)

    # Metrics should be zero, though the model wrapper internally executes the
    # forward pass once.
    self.assertSequenceEqual(
        self.evaluate(tff_model.local_variables), [0, 0, 0.0, 0.0])

    batch = {
        'x':
            np.stack([
                np.zeros(feature_dims, np.float32),
                np.ones(feature_dims, np.float32)
            ]),
        'y': [[0.0], [1.0]],
    }
    # from_model() was called without an optimizer which creates a tff.Model.
    # There is no train_on_batch() method available in tff.Model.
    with self.assertRaisesRegexp(AttributeError,
                                 'no attribute \'train_on_batch\''):
      tff_model.train_on_batch(batch)

    output = tff_model.forward_pass(batch)
    # Since the model initializes all weights and biases to zero, we expect
    # all predictions to be zero:
    #    0*x1 + 0*x2 + ... + 0 = 0
    self.assertAllEqual(output.predictions, [[0.0], [0.0]])
    # For the single batch:
    #
    # Example | Prediction | Label | Residual | Loss
    # --------+------------+-------+----------+ -----
    #    1    |    0.0     |  0.0  |    0.0   |  0.0
    #    2    |    0.0     |  1.0  |    1.0   |  1.0
    #
    # Total loss: 1.0
    # Batch average loss:  0.5
    self.assertEqual(self.evaluate(output.loss), 0.5)
    metrics = self.evaluate(tff_model.report_local_outputs())
    self.assertEqual(metrics['num_batches'], [1])
    self.assertEqual(metrics['num_examples'], [2])
    self.assertGreater(metrics['loss'][0], 0)
    self.assertEqual(metrics['loss'][1], 2)

  @parameterized.parameters(
      itertools.product(
          [1, 3],
          [
              model_examples.build_linear_regresion_keras_functional_model,
              model_examples.build_linear_regresion_keras_sequential_model,
              model_examples.build_linear_regresion_keras_subclass_model,
          ],
          [
              tf.keras.losses.MeanSquaredError(),
              'mean_squared_error',
              # TODO(b/124534248): enable after designign weighted losses.
              # tf.keras.losses.mean_squared_error,
          ]))
  def test_tff_model_from_compiled_keras_model(self, feature_dims, model_fn,
                                               loss_fn):
    keras_model = model_fn(feature_dims)
    # If the model is intended to be used for training, it must be compiled.
    keras_model.compile(
        optimizer=gradient_descent.SGD(learning_rate=0.01),
        loss=loss_fn,
        metrics=[NumBatchesCounter(), NumExamplesCounter()])
    tff_model = model_utils.from_compiled_keras_model(
        keras_model=keras_model, dummy_batch=_create_dummy_batch(feature_dims))

    # Metrics should be zero, though the model wrapper internally executes the
    # forward pass once.
    self.assertSequenceEqual(
        self.evaluate(tff_model.local_variables), [0, 0, 0.0, 0.0])

    batch = {
        'x':
            np.stack([
                np.zeros(feature_dims, np.float32),
                np.full(feature_dims, 5.0, np.float32),
            ]),
        'y': [[0.0], [5.0 * feature_dims]],
    }

    prior_loss = float('inf')
    num_iterations = 3
    for _ in range(num_iterations):
      output = self.evaluate(tff_model.train_on_batch(batch))
      self.assertLess(output.loss, prior_loss)
      prior_loss = output.loss

    metrics = self.evaluate(tff_model.report_local_outputs())
    self.assertEqual(metrics['num_batches'], [num_iterations])
    self.assertEqual(metrics['num_examples'], [2 * num_iterations])
    self.assertGreater(metrics['loss'][0], 0)
    self.assertEqual(metrics['loss'][1], 2 * num_iterations)

  def test_keras_model_using_embeddings(self):
    model = model_examples.build_embedding_keras_model()

    def loss_fn(y_true, y_pred):
      loss_per_example = tf.keras.losses.sparse_categorical_crossentropy(
          y_true=y_true, y_pred=y_pred)
      return tf.reduce_mean(loss_per_example)

    model.compile(
        optimizer=adam.Adam(),
        loss=loss_fn,
        metrics=[NumBatchesCounter(), NumExamplesCounter()])

    dummy_batch = collections.OrderedDict([
        ('x', np.zeros([1])),
        ('y', np.zeros([1])),
    ])
    tff_model = model_utils.from_compiled_keras_model(
        keras_model=model, dummy_batch=dummy_batch)

    # Create a batch with the size of the vocab. These examples will attempt to
    # train the embedding so that the model produces
    #   i -> (i / output_size) + 5
    input_vocab_size = 10
    output_vocab_size = 5
    xs = []
    ys = []
    for input_id in range(input_vocab_size):
      xs.append(input_id)
      ys.append((input_id / output_vocab_size + 5) % output_vocab_size)
    batch = {
        'x': np.expand_dims(np.array(xs, dtype=np.int64), axis=-1),
        'y': np.expand_dims(np.array(ys, dtype=np.int64), axis=-1),
    }
    prior_loss = float('inf')

    num_iterations = 3
    for _ in range(num_iterations):
      r = self.evaluate(tff_model.train_on_batch(batch))
      self.assertLess(r.loss, prior_loss)
      prior_loss = r.loss

    m = self.evaluate(tff_model.report_local_outputs())
    self.assertEqual(m['num_batches'], [num_iterations])
    self.assertEqual(m['num_examples'], [input_vocab_size * num_iterations])
    self.assertGreater(m['loss'][0], 0.0)
    self.assertEqual(m['loss'][1], input_vocab_size * num_iterations)

  def test_keras_model_using_batch_norm(self):
    model = model_examples.build_conv_batch_norm_keras_model()

    def loss_fn(y_true, y_pred):
      loss_per_example = tf.keras.losses.sparse_categorical_crossentropy(
          y_true=y_true, y_pred=y_pred)
      return tf.reduce_mean(loss_per_example)

    model.compile(
        optimizer=gradient_descent.SGD(learning_rate=0.01),
        loss=loss_fn,
        metrics=[NumBatchesCounter(), NumExamplesCounter()])

    dummy_batch = collections.OrderedDict([
        ('x', np.zeros([1, 28 * 28], dtype=np.float32)),
        ('y', np.zeros([1, 1], dtype=np.int64)),
    ])
    tff_model = model_utils.from_compiled_keras_model(
        keras_model=model, dummy_batch=dummy_batch)

    batch_size = 2
    batch = {
        'x':
            np.random.uniform(low=0.0, high=1.0,
                              size=[batch_size, 28 * 28]).astype(np.float32),
        'y':
            np.random.random_integers(low=0, high=9, size=[batch_size,
                                                           1]).astype(np.int64),
    }

    num_iterations = 2
    for _ in range(num_iterations):
      self.evaluate(tff_model.train_on_batch(batch))

    m = self.evaluate(tff_model.report_local_outputs())
    self.assertEqual(m['num_batches'], [num_iterations])
    self.assertEqual(m['num_examples'], [batch_size * num_iterations])
    self.assertGreater(m['loss'][0], 0.0)
    self.assertEqual(m['loss'][1], batch_size * num_iterations)

    # Ensure we can assign the FL trained model weights to a new model.
    tff_weights = model_utils.ModelWeights.from_model(tff_model)
    keras_model = model_examples.build_conv_batch_norm_keras_model()
    tff_weights.assign_weights_to(keras_model)

    for keras_w, tff_w in zip(keras_model.weights, tff_weights.keras_weights):
      self.assertAllClose(
          self.evaluate(keras_w),
          self.evaluate(tff_w),
          atol=1e-4,
          msg='Variable [{}]'.format(keras_w.name))

  def test_wrap_tff_model_in_tf_computation(self):
    feature_dims = 3

    @tff.tf_computation()
    def _train_loop():
      keras_model = model_examples.build_linear_regresion_keras_functional_model(
          feature_dims)
      # If the model is intended to be used for training, it must be compiled.
      keras_model.compile(
          optimizer=gradient_descent.SGD(learning_rate=0.01),
          loss=tf.keras.losses.MeanSquaredError(),
          metrics=[NumBatchesCounter(),
                   NumExamplesCounter()])
      tff_model = model_utils.from_compiled_keras_model(
          keras_model=keras_model,
          dummy_batch=_create_dummy_batch(feature_dims))
      batch = {
          'x':
              np.array([[0.0] * feature_dims, [5.0] * feature_dims],
                       dtype=np.float32),
          'y':
              np.array([[0.0], [5.0 * feature_dims]], dtype=np.float32),
      }
      batch_output = tff_model.train_on_batch(batch)
      with tf.control_dependencies(list(batch_output)):
        metrics = tff_model.report_local_outputs()
      return batch_output, metrics

    output, metrics = _train_loop()
    self.assertGreater(output.loss, 0.0)
    self.assertEqual(metrics.num_batches[0], 1)
    self.assertEqual(metrics.num_examples[0], 2)
    self.assertGreater(metrics.loss[0], 0.0)
    self.assertEqual(metrics.loss[1], 2)

  def test_keras_model_federated_output_computation(self):
    feature_dims = 3

    def _make_keras_model():
      keras_model = model_examples.build_linear_regresion_keras_functional_model(
          feature_dims)
      keras_model.compile(
          optimizer=gradient_descent.SGD(learning_rate=0.01),
          loss=tf.keras.losses.MeanSquaredError(),
          metrics=[NumBatchesCounter(),
                   NumExamplesCounter()])
      return keras_model

    def _model_fn():
      return model_utils.from_compiled_keras_model(
          keras_model=_make_keras_model(),
          dummy_batch=_create_dummy_batch(feature_dims))

    num_iterations = 3
    # TODO(b/122081673): This should be a @tf.function and the control
    # dependencies can go away (probably nothing blocking this, but it
    # just needs to be done and tested).
    @tff.tf_computation()
    def _train_loop():
      tff_model = _model_fn()
      ops = []
      for _ in range(num_iterations):
        with tf.control_dependencies(ops):
          batch_output = tff_model.train_on_batch({
              'x': np.ones([2, feature_dims], dtype=np.float32),
              'y': np.ones([2, 1], dtype=np.float32)
          })
          ops = list(batch_output)
      with tf.control_dependencies(ops):
        return (tff_model.report_local_outputs(), tff_model.weights)

    client_local_outputs, tff_weights = _train_loop()

    # Simulate entering the 'SERVER' context with a new graph.
    tf.keras.backend.clear_session()
    aggregated_outputs = _model_fn().federated_output_computation(
        [client_local_outputs])
    aggregated_outputs = collections.OrderedDict(
        anonymous_tuple.to_elements(aggregated_outputs))
    self.assertEqual(aggregated_outputs['num_batches'], num_iterations)
    self.assertEqual(aggregated_outputs['num_examples'], 2 * num_iterations)
    self.assertGreater(aggregated_outputs['loss'], 0.0)

    keras_model = _make_keras_model()
    keras_model.set_weights(
        model_utils.keras_weights_from_tff_weights(tff_weights))

  def test_keras_model_and_optimizer(self):
    # Expect TFF to compile the keras model if given an optimizer.
    keras_model = model_examples.build_linear_regresion_keras_functional_model(
        feature_dims=1)
    tff_model = model_utils.from_keras_model(
        keras_model=keras_model,
        dummy_batch=_create_dummy_batch(1),
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=gradient_descent.SGD(learning_rate=0.01))
    self.assertIsInstance(tff_model, model_utils.EnhancedTrainableModel)
    # pylint: disable=internal-access
    self.assertTrue(hasattr(tff_model._model._keras_model, 'optimizer'))
    # pylint: enable=internal-access

  def test_enhance(self):
    model = model_utils.enhance(model_examples.LinearRegression(3))
    self.assertIsInstance(model, model_utils.EnhancedModel)

    with self.assertRaisesRegexp(ValueError, 'another EnhancedModel'):
      model_utils.EnhancedModel(model)

  def test_enhanced_var_lists(self):

    class BadModel(model_examples.TrainableLinearRegression):

      @property
      def trainable_variables(self):
        return ['not_a_variable']

      @property
      def local_variables(self):
        return 1

      def forward_pass(self, batch, training=True):
        return 'Not BatchOutput'

      def train_on_batch(self, batch):
        return 'Not BatchOutput'

    bad_model = model_utils.enhance(BadModel())
    self.assertRaisesRegexp(TypeError,
                            'Variable', lambda: bad_model.trainable_variables)
    self.assertRaisesRegexp(TypeError,
                            'Iterable', lambda: bad_model.local_variables)
    self.assertRaisesRegexp(TypeError,
                            'BatchOutput', lambda: bad_model.forward_pass(1))
    self.assertRaisesRegexp(TypeError,
                            'BatchOutput', lambda: bad_model.train_on_batch(1))


if __name__ == '__main__':
  tf.enable_eager_execution()
  test.main()
