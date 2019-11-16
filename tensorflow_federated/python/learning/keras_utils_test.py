# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
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
"""Tests for keras_utils.

These tests also serve as examples for users who are familiar with Keras.
"""

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python import core as tff
from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils


class NumBatchesCounter(tf.keras.metrics.Sum):
  """A `tf.keras.metrics.Metric` that counts the number of batches seen."""

  def __init__(self, name='num_batches', dtype=tf.int64):  # pylint: disable=useless-super-delegation
    super().__init__(name, dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return super().update_state(1, sample_weight)


class NumExamplesCounter(tf.keras.metrics.Sum):
  """A `tf.keras.metrics.Metric` that counts the number of examples seen."""

  def __init__(self, name='num_examples', dtype=tf.int64):  # pylint: disable=useless-super-delegation
    super().__init__(name, dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return super().update_state(tf.shape(y_pred)[0], sample_weight)


def _create_dummy_batch(feature_dims):
  """Creates a dummy batch of zeros."""
  return collections.OrderedDict([('x', tf.zeros([1, feature_dims])),
                                  ('y', tf.zeros([1]))])


def _create_tff_model_from_keras_model_tuples():
  tuples = []
  for n_dims in [1, 3]:
    for name, model_fn in [
        ('functional',
         model_examples.build_linear_regression_keras_functional_model),
        ('sequential',
         model_examples.build_linear_regression_keras_sequential_model),
        ('sublclass',
         model_examples.build_linear_regression_keras_subclass_model),
    ]:
      tuples.append(('{}_model_{}_dims'.format(name, n_dims), n_dims, model_fn))
  return tuples


def _create_tff_model_from_compiled_keras_model_tuples():
  tuples = []
  for n_dims in [1, 3]:
    for model_name, model_fn in [
        ('functional',
         model_examples.build_linear_regression_keras_functional_model),
        ('sequential',
         model_examples.build_linear_regression_keras_sequential_model),
        ('sublclass',
         model_examples.build_linear_regression_keras_subclass_model),
    ]:
      for loss_name, loss in [
          ('tf.keras.losses_instance', tf.keras.losses.MeanSquaredError()),
          ('string_handle', 'mean_squared_error'),
          # TODO(b/124534248): enable after designing weighted losses.
          # ('tf.keras.losses_function', tf.keras.losses.mean_squared_error),
      ]:
        tuples.append(
            ('{}_model_{}_loss_fn_{}_dims'.format(model_name, loss_name,
                                                  n_dims), n_dims, model_fn,
             loss))
  return tuples


class KerasUtilsTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    tf.keras.backend.clear_session()
    tff.framework.set_default_executor(tff.framework.create_local_executor())
    super().setUp()

  def test_convert_fails_on_non_keras_model(self):
    with self.assertRaisesRegex(TypeError, r'keras\..*\.Model'):
      keras_utils.from_keras_model(
          keras_model=0,  # not a tf.keras.Model
          dummy_batch=_create_dummy_batch(1),
          loss=tf.keras.losses.MeanSquaredError())

  def test_from_compiled_keras_model_fails_on_uncompiled_model(self):
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims=1)

    with self.assertRaisesRegex(ValueError, '`keras_model` must be compiled'):
      keras_utils.from_compiled_keras_model(
          keras_model=keras_model,
          dummy_batch=_create_dummy_batch(feature_dims=1))

  # Test class for batches using namedtuple.
  _make_test_batch = collections.namedtuple('TestBatch', ['x', 'y'])

  @parameterized.named_parameters(
      ('container',
       collections.OrderedDict([('x', np.ones([1, 1], np.float32)),
                                ('y', np.zeros([1, 1], np.float32))])),
      ('container_fn',
       _make_test_batch(
           x=np.ones([1, 1], np.float32), y=np.zeros([1, 1], np.float32))),
  )
  def test_dummy_batch_types(self, dummy_batch):
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims=1)
    tff_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        dummy_batch=dummy_batch,
        loss=tf.keras.losses.MeanSquaredError())
    self.assertIsInstance(tff_model, model_utils.EnhancedModel)

  @parameterized.named_parameters(
      # Test cases for the cartesian product of all parameter values.
      *_create_tff_model_from_keras_model_tuples())
  def test_tff_model_from_keras_model(self, feature_dims, model_fn):
    keras_model = model_fn(feature_dims)
    tff_model = keras_utils.from_keras_model(
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
    with self.assertRaisesRegex(AttributeError,
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
    # Batch average loss: 0.5
    self.assertEqual(self.evaluate(output.loss), 0.5)
    metrics = self.evaluate(tff_model.report_local_outputs())
    self.assertEqual(metrics['num_batches'], [1])
    self.assertEqual(metrics['num_examples'], [2])
    self.assertGreater(metrics['loss'][0], 0)
    self.assertEqual(metrics['loss'][1], 2)

  @parameterized.named_parameters(
      *_create_tff_model_from_compiled_keras_model_tuples())
  def test_tff_model_from_compiled_keras_model(self, feature_dims, model_fn,
                                               loss_fn):
    keras_model = model_fn(feature_dims)
    # If the model is intended to be used for training, it must be compiled.
    keras_model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=loss_fn,
        metrics=[NumBatchesCounter(), NumExamplesCounter()])
    tff_model = keras_utils.from_compiled_keras_model(
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
        optimizer=tf.keras.optimizers.Adam(),
        loss=loss_fn,
        metrics=[NumBatchesCounter(), NumExamplesCounter()])

    dummy_batch = collections.OrderedDict([
        ('x', np.zeros([1])),
        ('y', np.zeros([1])),
    ])
    tff_model = keras_utils.from_compiled_keras_model(
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

  def test_preprocess_batch_only_converts_leaves_to_tensors(self):
    dummy_batch = collections.OrderedDict([
        ('x', [[np.zeros([1, 1], dtype=np.float32)],
               [np.zeros([1, 1], dtype=np.float32)]]),
        ('y', [np.zeros([1, 1], dtype=np.float32)]),
    ])
    processed_batch = keras_utils._preprocess_dummy_batch(dummy_batch)
    self.assertIsInstance(processed_batch['x'], list)
    self.assertIsInstance(processed_batch['y'], list)
    self.assertIsInstance(processed_batch['x'][0], list)
    self.assertIsInstance(processed_batch['x'][0][0], tf.Tensor)
    self.assertIsInstance(processed_batch['y'][0], tf.Tensor)

  def test_keras_model_multiple_inputs(self):
    model = model_examples.build_multiple_inputs_keras_model()

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=tf.keras.losses.MSE,
        metrics=[NumBatchesCounter(), NumExamplesCounter()])

    dummy_batch = collections.OrderedDict([
        ('x', {
            'a': np.zeros([1, 1], dtype=np.float32),
            'b': np.zeros([1, 1], dtype=np.float32),
        }),
        ('y', np.zeros([1, 1], dtype=np.float32)),
    ])
    tff_model = keras_utils.from_compiled_keras_model(
        keras_model=model, dummy_batch=dummy_batch)

    batch_size = 2
    batch = {
        'x': {
            'a': np.ones(shape=[batch_size, 1], dtype=np.float32),
            'b': np.ones(shape=[batch_size, 1], dtype=np.float32),
        },
        'y': np.asarray([[2.0], [2.0]]).astype(np.float32),
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
    keras_model = model_examples.build_multiple_inputs_keras_model()
    tff_weights.assign_weights_to(keras_model)
    keras_model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=tf.keras.losses.MSE,
        metrics=[NumBatchesCounter(), NumExamplesCounter()])
    loaded_model = keras_utils.from_compiled_keras_model(
        keras_model=keras_model, dummy_batch=dummy_batch)

    orig_model_output = tff_model.forward_pass(batch)
    loaded_model_output = loaded_model.forward_pass(batch)
    self.assertAlmostEqual(
        self.evaluate(orig_model_output.loss),
        self.evaluate(loaded_model_output.loss))

  def test_keras_model_using_batch_norm(self):
    model = model_examples.build_conv_batch_norm_keras_model()

    def loss_fn(y_true, y_pred):
      loss_per_example = tf.keras.losses.sparse_categorical_crossentropy(
          y_true=y_true, y_pred=y_pred)
      return tf.reduce_mean(loss_per_example)

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=loss_fn,
        metrics=[NumBatchesCounter(), NumExamplesCounter()])

    dummy_batch = collections.OrderedDict([
        ('x', np.zeros([1, 28 * 28], dtype=np.float32)),
        ('y', np.zeros([1, 1], dtype=np.int64)),
    ])
    tff_model = keras_utils.from_compiled_keras_model(
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

    def assert_all_weights_close(keras_weights, tff_weights):
      for keras_w, tff_w in zip(keras_weights, tff_weights.values()):
        self.assertAllClose(
            self.evaluate(keras_w),
            self.evaluate(tff_w),
            atol=1e-4,
            msg='Variable [{}]'.format(keras_w.name))

    assert_all_weights_close(keras_model.trainable_weights,
                             tff_weights.trainable)
    assert_all_weights_close(keras_model.non_trainable_weights,
                             tff_weights.non_trainable)

  def test_keras_model_federated_output_computation(self):
    feature_dims = 3

    def _make_keras_model():
      keras_model = model_examples.build_linear_regression_keras_functional_model(
          feature_dims)
      keras_model.compile(
          optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
          loss=tf.keras.losses.MeanSquaredError(),
          metrics=[NumBatchesCounter(),
                   NumExamplesCounter()])
      return keras_model

    def _model_fn():
      return keras_utils.from_compiled_keras_model(
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
    keras_utils.assign_weights_to_keras_model(keras_model, tff_weights)

  def test_keras_model_and_optimizer(self):
    # Expect TFF to compile the keras model if given an optimizer.
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims=1)
    tff_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        dummy_batch=_create_dummy_batch(1),
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01))
    self.assertIsInstance(tff_model, model_utils.EnhancedTrainableModel)
    # pylint: disable=internal-access
    self.assertTrue(hasattr(tff_model._model._keras_model, 'optimizer'))
    # pylint: enable=internal-access

  def test_keras_model_multiple_outputs(self):
    keras_model = model_examples.build_multiple_outputs_keras_model()
    dummy_batch = collections.OrderedDict([
        ('x', [
            np.zeros([1, 1], dtype=np.float32),
            np.zeros([1, 1], dtype=np.float32)
        ]),
        ('y', [
            np.zeros([1, 1], dtype=np.float32),
            np.ones([1, 1], dtype=np.float32),
            np.ones([1, 1], dtype=np.float32)
        ]),
    ])

    with self.subTest('loss_output_len_mismatch'):
      with self.assertRaises(ValueError):
        _ = keras_utils.from_keras_model(
            keras_model=keras_model,
            dummy_batch=dummy_batch,
            loss=[
                tf.keras.losses.MeanSquaredError(),
                tf.keras.losses.MeanSquaredError()
            ])

    with self.subTest('invalid_loss'):
      with self.assertRaises(TypeError):
        _ = keras_utils.from_keras_model(
            keras_model=keras_model, dummy_batch=dummy_batch, loss=3)

    with self.subTest('loss_list_no_opt'):
      tff_model = keras_utils.from_keras_model(
          keras_model=keras_model,
          dummy_batch=dummy_batch,
          loss=[
              tf.keras.losses.MeanSquaredError(),
              tf.keras.losses.MeanSquaredError(),
              tf.keras.losses.MeanSquaredError()
          ])

      self.assertIsInstance(tff_model, model_utils.EnhancedModel)
      output = tff_model.forward_pass(dummy_batch)
      self.assertAllClose(output.loss, 2.0)

    with self.subTest('loss_dict_no_opt'):
      tff_model = keras_utils.from_keras_model(
          keras_model=keras_model,
          dummy_batch=dummy_batch,
          loss={
              'dense': tf.keras.losses.MeanSquaredError(),
              'dense_1': tf.keras.losses.MeanSquaredError(),
              'dense_2': tf.keras.losses.MeanSquaredError()
          })

      self.assertIsInstance(tff_model, model_utils.EnhancedModel)
      output = tff_model.forward_pass(dummy_batch)
      self.assertAllClose(output.loss, 2.0)

    with self.subTest('trainable_model'):
      tff_model = keras_utils.from_keras_model(
          keras_model=keras_model,
          dummy_batch=dummy_batch,
          loss=[
              tf.keras.losses.MeanSquaredError(),
              tf.keras.losses.MeanSquaredError(),
              tf.keras.losses.MeanSquaredError()
          ],
          optimizer=tf.keras.optimizers.SGD(learning_rate=0.01))

      self.assertIsInstance(tff_model, model_utils.EnhancedTrainableModel)
      self.assertTrue(hasattr(tff_model._model._keras_model, 'optimizer'))
      output = tff_model.forward_pass(dummy_batch)
      self.assertAllClose(output.loss, 2.0)

    keras_model = model_examples.build_multiple_outputs_keras_model()
    with self.subTest('loss_weights_as_list'):
      tff_model = keras_utils.from_keras_model(
          keras_model=keras_model,
          dummy_batch=dummy_batch,
          loss=[
              tf.keras.losses.MeanSquaredError(),
              tf.keras.losses.MeanSquaredError(),
              tf.keras.losses.MeanSquaredError()
          ],
          loss_weights=[0.1, 0.2, 0.3])

      output = tff_model.forward_pass(dummy_batch)
      self.assertAllClose(output.loss, 0.5)

    with self.subTest('loss_weights_as_dict'):
      tff_model = keras_utils.from_keras_model(
          keras_model=keras_model,
          dummy_batch=dummy_batch,
          loss=[
              tf.keras.losses.MeanSquaredError(),
              tf.keras.losses.MeanSquaredError(),
              tf.keras.losses.MeanSquaredError()
          ],
          loss_weights={
              'dense_5': 0.1,
              'dense_6': 0.2,
              'dense_7': 0.3
          })

      output = tff_model.forward_pass(dummy_batch)
      self.assertAllClose(output.loss, 0.5)

    with self.subTest('loss_weights_assert_fail_list'):
      with self.assertRaises(ValueError):
        _ = keras_utils.from_keras_model(
            keras_model=keras_model,
            dummy_batch=dummy_batch,
            loss=[
                tf.keras.losses.MeanSquaredError(),
                tf.keras.losses.MeanSquaredError(),
                tf.keras.losses.MeanSquaredError()
            ],
            loss_weights=[0.1, 0.2])

    with self.subTest('loss_weights_assert_fail_dict'):
      with self.assertRaises(KeyError):
        _ = keras_utils.from_keras_model(
            keras_model=keras_model,
            dummy_batch=dummy_batch,
            loss=[
                tf.keras.losses.MeanSquaredError(),
                tf.keras.losses.MeanSquaredError(),
                tf.keras.losses.MeanSquaredError()
            ],
            loss_weights={
                'dense_5': 0.1,
                'dense_6': 0.2,
                'dummy': 0.4
            })

  def test_keras_model_lookup_table(self):
    model = model_examples.build_lookup_table_keras_model()

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=tf.keras.losses.MSE,
        metrics=[NumBatchesCounter(), NumExamplesCounter()])

    dummy_batch = collections.OrderedDict([
        ('x', tf.constant([['G']], dtype=tf.string)),
        ('y', tf.zeros([1, 1], dtype=tf.float32)),
    ])
    tff_model = keras_utils.from_compiled_keras_model(
        keras_model=model, dummy_batch=dummy_batch)

    batch_size = 3
    batch = {
        'x': tf.constant([['G'], ['B'], ['R']], dtype=tf.string),
        'y': tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32),
    }

    num_iterations = 2
    for _ in range(num_iterations):
      self.evaluate(tff_model.train_on_batch(batch))

    metrics = self.evaluate(tff_model.report_local_outputs())
    self.assertEqual(metrics['num_batches'], [num_iterations])
    self.assertEqual(metrics['num_examples'], [batch_size * num_iterations])
    self.assertGreater(metrics['loss'][0], 0.0)
    self.assertEqual(metrics['loss'][1], batch_size * num_iterations)

    # Ensure we can assign the FL trained model weights to a new model.
    tff_weights = model_utils.ModelWeights.from_model(tff_model)
    keras_model = model_examples.build_lookup_table_keras_model()
    tff_weights.assign_weights_to(keras_model)
    keras_model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=tf.keras.losses.MSE,
        metrics=[NumBatchesCounter(), NumExamplesCounter()])
    loaded_model = keras_utils.from_compiled_keras_model(
        keras_model=keras_model, dummy_batch=dummy_batch)

    orig_model_output = tff_model.forward_pass(batch)
    loaded_model_output = loaded_model.forward_pass(batch)
    self.assertAlmostEqual(
        self.evaluate(orig_model_output.loss),
        self.evaluate(loaded_model_output.loss))


if __name__ == '__main__':
  test.main()
