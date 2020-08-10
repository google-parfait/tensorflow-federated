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

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.backends.native import execution_contexts
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


def _create_dummy_types(feature_dims):
  """Creates a dummy batch of zeros."""
  return collections.OrderedDict(
      x=tf.TensorSpec(shape=[1, feature_dims], dtype=tf.float32),
      y=tf.TensorSpec(shape=[1], dtype=tf.float32))


def _create_tff_model_from_keras_model_tuples():
  tuples = []
  for n_dims in [1, 3]:
    for name, model_fn in [
        ('functional',
         model_examples.build_linear_regression_keras_functional_model),
        ('sequential',
         model_examples.build_linear_regression_keras_sequential_model),
        ('sequential_regularized', model_examples
         .build_linear_regression_regularized_keras_sequential_model)
    ]:
      tuples.append(('{}_model_{}_dims'.format(name, n_dims), n_dims, model_fn))
  return tuples


class KerasUtilsTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    tf.keras.backend.clear_session()
    super().setUp()

  def assertIsSubClass(self, cls1, cls2):
    if not issubclass(cls1, cls2):
      raise AssertionError('{} is not a subclass of {}'.format(cls1, cls2))

  def test_convert_fails_on_non_keras_model(self):
    with self.assertRaisesRegex(TypeError, r'keras\..*\.Model'):
      keras_utils.from_keras_model(
          keras_model=0,  # not a tf.keras.Model
          input_spec=_create_dummy_types(1),
          loss=tf.keras.losses.MeanSquaredError())

  # Test class for batches using namedtuple.
  _make_test_batch = collections.namedtuple('TestBatch', ['x', 'y'])

  @parameterized.named_parameters(
      ('container',
       collections.OrderedDict(
           [('x', tf.TensorSpec(shape=[None, 1], dtype=tf.float32)),
            ('y', tf.TensorSpec(shape=[None, 1], dtype=tf.float32))])),
      ('container_fn',
       _make_test_batch(
           x=tf.TensorSpec(shape=[1, 1], dtype=tf.float32),
           y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32))),
      ('tff_type',
       computation_types.to_type(
           collections.OrderedDict(
               [('x', tf.TensorSpec(shape=[None, 1], dtype=tf.float32)),
                ('y', tf.TensorSpec(shape=[None, 1], dtype=tf.float32))]))),
  )
  def test_input_spec_batch_types(self, input_spec):
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims=1)
    tff_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.MeanSquaredError())
    self.assertIsInstance(tff_model, model_utils.EnhancedModel)

  @parameterized.named_parameters(
      # Test cases for the cartesian product of all parameter values.
      *_create_tff_model_from_keras_model_tuples())
  def test_tff_model_from_keras_model(self, feature_dims, model_fn):
    keras_model = model_fn(feature_dims)
    tff_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        input_spec=_create_dummy_types(feature_dims),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[NumBatchesCounter(), NumExamplesCounter()])
    self.assertIsInstance(tff_model, model_utils.EnhancedModel)

    # Metrics should be zero, though the model wrapper internally executes the
    # forward pass once.
    self.assertSequenceEqual(
        self.evaluate(tff_model.local_variables), [0, 0, 0.0, 0.0])

    batch = collections.OrderedDict(
        x=np.stack([
            np.zeros(feature_dims, np.float32),
            np.ones(feature_dims, np.float32)
        ]),
        y=[[0.0], [1.0]])
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
    # Note that though regularization might be applied, this has no effect on
    # the loss since all weights are 0.
    # Total loss: 1.0
    # Batch average loss: 0.5
    self.assertEqual(self.evaluate(output.loss), 0.5)
    metrics = self.evaluate(tff_model.report_local_outputs())
    self.assertEqual(metrics['num_batches'], [1])
    self.assertEqual(metrics['num_examples'], [2])
    self.assertGreater(metrics['loss'][0], 0)
    self.assertEqual(metrics['loss'][1], 2)

  def test_tff_model_from_keras_model_regularization(self):
    keras_model = model_examples.build_linear_regression_ones_regularized_keras_sequential_model(
        3)
    tff_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        input_spec=_create_dummy_types(3),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[NumBatchesCounter(), NumExamplesCounter()])
    self.assertIsInstance(tff_model, model_utils.EnhancedModel)

    # Metrics should be zero, though the model wrapper internally executes the
    # forward pass once.
    self.assertSequenceEqual(
        self.evaluate(tff_model.local_variables), [0, 0, 0.0, 0.0])

    batch = collections.OrderedDict(
        x=np.stack([np.zeros(3, np.float32),
                    np.ones(3, np.float32)]),
        y=[[0.0], [1.0]])
    # from_model() was called without an optimizer which creates a tff.Model.
    # There is no train_on_batch() method available in tff.Model.
    with self.assertRaisesRegex(AttributeError,
                                'no attribute \'train_on_batch\''):
      tff_model.train_on_batch(batch)

    output = tff_model.forward_pass(batch)
    # Since the model initializes all weights and biases to zero, we expect
    # all predictions to be zero:
    #    0*x1 + 0*x2 + ... + 0 = 0
    self.assertAllEqual(output.predictions, [[1.0], [4.0]])
    # For the single batch:
    #
    # Example | Prediction | Label | Residual | Loss
    # --------+------------+-------+----------+ -----
    #    1    |    1.0     |  0.0  |    1.0   |  1.0
    #    2    |    4.0     |  1.0  |    3.0   |  9.0
    #
    # Regularization loss: with an L2 regularization constant of 0.01: kernel
    # regularizer loss is (3 * 1**2) * 0.01, bias regularizer loss is
    # 1**2 * 0.01, so total regularization loss is 0.04.
    # Total loss: 10.0
    # Batch average loss: 5.0
    # Total batch loss with regularization: 5.04
    self.assertAlmostEqual(self.evaluate(output.loss), 5.04)
    metrics = self.evaluate(tff_model.report_local_outputs())
    self.assertEqual(metrics['num_batches'], [1])
    self.assertEqual(metrics['num_examples'], [2])
    self.assertGreater(metrics['loss'][0], 0)
    self.assertEqual(metrics['loss'][1], 2)

  @parameterized.named_parameters(*_create_tff_model_from_keras_model_tuples())
  def test_tff_model_from_keras_model_input_spec(self, feature_dims, model_fn):
    keras_model = model_fn(feature_dims)
    tff_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[NumBatchesCounter(), NumExamplesCounter()],
        input_spec=_create_dummy_types(feature_dims))
    self.assertIsInstance(tff_model, model_utils.EnhancedModel)

    # Metrics should be zero, though the model wrapper internally executes the
    # forward pass once.
    self.assertSequenceEqual(
        self.evaluate(tff_model.local_variables), [0, 0, 0.0, 0.0])

    batch = collections.OrderedDict(
        x=np.stack([
            np.zeros(feature_dims, np.float32),
            np.ones(feature_dims, np.float32)
        ]),
        y=[[0.0], [1.0]])
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

  def test_tff_model_from_keras_model_with_custom_loss_with_integer_label(self):

    class _CustomLossRequiringLabelBeInteger(tf.keras.losses.Loss):

      def __init__(self):
        super().__init__(name='custom_loss_requiring_label_be_integer')

      def call(self, y_true, y_pred):
        # Note that this TF function requires that the label `y_true` be of an
        # integer dtype; a TypeError is thrown if `y_true` isn't int32 or int64.
        return tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)

    keras_model = tf.keras.Sequential(
        [tf.keras.Input(shape=(2,)),
         tf.keras.layers.Dense(units=10)])

    input_spec = [
        tf.TensorSpec(shape=[1, 2], dtype=tf.float32),
        tf.TensorSpec(shape=[1], dtype=tf.int64)
    ]

    tff_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        loss=_CustomLossRequiringLabelBeInteger(),
        input_spec=input_spec)

    batch = collections.OrderedDict(
        x=tf.convert_to_tensor(np.ones((1, 2)), dtype=tf.float32),
        y=tf.convert_to_tensor([0], dtype=tf.int64))

    # Expect this call to .forward_pass to succeed (no Errors raised).
    tff_model.forward_pass(batch)

  def test_tff_model_type_spec_from_keras_model_unspecified_sequence_len(self):
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None,)),
        tf.keras.layers.Embedding(input_dim=10, output_dim=10),
        tf.keras.layers.LSTM(1)
    ])
    input_spec = [
        tf.TensorSpec(shape=[None, None], dtype=tf.int64),
        tf.TensorSpec(shape=[None], dtype=tf.float32)
    ]
    tff_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        loss=tf.keras.losses.MeanSquaredError(),
        input_spec=input_spec)
    self.assertIsInstance(tff_model, model_utils.EnhancedModel)
    self.assertEqual(tff_model.input_spec, input_spec)

    batch = collections.OrderedDict(x=np.ones([2, 5], np.int64), y=[0.0, 1.0])
    output = tff_model.forward_pass(batch)

    self.assertAllEqual(output.predictions.shape, [2, 1])

    # A batch with different sequence length should be processed in a similar
    # way
    batch = collections.OrderedDict(x=np.ones([2, 10], np.int64), y=[0.0, 1.0])
    output = tff_model.forward_pass(batch)

    self.assertAllEqual(output.predictions.shape, [2, 1])

  def test_keras_model_using_embeddings(self):
    model = model_examples.build_embedding_keras_model()
    input_spec = collections.OrderedDict(
        x=tf.TensorSpec(shape=[None], dtype=tf.float32),
        y=tf.TensorSpec(shape=[None], dtype=tf.float32))
    tff_model = keras_utils.from_keras_model(
        keras_model=model,
        input_spec=input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[NumBatchesCounter(), NumExamplesCounter()])

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
    batch = collections.OrderedDict(
        x=np.expand_dims(np.array(xs, dtype=np.int64), axis=-1),
        y=np.expand_dims(np.array(ys, dtype=np.int64), axis=-1))

    num_train_steps = 3
    for _ in range(num_train_steps):
      batch_output = self.evaluate(tff_model.forward_pass(batch))
      self.assertGreater(batch_output.loss, 0.0)

    m = self.evaluate(tff_model.report_local_outputs())
    self.assertEqual(m['num_batches'], [num_train_steps])
    self.assertEqual(m['num_examples'], [input_vocab_size * num_train_steps])
    self.assertGreater(m['loss'][0], 0.0)
    self.assertEqual(m['loss'][1], input_vocab_size * num_train_steps)

  def test_keras_model_multiple_inputs(self):
    input_spec = collections.OrderedDict(
        x=collections.OrderedDict(
            a=tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
            b=tf.TensorSpec(shape=[1, 1], dtype=tf.float32)),
        y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32))
    model = model_examples.build_multiple_inputs_keras_model()
    tff_model = keras_utils.from_keras_model(
        keras_model=model,
        input_spec=input_spec,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[NumBatchesCounter(), NumExamplesCounter()])

    batch_size = 2
    real_batch = collections.OrderedDict(
        x=collections.OrderedDict(
            a=np.ones(shape=[batch_size, 1], dtype=np.float32),
            b=np.ones(shape=[batch_size, 1], dtype=np.float32)),
        y=np.asarray([[2.0], [2.0]]).astype(np.float32))

    num_train_steps = 2
    for _ in range(num_train_steps):
      self.evaluate(tff_model.forward_pass(real_batch))

    m = self.evaluate(tff_model.report_local_outputs())
    self.assertEqual(m['num_batches'], [num_train_steps])
    self.assertEqual(m['num_examples'], [batch_size * num_train_steps])
    self.assertGreater(m['loss'][0], 0.0)
    self.assertEqual(m['loss'][1], batch_size * num_train_steps)

    # Ensure we can assign the FL trained model weights to a new model.
    tff_weights = model_utils.ModelWeights.from_model(tff_model)
    keras_model = model_examples.build_multiple_inputs_keras_model()
    tff_weights.assign_weights_to(keras_model)
    loaded_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[NumBatchesCounter(), NumExamplesCounter()])

    orig_model_output = tff_model.forward_pass(real_batch)
    loaded_model_output = loaded_model.forward_pass(real_batch)
    self.assertAlmostEqual(
        self.evaluate(orig_model_output.loss),
        self.evaluate(loaded_model_output.loss))

  def test_keras_model_using_batch_norm(self):
    model = model_examples.build_conv_batch_norm_keras_model()
    input_spec = collections.OrderedDict(
        x=tf.TensorSpec(shape=[None, 28 * 28], dtype=tf.float32),
        y=tf.TensorSpec(shape=[None, 1], dtype=tf.int64))
    tff_model = keras_utils.from_keras_model(
        keras_model=model,
        input_spec=input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[NumBatchesCounter(), NumExamplesCounter()])

    batch_size = 2
    batch = collections.OrderedDict(
        x=np.random.uniform(low=0.0, high=1.0,
                            size=[batch_size, 28 * 28]).astype(np.float32),
        y=np.random.random_integers(low=0, high=9, size=[batch_size,
                                                         1]).astype(np.int64))

    num_train_steps = 2
    for _ in range(num_train_steps):
      self.evaluate(tff_model.forward_pass(batch))

    m = self.evaluate(tff_model.report_local_outputs())
    self.assertEqual(m['num_batches'], [num_train_steps])
    self.assertEqual(m['num_examples'], [batch_size * num_train_steps])
    self.assertGreater(m['loss'][0], 0.0)
    self.assertEqual(m['loss'][1], batch_size * num_train_steps)

    # Ensure we can assign the FL trained model weights to a new model.
    tff_weights = model_utils.ModelWeights.from_model(tff_model)
    keras_model = model_examples.build_conv_batch_norm_keras_model()
    tff_weights.assign_weights_to(keras_model)

    def assert_all_weights_close(keras_weights, tff_weights):
      for keras_w, tff_w in zip(keras_weights, tff_weights):
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
    num_train_steps = 3

    def _make_keras_model():
      keras_model = model_examples.build_linear_regression_keras_functional_model(
          feature_dims)
      return keras_model

    def _model_fn():
      return keras_utils.from_keras_model(
          keras_model=_make_keras_model(),
          input_spec=_create_dummy_types(feature_dims),
          loss=tf.keras.losses.MeanSquaredError(),
          metrics=[NumBatchesCounter(),
                   NumExamplesCounter()])

    @computations.tf_computation()
    def _train():
      # Create variables outside the tf.function.
      tff_model = _model_fn()
      optimizer = tf.keras.optimizers.SGD(0.1)

      @tf.function
      def _train_loop():
        for _ in range(num_train_steps):
          with tf.GradientTape() as tape:
            batch_output = tff_model.forward_pass(
                collections.OrderedDict(
                    x=np.ones([2, feature_dims], dtype=np.float32),
                    y=np.ones([2, 1], dtype=np.float32)))
          gradients = tape.gradient(batch_output.loss,
                                    tff_model.trainable_variables)
          optimizer.apply_gradients(
              zip(gradients, tff_model.trainable_variables))
        return tff_model.report_local_outputs(), tff_model.weights

      return _train_loop()

    # Simulate 'CLIENT' local training.
    client_local_outputs, tff_weights = _train()

    # Simulate entering the 'SERVER' context.
    tf.keras.backend.clear_session()

    aggregated_outputs = _model_fn().federated_output_computation(
        [client_local_outputs])
    self.assertEqual(aggregated_outputs['num_batches'], num_train_steps)
    self.assertEqual(aggregated_outputs['num_examples'], 2 * num_train_steps)
    self.assertGreater(aggregated_outputs['loss'], 0.0)

    keras_model = _make_keras_model()
    keras_utils.assign_weights_to_keras_model(keras_model, tff_weights)

  def test_keras_model_multiple_outputs(self):
    keras_model = model_examples.build_multiple_outputs_keras_model()
    input_spec = collections.OrderedDict(
        x=[
            tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
        ],
        y=[
            tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
        ])

    with self.subTest('loss_output_len_mismatch'):
      with self.assertRaises(ValueError):
        _ = keras_utils.from_keras_model(
            keras_model=keras_model,
            input_spec=input_spec,
            loss=[
                tf.keras.losses.MeanSquaredError(),
                tf.keras.losses.MeanSquaredError()
            ])

    with self.subTest('invalid_loss'):
      with self.assertRaises(TypeError):
        _ = keras_utils.from_keras_model(
            keras_model=keras_model, input_spec=input_spec, loss=3)

    with self.subTest('loss_list_no_opt'):
      tff_model = keras_utils.from_keras_model(
          keras_model=keras_model,
          input_spec=input_spec,
          loss=[
              tf.keras.losses.MeanSquaredError(),
              tf.keras.losses.MeanSquaredError(),
              tf.keras.losses.MeanSquaredError()
          ])

      self.assertIsInstance(tff_model, model_utils.EnhancedModel)
      dummy_batch = collections.OrderedDict(
          x=[
              np.zeros([1, 1], dtype=np.float32),
              np.zeros([1, 1], dtype=np.float32)
          ],
          y=[
              np.zeros([1, 1], dtype=np.float32),
              np.ones([1, 1], dtype=np.float32),
              np.ones([1, 1], dtype=np.float32)
          ])
      output = tff_model.forward_pass(dummy_batch)
      self.assertAllClose(output.loss, 2.0)

    keras_model = model_examples.build_multiple_outputs_keras_model()
    with self.subTest('loss_weights_as_list'):
      tff_model = keras_utils.from_keras_model(
          keras_model=keras_model,
          input_spec=input_spec,
          loss=[
              tf.keras.losses.MeanSquaredError(),
              tf.keras.losses.MeanSquaredError(),
              tf.keras.losses.MeanSquaredError()
          ],
          loss_weights=[0.1, 0.2, 0.3])

      output = tff_model.forward_pass(dummy_batch)
      self.assertAllClose(output.loss, 0.5)

      output = tff_model.forward_pass(dummy_batch)
      self.assertAllClose(output.loss, 0.5)

    with self.subTest('loss_weights_assert_fail_list'):
      with self.assertRaises(ValueError):
        _ = keras_utils.from_keras_model(
            keras_model=keras_model,
            input_spec=input_spec,
            loss=[
                tf.keras.losses.MeanSquaredError(),
                tf.keras.losses.MeanSquaredError(),
                tf.keras.losses.MeanSquaredError()
            ],
            loss_weights=[0.1, 0.2])

    with self.subTest('loss_weights_assert_fail_dict'):
      with self.assertRaises(TypeError):
        _ = keras_utils.from_keras_model(
            keras_model=keras_model,
            input_spec=input_spec,
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

  def test_regularized_keras_model_multiple_outputs(self):
    keras_model = model_examples.build_multiple_outputs_regularized_keras_model(
    )
    input_spec = collections.OrderedDict(
        x=[
            tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
        ],
        y=[
            tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
        ])

    with self.subTest('loss_output_len_mismatch'):
      with self.assertRaises(ValueError):
        _ = keras_utils.from_keras_model(
            keras_model=keras_model,
            input_spec=input_spec,
            loss=[
                tf.keras.losses.MeanSquaredError(),
                tf.keras.losses.MeanSquaredError()
            ])

    with self.subTest('invalid_loss'):
      with self.assertRaises(TypeError):
        _ = keras_utils.from_keras_model(
            keras_model=keras_model, input_spec=input_spec, loss=3)

    with self.subTest('loss_list_no_opt'):
      tff_model = keras_utils.from_keras_model(
          keras_model=keras_model,
          input_spec=input_spec,
          loss=[
              tf.keras.losses.MeanSquaredError(),
              tf.keras.losses.MeanSquaredError(),
              tf.keras.losses.MeanSquaredError()
          ])

      self.assertIsInstance(tff_model, model_utils.EnhancedModel)
      dummy_batch = collections.OrderedDict(
          x=[
              np.zeros([1, 1], dtype=np.float32),
              np.zeros([1, 1], dtype=np.float32)
          ],
          y=[
              np.zeros([1, 1], dtype=np.float32),
              np.ones([1, 1], dtype=np.float32),
              np.ones([1, 1], dtype=np.float32)
          ])
      output = tff_model.forward_pass(dummy_batch)

      # Labels are (0, 1, 1), preds are (1, 1, 3).
      # Total MSE is 1**2 + 0**2 + 2**2 = 5.
      # Since all weights are initialized to ones and regularization constant is
      # 0.01, regularization loss is 0.01 * (num_params). There are 4 dense
      # layers that take in one input and produce one output, and these each
      # have a single weight and a single bias. There is one dense layer with
      # two inputs and one output, so it has two weights and a single bias.
      # So there are 11 params total and regularization loss is 0.11, for a
      # total batch loss of 5.11.
      self.assertAllClose(output.loss, 5.11)

    keras_model = model_examples.build_multiple_outputs_regularized_keras_model(
    )
    with self.subTest('loss_weights_as_list'):
      tff_model = keras_utils.from_keras_model(
          keras_model=keras_model,
          input_spec=input_spec,
          loss=[
              tf.keras.losses.MeanSquaredError(),
              tf.keras.losses.MeanSquaredError(),
              tf.keras.losses.MeanSquaredError()
          ],
          loss_weights=[0.1, 0.2, 0.3])

      output = tff_model.forward_pass(dummy_batch)

      # Labels are (0, 1, 1), preds are (1, 1, 3).
      # Weighted MSE is 0.1 * 1**2 + 0.2 * 0**2 + 0.3 * 2**2 = 1.3.
      # Regularization loss is 0.11 as before, for a total loss of 1.41.
      self.assertAllClose(output.loss, 1.41)

      output = tff_model.forward_pass(dummy_batch)
      self.assertAllClose(output.loss, 1.41)

    with self.subTest('loss_weights_assert_fail_list'):
      with self.assertRaises(ValueError):
        _ = keras_utils.from_keras_model(
            keras_model=keras_model,
            input_spec=input_spec,
            loss=[
                tf.keras.losses.MeanSquaredError(),
                tf.keras.losses.MeanSquaredError(),
                tf.keras.losses.MeanSquaredError()
            ],
            loss_weights=[0.1, 0.2])

    with self.subTest('loss_weights_assert_fail_dict'):
      with self.assertRaises(TypeError):
        _ = keras_utils.from_keras_model(
            keras_model=keras_model,
            input_spec=input_spec,
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
    input_spec = collections.OrderedDict(
        x=tf.TensorSpec(shape=[None, 1], dtype=tf.string),
        y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32))
    tff_model = keras_utils.from_keras_model(
        keras_model=model,
        input_spec=input_spec,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[NumBatchesCounter(), NumExamplesCounter()])

    batch_size = 3
    batch = collections.OrderedDict(
        x=tf.constant([['G'], ['B'], ['R']], dtype=tf.string),
        y=tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32))

    num_train_steps = 2
    for _ in range(num_train_steps):
      self.evaluate(tff_model.forward_pass(batch))

    metrics = self.evaluate(tff_model.report_local_outputs())
    self.assertEqual(metrics['num_batches'], [num_train_steps])
    self.assertEqual(metrics['num_examples'], [batch_size * num_train_steps])
    self.assertGreater(metrics['loss'][0], 0.0)
    self.assertEqual(metrics['loss'][1], batch_size * num_train_steps)

    # Ensure we can assign the FL trained model weights to a new model.
    tff_weights = model_utils.ModelWeights.from_model(tff_model)
    keras_model = model_examples.build_lookup_table_keras_model()
    tff_weights.assign_weights_to(keras_model)
    loaded_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[NumBatchesCounter(), NumExamplesCounter()])

    orig_model_output = tff_model.forward_pass(batch)
    loaded_model_output = loaded_model.forward_pass(batch)
    self.assertAlmostEqual(
        self.evaluate(orig_model_output.loss),
        self.evaluate(loaded_model_output.loss))

  def test_keras_model_fails_compiled(self):
    feature_dims = 3
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims)

    keras_model.compile(loss=tf.keras.losses.MeanSquaredError())

    with self.assertRaisesRegex(ValueError, 'compile'):
      keras_utils.from_keras_model(
          keras_model=keras_model,
          input_spec=_create_dummy_types(feature_dims),
          loss=tf.keras.losses.MeanSquaredError(),
          metrics=[NumBatchesCounter(),
                   NumExamplesCounter()])


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test.main()
