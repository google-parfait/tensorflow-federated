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
import warnings

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model as model_lib
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.metrics import aggregator
from tensorflow_federated.python.learning.metrics import counters


def _create_whimsy_types(feature_dims):
  """Creates a whimsy batch of zeros."""
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


def _create_input_spec_multiple_inputs_outputs():
  return collections.OrderedDict(
      x=[
          tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
          tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
      ],
      y=[
          tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
          tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
          tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
      ])


def _create_test_batch(feature_dims):
  return collections.OrderedDict(
      x=np.stack([
          np.zeros(feature_dims, np.float32),
          np.ones(feature_dims, np.float32)
      ]),
      y=np.stack([
          np.zeros([1], np.float32),
          np.ones([1], np.float32),
      ]))


class KerasUtilsTest(test_case.TestCase, parameterized.TestCase):

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
          input_spec=_create_whimsy_types(1),
          loss=tf.keras.losses.MeanSquaredError())

  # Test class for batches using namedtuple.
  _make_test_batch = collections.namedtuple('TestBatch', ['x', 'y'])

  @parameterized.named_parameters(
      ('container',
       collections.OrderedDict(
           x=tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
           y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32))),
      ('container_fn',
       _make_test_batch(
           x=tf.TensorSpec(shape=[1, 1], dtype=tf.float32),
           y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32))),
      ('tff_struct_with_python_type',
       computation_types.StructWithPythonType(
           collections.OrderedDict(
               x=tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
               y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32)),
           container_type=collections.OrderedDict)))
  def test_input_spec_python_container(self, input_spec):
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims=1)
    tff_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.MeanSquaredError())
    self.assertIsInstance(tff_model, model_lib.Model)
    tf.nest.map_structure(lambda x: self.assertIsInstance(x, tf.TensorSpec),
                          tff_model.input_spec)

  def test_input_spec_struct(self):
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims=1)
    input_spec = computation_types.StructType(
        collections.OrderedDict(
            x=tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
            y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32)))
    tff_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.MeanSquaredError())
    self.assertIsInstance(tff_model, model_lib.Model)
    self.assertIsInstance(tff_model.input_spec, collections.OrderedDict)
    tf.nest.map_structure(lambda x: self.assertIsInstance(x, tf.TensorSpec),
                          tff_model.input_spec)

  def test_input_spec_ragged_tensor(self):
    keras_model = model_examples.build_ragged_tensor_input_keras_model()
    input_spec = collections.OrderedDict(
        x=tf.RaggedTensorSpec(shape=[3, None], dtype=tf.int32),
        y=tf.TensorSpec(shape=[1], dtype=tf.bool))
    tff_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    self.assertIsInstance(tff_model, model_lib.Model)
    self.assertIsInstance(tff_model.input_spec['x'], tf.RaggedTensorSpec)

    batch = collections.OrderedDict(
        x=tf.ragged.constant([[1, 2, 3], [4], [5, 6]]),
        y=tf.constant([True, False, False]),
    )
    output = tff_model.forward_pass(batch)
    self.assertEqual(output.num_examples, 3)

  @parameterized.named_parameters(
      ('more_than_two_elements', [
          tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
          tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
          tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
      ]),
      ('dict_with_key_not_named_x',
       collections.OrderedDict(
           foo=tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
           y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32))),
      ('dict_with_key_not_named_y',
       collections.OrderedDict(
           x=tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
           bar=tf.TensorSpec(shape=[None, 1], dtype=tf.float32))),
  )
  def test_input_spec_batch_types_value_errors(self, input_spec):
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims=1)
    with self.assertRaises(ValueError):
      keras_utils.from_keras_model(
          keras_model=keras_model,
          input_spec=input_spec,
          loss=tf.keras.losses.MeanSquaredError())

  @parameterized.named_parameters(
      ('python_container_not_tensorspec',
       collections.OrderedDict(
           x=tf.constant(0.0, dtype=tf.float32),
           y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32)),
       'Expected input spec member to be of type.*TensorSpec'),
      ('tff_type_not_tensortype',
       computation_types.to_type(
           collections.OrderedDict(
               x=computation_types.SequenceType(
                   computation_types.TensorType(tf.float32)),
               y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32))),
       'Expected a `tff.Type` with all the leaf nodes being `tff.TensorType`s'))
  def test_input_spec_batch_types_type_errors(self, input_spec, error_message):
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims=1)
    with self.assertRaisesRegex(TypeError, error_message):
      keras_utils.from_keras_model(
          keras_model=keras_model,
          input_spec=input_spec,
          loss=tf.keras.losses.MeanSquaredError())

  @parameterized.named_parameters(
      # Test cases for the cartesian product of all parameter values.
      *_create_tff_model_from_keras_model_tuples())
  def test_tff_model_from_keras_model(self, feature_dims, model_fn):
    keras_model = model_fn(feature_dims)
    tff_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        input_spec=_create_whimsy_types(feature_dims),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()])
    self.assertIsInstance(tff_model, model_lib.Model)

    # Metrics should be zero, though the model wrapper internally executes the
    # forward pass once.
    self.assertSequenceEqual(tff_model.local_variables,
                             [0.0, 0.0, 0.0, 0.0, 0, 0])

    batch = _create_test_batch(feature_dims)
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
    self.assertEqual(output.loss, 0.5)
    metrics = tff_model.report_local_unfinalized_metrics()
    self.assertEqual(metrics['num_batches'], [1])
    self.assertEqual(metrics['num_examples'], [2])
    self.assertGreater(metrics['loss'][0], 0)
    self.assertEqual(metrics['loss'][1], 2)
    self.assertGreater(metrics['mean_absolute_error'][0], 0)
    self.assertEqual(metrics['mean_absolute_error'][1], 2)

    # TODO(b/202027329): Remove these checks when removing the two attributes:
    # `report_local_outptus` and `federated_output_computation`.
    # Ensure that `report_local_outptus` and `federated_output_computation`
    # raise a NotImplementedError.
    with self.assertRaisesRegex(NotImplementedError, 'Do not implement'):
      tff_model.report_local_outputs()
    with self.assertRaisesRegex(NotImplementedError, 'Do not implement'):
      tff_model.federated_output_computation()

  def test_tff_model_from_keras_model_regularization(self):
    keras_model = model_examples.build_linear_regression_ones_regularized_keras_sequential_model(
        3)
    tff_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        input_spec=_create_whimsy_types(3),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()])
    self.assertIsInstance(tff_model, model_lib.Model)

    # Metrics should be zero, though the model wrapper internally executes the
    # forward pass once.
    self.assertSequenceEqual(tff_model.local_variables,
                             [0.0, 0.0, 0.0, 0.0, 0, 0])

    batch = _create_test_batch(feature_dims=3)
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
    self.assertAlmostEqual(output.loss, 5.04)
    metrics = tff_model.report_local_unfinalized_metrics()
    self.assertEqual(metrics['num_batches'], [1])
    self.assertEqual(metrics['num_examples'], [2])
    self.assertGreater(metrics['loss'][0], 0)
    self.assertEqual(metrics['loss'][1], 2)
    self.assertGreater(metrics['mean_absolute_error'][0], 0)
    self.assertEqual(metrics['mean_absolute_error'][1], 2)

  @parameterized.named_parameters(*_create_tff_model_from_keras_model_tuples())
  def test_tff_model_from_keras_model_input_spec(self, feature_dims, model_fn):
    keras_model = model_fn(feature_dims)
    tff_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
        input_spec=_create_whimsy_types(feature_dims))
    self.assertIsInstance(tff_model, model_lib.Model)

    # Metrics should be zero, though the model wrapper internally executes the
    # forward pass once.
    self.assertSequenceEqual(tff_model.local_variables,
                             [0.0, 0.0, 0.0, 0.0, 0, 0])

    batch = _create_test_batch(feature_dims)
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
    self.assertEqual(output.loss, 0.5)
    metrics = tff_model.report_local_unfinalized_metrics()
    self.assertEqual(metrics['num_batches'], [1])
    self.assertEqual(metrics['num_examples'], [2])
    self.assertGreater(metrics['loss'][0], 0)
    self.assertEqual(metrics['loss'][1], 2)
    self.assertGreater(metrics['mean_absolute_error'][0], 0)
    self.assertEqual(metrics['mean_absolute_error'][1], 2)

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
        tf.keras.layers.InputLayer(input_shape=(None,)),
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
    self.assertIsInstance(tff_model, model_lib.Model)
    self.assertEqual(tff_model.input_spec, input_spec)

    batch = _create_test_batch(feature_dims=5)
    output = tff_model.forward_pass(batch)

    self.assertAllEqual(output.predictions.shape, [2, 1])

    # A batch with different sequence length should be processed in a similar
    # way
    batch = _create_test_batch(feature_dims=10)
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
        metrics=[tf.keras.metrics.MeanAbsoluteError()])

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
      batch_output = tff_model.forward_pass(batch)
      self.assertGreater(batch_output.loss, 0.0)

    m = tff_model.report_local_unfinalized_metrics()
    self.assertEqual(m['num_batches'], [num_train_steps])
    self.assertEqual(m['num_examples'], [input_vocab_size * num_train_steps])
    self.assertGreater(m['loss'][0], 0.0)
    self.assertEqual(m['loss'][1], input_vocab_size * num_train_steps)
    self.assertGreater(m['mean_absolute_error'][0], 0)
    self.assertEqual(m['mean_absolute_error'][1], 300)

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
        metrics=[tf.keras.metrics.MeanAbsoluteError()])

    batch_size = 2
    real_batch = collections.OrderedDict(
        x=collections.OrderedDict(
            a=np.ones(shape=[batch_size, 1], dtype=np.float32),
            b=np.ones(shape=[batch_size, 1], dtype=np.float32)),
        y=np.asarray([[2.0], [2.0]]).astype(np.float32))

    num_train_steps = 2
    for _ in range(num_train_steps):
      tff_model.forward_pass(real_batch)

    m = tff_model.report_local_unfinalized_metrics()
    self.assertEqual(m['num_batches'], [num_train_steps])
    self.assertEqual(m['num_examples'], [batch_size * num_train_steps])
    self.assertGreater(m['loss'][0], 0.0)
    self.assertEqual(m['loss'][1], batch_size * num_train_steps)
    self.assertGreater(m['mean_absolute_error'][0], 0)
    self.assertEqual(m['mean_absolute_error'][1], 4)

    # Ensure we can assign the FL trained model weights to a new model.
    tff_weights = model_utils.ModelWeights.from_model(tff_model)
    keras_model = model_examples.build_multiple_inputs_keras_model()
    tff_weights.assign_weights_to(keras_model)
    loaded_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()])

    orig_model_output = tff_model.forward_pass(real_batch)
    loaded_model_output = loaded_model.forward_pass(real_batch)
    self.assertAlmostEqual(orig_model_output.loss, loaded_model_output.loss)

  def test_keras_model_using_batch_norm_gets_warning(self):
    model = model_examples.build_conv_batch_norm_keras_model()
    input_spec = collections.OrderedDict(
        x=tf.TensorSpec(shape=[None, 28 * 28], dtype=tf.float32),
        y=tf.TensorSpec(shape=[None, 1], dtype=tf.int64))

    with warnings.catch_warnings(record=True) as warning:
      warnings.simplefilter('always')
      # Build a `tff.learning.Model` from a `tf.keras.Model`
      tff_model = keras_utils.from_keras_model(
          keras_model=model,
          input_spec=input_spec,
          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
          metrics=[tf.keras.metrics.MeanAbsoluteError()])
      # Ensure we can get warning of Batch Normalization.
      self.assertLen(warning, 1)
      self.assertIsSubClass(warning[-1].category, UserWarning)
      self.assertRegex(str(warning[-1].message), 'Batch Normalization')

    batch_size = 2
    batch = collections.OrderedDict(
        x=np.random.uniform(low=0.0, high=1.0,
                            size=[batch_size, 28 * 28]).astype(np.float32),
        y=np.random.random_integers(low=0, high=9, size=[batch_size,
                                                         1]).astype(np.int64))

    num_train_steps = 2
    for _ in range(num_train_steps):
      tff_model.forward_pass(batch)

    m = tff_model.report_local_unfinalized_metrics()
    self.assertEqual(m['num_batches'], [num_train_steps])
    self.assertEqual(m['num_examples'], [batch_size * num_train_steps])
    self.assertGreater(m['loss'][0], 0.0)
    self.assertEqual(m['loss'][1], batch_size * num_train_steps)
    self.assertGreater(m['mean_absolute_error'][0], 0)
    self.assertEqual(m['mean_absolute_error'][1], 4)

    # Ensure we can assign the FL trained model weights to a new model.
    tff_weights = model_utils.ModelWeights.from_model(tff_model)
    keras_model = model_examples.build_conv_batch_norm_keras_model()
    tff_weights.assign_weights_to(keras_model)

    def assert_all_weights_close(keras_weights, tff_weights):
      for keras_w, tff_w in zip(keras_weights, tff_weights):
        self.assertAllClose(
            keras_w, tff_w, atol=1e-4, msg='Variable [{}]'.format(keras_w.name))

    assert_all_weights_close(keras_model.trainable_weights,
                             tff_weights.trainable)
    assert_all_weights_close(keras_model.non_trainable_weights,
                             tff_weights.non_trainable)

  def test_keras_model_aggregated_metrics(self):
    feature_dims = 3
    num_train_steps = 3

    def _make_keras_model():
      keras_model = model_examples.build_linear_regression_keras_functional_model(
          feature_dims)
      return keras_model

    def _model_fn():
      return keras_utils.from_keras_model(
          keras_model=_make_keras_model(),
          input_spec=_create_whimsy_types(feature_dims),
          loss=tf.keras.losses.MeanSquaredError(),
          metrics=[tf.keras.metrics.MeanAbsoluteError()])

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
        return (tff_model.report_local_unfinalized_metrics(),
                model_utils.ModelWeights.from_model(tff_model))

      return _train_loop()

    # Simulate 'CLIENT' local training.
    client_unfinalized_metrics, tff_weights = _train()

    # Simulate entering the 'SERVER' context.
    tf.keras.backend.clear_session()

    tff_model = _model_fn()
    metrics_aggregator = aggregator.sum_then_finalize
    unfinalized_metrics_type = type_conversions.type_from_tensors(
        tff_model.report_local_unfinalized_metrics())
    metrics_aggregation_computation = metrics_aggregator(
        tff_model.metric_finalizers(), unfinalized_metrics_type)
    aggregated_outputs = metrics_aggregation_computation(
        [client_unfinalized_metrics])
    self.assertEqual(aggregated_outputs['num_batches'], num_train_steps)
    self.assertEqual(aggregated_outputs['num_examples'], 2 * num_train_steps)
    self.assertGreater(aggregated_outputs['loss'], 0.0)
    self.assertGreater(aggregated_outputs['mean_absolute_error'], 0)

    keras_model = _make_keras_model()
    tff_weights.assign_weights_to(keras_model)

  def test_keras_model_metric_finalizers_work_with_report_local_unfinalized_metrics(
      self):
    feature_dims = 3
    tff_model = keras_utils.from_keras_model(
        keras_model=model_examples
        .build_linear_regression_keras_functional_model(feature_dims),
        input_spec=_create_whimsy_types(feature_dims),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            counters.NumBatchesCounter(),
            counters.NumExamplesCounter(),
            tf.keras.metrics.MeanAbsoluteError()
        ])

    batch_input = collections.OrderedDict(
        x=np.ones([2, feature_dims], dtype=np.float32),
        y=np.ones([2, 1], dtype=np.float32))
    tff_model.forward_pass(batch_input)
    local_unfinalized_metrics = tff_model.report_local_unfinalized_metrics()

    # Creating a TFF computation is needed because the `tf.function`-decorated
    # `metric_finalizers` will create `tf.Variable`s on the non-first call (and
    # hence, will throw an error if it is directly invoked).
    @computations.tf_computation(
        type_conversions.type_from_tensors(local_unfinalized_metrics))
    def finalizer_computation(unfinalized_metrics):
      finalized_metrics = collections.OrderedDict()
      for metric_name, finalizer in tff_model.metric_finalizers().items():
        finalized_metrics[metric_name] = finalizer(
            unfinalized_metrics[metric_name])
      return finalized_metrics

    finalized_metrics = finalizer_computation(local_unfinalized_metrics)
    self.assertDictEqual(
        collections.OrderedDict(
            # The model is initialized with zeros, so `loss` (MeanSquaredError)
            # and `mean_absolute_error` are both 1.0.
            num_batches=1,
            num_examples=2,
            mean_absolute_error=1.0,
            loss=1.0),
        finalized_metrics)

  @parameterized.named_parameters(
      ('container', _create_input_spec_multiple_inputs_outputs()),
      ('container_fn',
       _make_test_batch(
           x=_create_input_spec_multiple_inputs_outputs()['x'],
           y=_create_input_spec_multiple_inputs_outputs()['y'])),
      ('tff_struct_with_python_type',
       computation_types.StructWithPythonType(
           _create_input_spec_multiple_inputs_outputs(),
           container_type=collections.OrderedDict)),
      ('tff_struct_type',
       computation_types.StructType(
           _create_input_spec_multiple_inputs_outputs())),
  )
  def test_keras_model_multiple_outputs(self, input_spec):
    keras_model = model_examples.build_multiple_outputs_keras_model()

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

    with self.subTest('loss_as_dict_fails'):
      with self.assertRaises(TypeError):
        _ = keras_utils.from_keras_model(
            keras_model=keras_model,
            input_spec=input_spec,
            loss={
                'dense_5': tf.keras.losses.MeanSquaredError(),
                'dense_6': tf.keras.losses.MeanSquaredError(),
                'whimsy': tf.keras.losses.MeanSquaredError()
            })

    with self.subTest('loss_list_no_opt'):
      tff_model = keras_utils.from_keras_model(
          keras_model=keras_model,
          input_spec=input_spec,
          loss=[
              tf.keras.losses.MeanSquaredError(),
              tf.keras.losses.MeanSquaredError(),
              tf.keras.losses.MeanSquaredError()
          ])

      self.assertIsInstance(tff_model, model_lib.Model)
      example_batch = collections.OrderedDict(
          x=[
              np.zeros([1, 1], dtype=np.float32),
              np.zeros([1, 1], dtype=np.float32)
          ],
          y=[
              np.zeros([1, 1], dtype=np.float32),
              np.ones([1, 1], dtype=np.float32),
              np.ones([1, 1], dtype=np.float32)
          ])
      output = tff_model.forward_pass(example_batch)
      self.assertAllClose(output.loss, 2.0)

    class CustomLoss(tf.keras.losses.Loss):

      def __init__(self):
        super().__init__(name='custom_loss')

      def call(self, y_true, y_pred):
        loss = tf.constant(0.0)
        for label, prediction in zip(y_true, y_pred):
          loss += tf.keras.losses.MeanSquaredError()(label, prediction)
        return loss

    keras_model = model_examples.build_multiple_outputs_keras_model()
    with self.subTest('single_custom_loss_can_work_with_multiple_outputs'):
      tff_model = keras_utils.from_keras_model(
          keras_model=keras_model, input_spec=input_spec, loss=CustomLoss())

      output = tff_model.forward_pass(example_batch)
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

      output = tff_model.forward_pass(example_batch)
      self.assertAllClose(output.loss, 0.5)

      output = tff_model.forward_pass(example_batch)
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
                'whimsy': 0.4
            })

  @parameterized.named_parameters(
      ('container', _create_input_spec_multiple_inputs_outputs()),
      ('container_fn',
       _make_test_batch(
           x=_create_input_spec_multiple_inputs_outputs()['x'],
           y=_create_input_spec_multiple_inputs_outputs()['y'])),
      ('tff_struct_with_python_type',
       computation_types.StructWithPythonType(
           _create_input_spec_multiple_inputs_outputs(),
           container_type=collections.OrderedDict)),
      ('tff_struct_type',
       computation_types.StructType(
           _create_input_spec_multiple_inputs_outputs())),
  )
  def test_regularized_keras_model_multiple_outputs(self, input_spec):
    keras_model = model_examples.build_multiple_outputs_regularized_keras_model(
    )

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

      self.assertIsInstance(tff_model, model_lib.Model)
      example_batch = collections.OrderedDict(
          x=[
              np.zeros([1, 1], dtype=np.float32),
              np.zeros([1, 1], dtype=np.float32)
          ],
          y=[
              np.zeros([1, 1], dtype=np.float32),
              np.ones([1, 1], dtype=np.float32),
              np.ones([1, 1], dtype=np.float32)
          ])
      output = tff_model.forward_pass(example_batch)

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

      output = tff_model.forward_pass(example_batch)

      # Labels are (0, 1, 1), preds are (1, 1, 3).
      # Weighted MSE is 0.1 * 1**2 + 0.2 * 0**2 + 0.3 * 2**2 = 1.3.
      # Regularization loss is 0.11 as before, for a total loss of 1.41.
      self.assertAllClose(output.loss, 1.41)

      output = tff_model.forward_pass(example_batch)
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
                'whimsy': 0.4
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
        metrics=[tf.keras.metrics.MeanAbsoluteError()])

    batch_size = 3
    batch = collections.OrderedDict(
        x=tf.constant([['G'], ['B'], ['R']], dtype=tf.string),
        y=tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32))

    num_train_steps = 2
    for _ in range(num_train_steps):
      tff_model.forward_pass(batch)

    metrics = tff_model.report_local_unfinalized_metrics()
    self.assertEqual(metrics['num_batches'], [num_train_steps])
    self.assertEqual(metrics['num_examples'], [batch_size * num_train_steps])
    self.assertGreater(metrics['loss'][0], 0.0)
    self.assertEqual(metrics['loss'][1], batch_size * num_train_steps)
    self.assertGreater(metrics['mean_absolute_error'][0], 0)
    self.assertEqual(metrics['mean_absolute_error'][1], 6)

    # Ensure we can assign the FL trained model weights to a new model.
    tff_weights = model_utils.ModelWeights.from_model(tff_model)
    keras_model = model_examples.build_lookup_table_keras_model()
    tff_weights.assign_weights_to(keras_model)
    loaded_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()])

    orig_model_output = tff_model.forward_pass(batch)
    loaded_model_output = loaded_model.forward_pass(batch)
    self.assertAlmostEqual(orig_model_output.loss, loaded_model_output.loss)

  def test_keras_model_preprocessing(self):
    self.skipTest('b/171254807')
    model = model_examples.build_preprocessing_lookup_keras_model()
    input_spec = collections.OrderedDict(
        x=tf.TensorSpec(shape=[None, 1], dtype=tf.string),
        y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32))
    tff_model = keras_utils.from_keras_model(
        keras_model=model,
        input_spec=input_spec,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()])

    batch_size = 3
    batch = collections.OrderedDict(
        x=tf.constant([['A'], ['B'], ['A']], dtype=tf.string),
        y=tf.constant([[0], [1], [1]], dtype=tf.float32))

    num_train_steps = 2
    for _ in range(num_train_steps):
      tff_model.forward_pass(batch)

    metrics = tff_model.report_local_unfinalized_metrics()
    self.assertEqual(metrics['num_batches'], [num_train_steps])
    self.assertEqual(metrics['num_examples'], [batch_size * num_train_steps])
    self.assertGreater(metrics['loss'][0], 0.0)
    self.assertEqual(metrics['loss'][1], batch_size * num_train_steps)
    self.assertGreater(metrics['mean_absolute_error'][0], 0)
    self.assertEqual(metrics['mean_absolute_error'][1], 2)

    # Ensure we can assign the FL trained model weights to a new model.
    tff_weights = model_utils.ModelWeights.from_model(tff_model)
    keras_model = model_examples.build_lookup_table_keras_model()
    tff_weights.assign_weights_to(keras_model)
    loaded_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()])

    orig_model_output = tff_model.forward_pass(batch)
    loaded_model_output = loaded_model.forward_pass(batch)
    self.assertAlmostEqual(orig_model_output.loss, loaded_model_output.loss)

  def test_keras_model_fails_compiled(self):
    feature_dims = 3
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims)

    keras_model.compile(loss=tf.keras.losses.MeanSquaredError())

    with self.assertRaisesRegex(ValueError, 'compile'):
      keras_utils.from_keras_model(
          keras_model=keras_model,
          input_spec=_create_whimsy_types(feature_dims),
          loss=tf.keras.losses.MeanSquaredError(),
          metrics=[tf.keras.metrics.MeanAbsoluteError()])

  def test_custom_keras_metric_with_extra_init_args_raises(self):

    class CustomCounter(tf.keras.metrics.Sum):
      """A custom `tf.keras.metrics.Metric` with extra args in `__init__`."""

      def __init__(self, name='new_counter', arg1=0, dtype=tf.int64):
        super().__init__(name, dtype)
        self._arg1 = arg1

      def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(1, sample_weight)

    feature_dims = 3
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims)
    tff_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        input_spec=_create_whimsy_types(feature_dims),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[CustomCounter(arg1=1)])
    metrics_aggregator = aggregator.sum_then_finalize
    unfinalized_metrics_type = type_conversions.type_from_tensors(
        tff_model.report_local_unfinalized_metrics())

    with self.assertRaisesRegex(TypeError, 'extra arguments'):
      metrics_aggregator(tff_model.metric_finalizers(),
                         unfinalized_metrics_type)

  def test_custom_keras_metric_no_extra_init_args_builds(self):

    class CustomCounter(tf.keras.metrics.Sum):
      """A custom `tf.keras.metrics.Metric` without extra args in `__init__`."""

      def __init__(self, name='new_counter', arg1=0, dtype=tf.int64):
        super().__init__(name, dtype)
        self._arg1 = arg1

      def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(1, sample_weight)

      def get_config(self):
        config = super().get_config()
        config['arg1'] = self._arg1
        return config

    feature_dims = 3
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims)
    tff_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        input_spec=_create_whimsy_types(feature_dims),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[CustomCounter(arg1=1)])
    metrics_aggregator = aggregator.sum_then_finalize
    unfinalized_metrics_type = type_conversions.type_from_tensors(
        tff_model.report_local_unfinalized_metrics())
    federated_metrics_aggregation = metrics_aggregator(
        tff_model.metric_finalizers(), unfinalized_metrics_type)

    self.assertIsInstance(federated_metrics_aggregation,
                          computation_base.Computation)

  @parameterized.named_parameters(
      # Test cases for the cartesian product of all parameter values.
      *_create_tff_model_from_keras_model_tuples())
  def test_keras_model_with_metric_constructors(self, feature_dims, model_fn):
    keras_model = model_fn(feature_dims)
    tff_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        input_spec=_create_whimsy_types(feature_dims),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()])
    self.assertIsInstance(tff_model, model_lib.Model)

    # Metrics should be zero, though the model wrapper internally executes the
    # forward pass once.
    self.assertSequenceEqual(tff_model.local_variables,
                             [0.0, 0.0, 0.0, 0.0, 0, 0])

    batch = _create_test_batch(feature_dims)
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
    self.assertEqual(output.loss, 0.5)
    metrics = tff_model.report_local_unfinalized_metrics()
    self.assertEqual(metrics['num_batches'], [1])
    self.assertEqual(metrics['num_examples'], [2])
    self.assertGreater(metrics['loss'][0], 0)
    self.assertEqual(metrics['loss'][1], 2)
    self.assertGreater(metrics['mean_absolute_error'][0], 0)
    self.assertEqual(metrics['mean_absolute_error'][1], 2)

  @parameterized.named_parameters(
      # Test cases for the cartesian product of all parameter values.
      *_create_tff_model_from_keras_model_tuples())
  def test_keras_model_without_input_metrics(self, feature_dims, model_fn):
    keras_model = model_fn(feature_dims)
    tff_model = keras_utils.from_keras_model(
        keras_model=keras_model,
        input_spec=_create_whimsy_types(feature_dims),
        loss=tf.keras.losses.MeanSquaredError())
    self.assertIsInstance(tff_model, model_lib.Model)

    # Metrics should be zero, though the model wrapper internally executes the
    # forward pass once.
    self.assertSequenceEqual(tff_model.local_variables, [0.0, 0.0, 0, 0])

    batch = _create_test_batch(feature_dims)
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
    self.assertEqual(output.loss, 0.5)
    metrics = tff_model.report_local_unfinalized_metrics()
    self.assertGreater(metrics['loss'][0], 0)
    self.assertEqual(metrics['loss'][1], 2)

  @parameterized.named_parameters(
      ('both_metrics_and_constructors',
       [counters.NumExamplesCounter,
        counters.NumBatchesCounter()], 'found both types'),
      ('non_callable', [tf.constant(1.0)], 'found a non-callable'),
      ('non_keras_metric_constructor', [tf.keras.losses.MeanSquaredError
                                       ], 'not a no-arg callable'))
  def test_keras_model_provided_invalid_metrics_raises(self, metrics,
                                                       error_message):
    feature_dims = 3
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims)

    with self.assertRaisesRegex(TypeError, error_message):
      keras_utils.from_keras_model(
          keras_model=keras_model,
          input_spec=_create_whimsy_types(feature_dims),
          loss=tf.keras.losses.MeanSquaredError(),
          metrics=metrics)

  # The metric names are senseical normally, but we just want to assert that
  # our explicit metrics override the defaults.
  @parameterized.named_parameters(
      ('num_examples', tf.keras.metrics.MeanSquaredError('num_examples')),
      ('num_batches', tf.keras.metrics.MeanSquaredError('num_batches')),
  )
  def test_custom_metrics_override_defaults(self, metric):
    feature_dims = 3
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims)

    model = keras_utils.from_keras_model(
        keras_model=keras_model,
        input_spec=_create_whimsy_types(feature_dims),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[metric])

    # By default the metrics have a single sum of count, but the test metrics
    # we add above have two values because they are a mean.
    self.assertLen(model.report_local_unfinalized_metrics()[metric.name], 2)


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  test_case.main()
