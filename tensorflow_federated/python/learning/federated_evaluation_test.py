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

import collections
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_test_utils
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.core.test import static_assert
from tensorflow_federated.python.learning import dataset_reduce
from tensorflow_federated.python.learning import federated_evaluation
from tensorflow_federated.python.learning.metrics import aggregator
from tensorflow_federated.python.learning.metrics import counters
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.models import keras_utils
from tensorflow_federated.python.learning.models import model_examples
from tensorflow_federated.python.learning.models import model_weights
from tensorflow_federated.python.learning.models import test_models
from tensorflow_federated.python.learning.models import variable
from tensorflow_federated.python.tensorflow_libs import tensorflow_test_utils


# Convenience aliases.
FederatedType = computation_types.FederatedType
FunctionType = computation_types.FunctionType
SequenceType = computation_types.SequenceType
StructType = computation_types.StructType
TensorType = computation_types.TensorType


class TestModel(variable.VariableModel):

  def __init__(self):
    self._variables = collections.namedtuple('Vars', 'max_temp num_over')(
        max_temp=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=[]),
            name='max_temp',
            trainable=True,
        ),
        num_over=tf.Variable(0.0, name='num_over', trainable=False),
    )

  @property
  def trainable_variables(self):
    return [self._variables.max_temp]

  @property
  def non_trainable_variables(self):
    return []

  @property
  def local_variables(self):
    return [self._variables.num_over]

  @property
  def input_spec(self):
    return collections.OrderedDict(temp=tf.TensorSpec([None], tf.float32))

  @tf.function
  def predict_on_batch(self, batch, training=True):
    del training  # Unused.
    return tf.zeros_like(batch['temp'])

  @tf.function
  def forward_pass(self, batch, training=True):
    assert not training
    num_over = tf.reduce_sum(
        tf.cast(tf.greater(batch['temp'], self._variables.max_temp), tf.float32)
    )
    self._variables.num_over.assign_add(num_over)
    loss = tf.constant(0.0)
    predictions = self.predict_on_batch(batch, training)
    return variable.BatchOutput(
        loss=loss,
        predictions=predictions,
        num_examples=tf.shape(predictions)[0],
    )

  @tf.function
  def report_local_unfinalized_metrics(self):
    return collections.OrderedDict(num_over=self._variables.num_over)

  def metric_finalizers(self):
    return collections.OrderedDict(num_over=tf.function(func=lambda x: x))

  @tf.function
  def reset_metrics(self):
    """Resets metrics variables to initial value."""
    for var in self.local_variables:
      var.assign(tf.zeros_like(var))


class TestModelQuant(variable.VariableModel):
  """This model stores how much client data matches the input (num_same)."""

  def __init__(self):
    self._variables = collections.namedtuple('Vars', 'given_nums num_same')(
        given_nums=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(4,)),
            name='given_nums',
            trainable=True,
        ),
        num_same=tf.Variable(0.0, name='num_same', trainable=False),
    )

  @property
  def trainable_variables(self):
    return [self._variables.given_nums]

  @property
  def non_trainable_variables(self):
    return []

  @property
  def local_variables(self):
    return [self._variables.num_same]

  @property
  def input_spec(self):
    return collections.OrderedDict(temp=tf.TensorSpec([None], tf.float32))

  @tf.function
  def predict_on_batch(self, batch, training=True):
    del training  # Unused.
    return tf.zeros_like(batch['temp'])

  @tf.function
  def forward_pass(self, batch, training=True):
    """Unlike the TestModel implementation above, only tracks num_same."""
    assert not training
    # Calculate how many of the values in the training data match the input.
    num_same = tf.reduce_sum(
        tf.cast(tf.equal(batch['temp'], self._variables.given_nums), tf.float32)
    )
    self._variables.num_same.assign_add(num_same)
    # We're not actually training anything, so just use 0 loss and predictions.
    loss = tf.constant(0.0)
    predictions = self.predict_on_batch(batch, training)
    return variable.BatchOutput(
        loss=loss,
        predictions=predictions,
        num_examples=tf.shape(predictions)[0],
    )

  @tf.function
  def report_local_unfinalized_metrics(self):
    return collections.OrderedDict(num_same=self._variables.num_same)

  def metric_finalizers(self):
    return collections.OrderedDict(num_same=tf.function(func=lambda x: x))

  @tf.function
  def reset_metrics(self):
    """Resets metrics variables to initial value."""
    for var in self.local_variables:
      var.assign(tf.zeros_like(var))


def _model_fn_from_keras():
  keras_model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(1,)),
      tf.keras.layers.Dense(
          1,
          kernel_initializer='ones',
          bias_initializer='zeros',
          activation=None)
  ], name='my_model')  # pyformat: disable
  # TODO: b/165666045 - pyformat would create a big gap here
  return keras_utils.from_keras_model(
      keras_model,
      input_spec=collections.OrderedDict(
          x=tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
          y=tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
      ),
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.Accuracy()],
  )


class FederatedEvaluationTest(parameterized.TestCase):

  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_local_evaluation(self):
    model_weights_type = model_weights.weights_type_from_model(TestModel)
    batch_type = computation_types.tensorflow_to_type(TestModel().input_spec)
    client_evaluate = federated_evaluation.build_local_evaluation(
        TestModel, model_weights_type, batch_type
    )
    type_test_utils.assert_types_equivalent(
        client_evaluate.type_signature,
        FunctionType(
            parameter=StructType([
                ('incoming_model_weights', model_weights_type),
                (
                    'dataset',
                    SequenceType(
                        StructType([(
                            'temp',
                            TensorType(dtype=np.float32, shape=[None]),
                        )])
                    ),
                ),
            ]),
            result=collections.OrderedDict(
                local_outputs=collections.OrderedDict(num_over=np.float32),
                num_examples=np.int64,
            ),
        ),
    )

    def _temp_dict(temps):
      return {'temp': np.array(temps, dtype=np.float32)}

    client_result = client_evaluate(
        collections.OrderedDict(trainable=[5.0], non_trainable=[]),
        [_temp_dict([1.0, 10.0, 2.0, 8.0]), _temp_dict([6.0, 11.0])],
    )
    self.assertEqual(
        client_result,
        collections.OrderedDict(
            local_outputs=collections.OrderedDict(num_over=4.0), num_examples=6
        ),
    )

  def test_federated_evaluation_deprecation_warning(self):
    with self.assertWarnsRegex(DeprecationWarning, 'build_fed_eval'):
      federated_evaluation.build_federated_evaluation(TestModel)

  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_federated_evaluation(self):
    evaluate = federated_evaluation.build_federated_evaluation(TestModel)
    model_weights_type = model_weights.weights_type_from_model(TestModel)
    type_test_utils.assert_types_equivalent(
        evaluate.type_signature,
        FunctionType(
            parameter=StructType([
                (
                    'server_model_weights',
                    computation_types.FederatedType(
                        model_weights_type, placements.SERVER
                    ),
                ),
                (
                    'federated_dataset',
                    computation_types.FederatedType(
                        SequenceType(
                            StructType([(
                                'temp',
                                TensorType(dtype=np.float32, shape=[None]),
                            )])
                        ),
                        placements.CLIENTS,
                    ),
                ),
            ]),
            result=computation_types.FederatedType(
                collections.OrderedDict(
                    eval=collections.OrderedDict(num_over=np.float32)
                ),
                placements.SERVER,
            ),
        ),
    )

    def _temp_dict(temps):
      return {'temp': np.array(temps, dtype=np.float32)}

    result = evaluate(
        collections.OrderedDict(trainable=[5.0], non_trainable=[]),
        [
            [_temp_dict([1.0, 10.0, 2.0, 7.0]), _temp_dict([6.0, 11.0])],
            [_temp_dict([9.0, 12.0, 13.0])],
            [_temp_dict([1.0]), _temp_dict([22.0, 23.0])],
        ],
    )
    self.assertEqual(
        result,
        collections.OrderedDict(eval=collections.OrderedDict(num_over=9.0)),
    )

  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_federated_evaluation_fails_stateful_broadcast(self):
    # Create a test stateful measured process that doesn't do anything useful.

    @federated_computation.federated_computation
    def init_fn():
      return intrinsics.federated_eval(
          tensorflow_computation.tf_computation(
              lambda: tf.zeros(shape=[], dtype=tf.float32)
          ),
          placements.SERVER,
      )

    @federated_computation.federated_computation(
        computation_types.FederatedType(np.float32, placements.SERVER),
        computation_types.FederatedType(np.int32, placements.CLIENTS),
    )
    def next_fn(state, value):
      return measured_process.MeasuredProcessOutput(state, value, state)

    broadcaster = measured_process.MeasuredProcess(init_fn, next_fn)
    with self.assertRaisesRegex(ValueError, 'stateful broadcast'):
      federated_evaluation.build_federated_evaluation(
          TestModelQuant, broadcast_process=broadcaster
      )

  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_federated_evaluation_fails_non_measured_process_broadcast(self):
    broadcaster = tensorflow_computation.tf_computation(lambda x: x)
    with self.assertRaisesRegex(ValueError, '`MeasuredProcess`'):
      federated_evaluation.build_federated_evaluation(
          TestModelQuant, broadcast_process=broadcaster
      )

  @parameterized.named_parameters(
      ('non-simulation', False), ('simulation', True)
  )
  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_federated_evaluation_with_keras(self, simulation):
    evaluate_comp = federated_evaluation.build_federated_evaluation(
        _model_fn_from_keras, use_experimental_simulation_loop=simulation
    )
    initial_weights = tf.nest.map_structure(
        lambda x: x.read_value(),
        model_weights.ModelWeights.from_model(_model_fn_from_keras()),
    )

    def _input_dict(temps):
      return collections.OrderedDict(
          x=np.reshape(np.array(temps, dtype=np.float32), (-1, 1)),
          y=np.reshape(np.array(temps, dtype=np.float32), (-1, 1)),
      )

    result = evaluate_comp(
        initial_weights,
        [
            [_input_dict([1.0, 10.0, 2.0, 7.0]), _input_dict([6.0, 11.0])],
            [_input_dict([9.0, 12.0, 13.0])],
            [_input_dict([1.0]), _input_dict([22.0, 23.0])],
        ],
    )
    # Expect 100% accuracy and no loss because we've constructed the identity
    # function and have the same x's and y's for training data.
    self.assertDictEqual(
        result,
        collections.OrderedDict(
            eval=collections.OrderedDict(
                accuracy=1.0, loss=0.0, num_examples=12, num_batches=5
            )
        ),
    )

  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn,
  )
  @tensorflow_test_utils.skip_test_for_multi_gpu
  def test_federated_evaluation_dataset_reduce(self, mock_method):
    evaluate_comp = federated_evaluation.build_federated_evaluation(
        _model_fn_from_keras, use_experimental_simulation_loop=False
    )
    initial_weights = tf.nest.map_structure(
        lambda x: x.read_value(),
        model_weights.ModelWeights.from_model(_model_fn_from_keras()),
    )

    def _input_dict(temps):
      return collections.OrderedDict(
          x=np.reshape(np.array(temps, dtype=np.float32), (-1, 1)),
          y=np.reshape(np.array(temps, dtype=np.float32), (-1, 1)),
      )

    evaluate_comp(
        initial_weights,
        [
            [_input_dict([1.0, 10.0, 2.0, 7.0]), _input_dict([6.0, 11.0])],
            [_input_dict([9.0, 12.0, 13.0])],
            [_input_dict([1.0]), _input_dict([22.0, 23.0])],
        ],
    )

    mock_method.assert_called()

  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn,
  )
  @tensorflow_test_utils.skip_test_for_gpu
  def test_federated_evaluation_simulation_loop(self, mock_method):
    evaluate_comp = federated_evaluation.build_federated_evaluation(
        _model_fn_from_keras, use_experimental_simulation_loop=True
    )
    initial_weights = tf.nest.map_structure(
        lambda x: x.read_value(),
        model_weights.ModelWeights.from_model(_model_fn_from_keras()),
    )

    def _input_dict(temps):
      return collections.OrderedDict(
          x=np.reshape(np.array(temps, dtype=np.float32), (-1, 1)),
          y=np.reshape(np.array(temps, dtype=np.float32), (-1, 1)),
      )

    evaluate_comp(
        initial_weights,
        [[_input_dict([1.0, 10.0, 2.0, 7.0]), _input_dict([6.0, 11.0])]],
    )

    mock_method.assert_not_called()

  def test_construction_calls_model_fn(self):
    # Assert that the the process building does not call `model_fn` too many
    # times. `model_fn` can potentially be expensive (loading weights,
    # processing, etc).
    mock_model_fn = mock.Mock(side_effect=TestModel)
    federated_evaluation.build_federated_evaluation(mock_model_fn)
    # TODO: b/186451541 - reduce the number of calls to model_fn.
    self.assertEqual(mock_model_fn.call_count, 2)

  def test_no_unsecure_aggregation_with_secure_metrics_finalizer(self):
    evaluate_comp = federated_evaluation.build_federated_evaluation(
        _model_fn_from_keras,
        metrics_aggregator=aggregator.secure_sum_then_finalize,
    )
    static_assert.assert_not_contains_unsecure_aggregation(evaluate_comp)


class FunctionalFederatedEvaluationTest(tf.test.TestCase):

  def create_test_datasets(self) -> tf.data.Dataset:
    dataset1 = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            y=[[0.0], [0.0], [1.0], [1.0]],
        )
    )
    dataset2 = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            y=[[1.0], [2.0], [3.0], [4.0]],
        )
    )
    return [dataset1.repeat(2).batch(3), dataset2.repeat(2).batch(3)]

  def test_raises_on_non_callable_or_functional_model(self):
    with self.assertRaisesRegex(TypeError, 'is not a callable'):
      federated_evaluation.build_federated_evaluation(model_fn=0)

  @tensorflow_test_utils.skip_test_for_gpu
  def test_functional_local_evaluation_matches_non_functional(self):
    dataset = self.create_test_datasets()[0]
    batch_type = computation_types.tensorflow_to_type(dataset.element_spec)
    loss_fn = tf.keras.losses.MeanSquaredError
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims=2
    )

    # Defining artifacts using `tff.learning.models.VariableModel`
    def tff_model_fn():
      keras_model = (
          model_examples.build_linear_regression_keras_functional_model(
              feature_dims=2
          )
      )
      return keras_utils.from_keras_model(
          keras_model, loss=loss_fn(), input_spec=batch_type
      )

    tff_model = tff_model_fn()
    model_weights_type = model_weights.weights_type_from_model(tff_model)
    eval_weights = model_weights.ModelWeights.from_model(tff_model)

    local_eval = federated_evaluation.build_local_evaluation(
        tff_model_fn, model_weights_type, batch_type
    )
    local_output = local_eval(eval_weights, dataset)

    # Defining artifacts using `tff.learning.models.FunctionalModel`
    def build_metrics_fn():
      return collections.OrderedDict(
          loss=tf.keras.metrics.MeanSquaredError(),
          num_examples=counters.NumExamplesCounter(),
          num_batches=counters.NumBatchesCounter(),
      )

    functional_model = functional.functional_model_from_keras(
        keras_model,
        loss_fn=loss_fn(),
        input_spec=batch_type,
        metrics_constructor=build_metrics_fn,
    )

    def ndarray_to_tensorspec(ndarray):
      return tf.TensorSpec(
          shape=ndarray.shape, dtype=tf.dtypes.as_dtype(ndarray.dtype)
      )

    weights_type = tf.nest.map_structure(
        ndarray_to_tensorspec, functional_model.initial_weights
    )
    functional_eval = federated_evaluation.build_functional_local_evaluation(
        functional_model, weights_type, batch_type
    )
    functional_output = functional_eval(
        functional_model.initial_weights, dataset
    )

    # Testing equality
    self.assertDictEqual(local_output['local_outputs'], functional_output)

  @tensorflow_test_utils.skip_test_for_gpu
  def test_functional_evaluation_matches_non_functional(self):
    datasets = self.create_test_datasets()
    batch_type = computation_types.tensorflow_to_type(datasets[0].element_spec)
    loss_fn = tf.keras.losses.MeanSquaredError
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims=2
    )
    metrics_aggregator = aggregator.sum_then_finalize

    # Defining artifacts using `tff.learning.models.VariableModel`
    def tff_model_fn():
      keras_model = (
          model_examples.build_linear_regression_keras_functional_model(
              feature_dims=2
          )
      )
      return keras_utils.from_keras_model(
          keras_model, loss=loss_fn(), input_spec=batch_type
      )

    federated_eval = federated_evaluation._build_federated_evaluation(
        model_fn=tff_model_fn,
        broadcast_process=None,
        metrics_aggregator=metrics_aggregator,
        use_experimental_simulation_loop=False,
    )
    tff_model = tff_model_fn()
    eval_weights = model_weights.ModelWeights.from_model(tff_model)
    eval_metrics = federated_eval(eval_weights, datasets)

    # Defining artifacts using `tff.learning.models.FunctionalModel`
    def build_metrics_fn():
      return collections.OrderedDict(
          loss=tf.keras.metrics.MeanSquaredError(),
          num_examples=counters.NumExamplesCounter(),
          num_batches=counters.NumBatchesCounter(),
      )

    functional_model = functional.functional_model_from_keras(
        keras_model,
        loss_fn=loss_fn(),
        input_spec=batch_type,
        metrics_constructor=build_metrics_fn,
    )
    functional_eval = (
        federated_evaluation._build_functional_federated_evaluation(
            model=functional_model,
            broadcast_process=None,
            metrics_aggregator=metrics_aggregator,
        )
    )
    functional_metrics = functional_eval(
        functional_model.initial_weights, datasets
    )
    self.assertDictEqual(eval_metrics, functional_metrics)

  def test_no_unsecure_aggregation_with_secure_metrics_finalizer(self):
    functional_model = test_models.build_functional_linear_regression()
    evaluate_comp = federated_evaluation.build_federated_evaluation(
        functional_model, metrics_aggregator=aggregator.secure_sum_then_finalize
    )
    static_assert.assert_not_contains_unsecure_aggregation(evaluate_comp)


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  absltest.main()
