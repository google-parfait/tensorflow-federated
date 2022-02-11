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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import test_utils
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.core.test import static_assert
from tensorflow_federated.python.learning import federated_evaluation
from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import dataset_reduce
from tensorflow_federated.python.learning.framework import encoding_utils
from tensorflow_federated.python.learning.metrics import aggregator
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te

# Convenience aliases.
FederatedType = computation_types.FederatedType
FunctionType = computation_types.FunctionType
SequenceType = computation_types.SequenceType
StructType = computation_types.StructType
TensorType = computation_types.TensorType


class TestModel(model.Model):

  def __init__(self):
    self._variables = collections.namedtuple('Vars', 'max_temp num_over')(
        max_temp=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=[]),
            name='max_temp',
            trainable=True),
        num_over=tf.Variable(0.0, name='num_over', trainable=False))

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
        tf.cast(
            tf.greater(batch['temp'], self._variables.max_temp), tf.float32))
    self._variables.num_over.assign_add(num_over)
    loss = tf.constant(0.0)
    predictions = self.predict_on_batch(batch, training)
    return model.BatchOutput(
        loss=loss,
        predictions=predictions,
        num_examples=tf.shape(predictions)[0])

  @tf.function
  def report_local_outputs(self):
    raise NotImplementedError(
        'Do not implement. `report_local_outputs` and '
        '`federated_output_computation` are deprecated and will be removed '
        'in 2022Q1. You should use `report_local_unfinalized_metrics` and '
        '`metric_finalizers` instead. The cross-client metrics aggregation '
        'should be specified as the `metrics_aggregator` argument when you '
        'build a training process or evaluation computation using this model.')

  @property
  def federated_output_computation(self):
    raise NotImplementedError(
        'Do not implement. `report_local_outputs` and '
        '`federated_output_computation` are deprecated and will be removed '
        'in 2022Q1. You should use `report_local_unfinalized_metrics` and '
        '`metric_finalizers` instead. The cross-client metrics aggregation '
        'should be specified as the `metrics_aggregator` argument when you '
        'build a training process or evaluation computation using this model.')

  @tf.function
  def report_local_unfinalized_metrics(self):
    return collections.OrderedDict(num_over=self._variables.num_over)

  def metric_finalizers(self):
    return collections.OrderedDict(num_over=tf.function(func=lambda x: x))


class TestModelQuant(model.Model):
  """This model stores how much client data matches the input (num_same)."""

  def __init__(self):
    self._variables = collections.namedtuple('Vars', 'given_nums num_same')(
        given_nums=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(4,)),
            name='given_nums',
            trainable=True),
        num_same=tf.Variable(0.0, name='num_same', trainable=False))

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
        tf.cast(
            tf.equal(batch['temp'], self._variables.given_nums), tf.float32))
    self._variables.num_same.assign_add(num_same)
    # We're not actually training anything, so just use 0 loss and predictions.
    loss = tf.constant(0.0)
    predictions = self.predict_on_batch(batch, training)
    return model.BatchOutput(
        loss=loss,
        predictions=predictions,
        num_examples=tf.shape(predictions)[0])

  @tf.function
  def report_local_outputs(self):
    raise NotImplementedError(
        'Do not implement. `report_local_outputs` and '
        '`federated_output_computation` will be deprecated soon. Instead, '
        'you should use `report_local_unfinalized_metrics` and '
        '`metric_finalizers`. Cross-client metrics aggregation should be '
        'specified as the `metrics_aggregator` argument when you create a '
        'training process or evaluation computation using this model.')

  @property
  def federated_output_computation(self):
    raise NotImplementedError(
        'Do not implement. `report_local_outputs` and '
        '`federated_output_computation` will be deprecated soon. Instead, '
        'you should use `report_local_unfinalized_metrics` and '
        '`metric_finalizers`. Cross-client metrics aggregation should be '
        'specified as the `metrics_aggregator` argument when you create a '
        'training process or evaluation computation using this model.')

  @tf.function
  def report_local_unfinalized_metrics(self):
    return collections.OrderedDict(num_same=self._variables.num_same)

  def metric_finalizers(self):
    return collections.OrderedDict(num_same=tf.function(func=lambda x: x))


def _model_fn_from_keras():
  keras_model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(1,)),
      tf.keras.layers.Dense(
          1,
          kernel_initializer='ones',
          bias_initializer='zeros',
          activation=None)
  ], name='my_model')  # pyformat: disable
  # TODO(b/165666045): pyformat would create a big gap here
  return keras_utils.from_keras_model(
      keras_model,
      input_spec=collections.OrderedDict(
          x=tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
          y=tf.TensorSpec(shape=(None, 1), dtype=tf.float32)),
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.Accuracy()])


def _build_simple_quant_encoder(quantization_bits):
  """Returns a function to quantize an input tensor using quantization_bits."""

  def simple_quant_encoder(value: tf.Tensor):

    def quant_encoder(value: tf.Tensor):
      assert value.dtype in [tf.float32, tf.float64]
      return te.encoders.uniform_quantization(quantization_bits)

    spec = tf.TensorSpec(value.shape, value.dtype)
    return te.encoders.as_simple_encoder(quant_encoder(value), spec)

  return simple_quant_encoder


def _build_expected_broadcaster_next_signature():
  """Returns signature of the broadcaster used in multiple tests below."""
  state_type = computation_types.at_server(
      computation_types.StructType([('trainable', [
          (),
      ]), ('non_trainable', [])]))
  value_type = computation_types.at_server(
      model_utils.weights_type_from_model(TestModelQuant))
  result_type = computation_types.at_clients(
      model_utils.weights_type_from_model(TestModelQuant))
  measurements_type = computation_types.at_server(())
  return computation_types.FunctionType(
      parameter=collections.OrderedDict(state=state_type, value=value_type),
      result=collections.OrderedDict(
          state=state_type, result=result_type, measurements=measurements_type))


def _build_expected_test_quant_model_eval_signature():
  """Returns signature for build_federated_evaluation using TestModelQuant."""
  weights_parameter_type = computation_types.at_server(
      model_utils.weights_type_from_model(TestModelQuant))
  data_parameter_type = computation_types.at_clients(
      computation_types.SequenceType(
          collections.OrderedDict(
              temp=computation_types.TensorType(
                  shape=(None,), dtype=tf.float32))))
  return_type = computation_types.at_server(
      collections.OrderedDict(
          eval=collections.OrderedDict(num_same=tf.float32)))
  return computation_types.FunctionType(
      parameter=collections.OrderedDict(
          server_model_weights=weights_parameter_type,
          federated_dataset=data_parameter_type),
      result=return_type)


class FederatedEvaluationTest(test_case.TestCase, parameterized.TestCase):

  @test_utils.skip_test_for_multi_gpu
  def test_local_evaluation(self):
    model_weights_type = model_utils.weights_type_from_model(TestModel)
    batch_type = computation_types.to_type(TestModel().input_spec)
    client_evaluate = federated_evaluation.build_local_evaluation(
        TestModel, model_weights_type, batch_type)
    self.assert_types_equivalent(
        client_evaluate.type_signature,
        FunctionType(
            parameter=StructType([
                ('incoming_model_weights', model_weights_type),
                ('dataset',
                 SequenceType(
                     StructType([('temp',
                                  TensorType(dtype=tf.float32,
                                             shape=[None]))]))),
            ]),
            result=collections.OrderedDict(
                local_outputs=collections.OrderedDict(num_over=tf.float32),
                num_examples=tf.int64)))

    def _temp_dict(temps):
      return {'temp': np.array(temps, dtype=np.float32)}

    client_result = client_evaluate(
        collections.OrderedDict(trainable=[5.0], non_trainable=[]),
        [_temp_dict([1.0, 10.0, 2.0, 8.0]),
         _temp_dict([6.0, 11.0])])
    self.assertEqual(
        client_result,
        collections.OrderedDict(
            local_outputs=collections.OrderedDict(num_over=4.0),
            num_examples=6))

  @test_utils.skip_test_for_multi_gpu
  def test_federated_evaluation(self):
    evaluate = federated_evaluation.build_federated_evaluation(TestModel)
    model_weights_type = model_utils.weights_type_from_model(TestModel)
    self.assert_types_equivalent(
        evaluate.type_signature,
        FunctionType(
            parameter=StructType([
                ('server_model_weights',
                 computation_types.at_server(model_weights_type)),
                ('federated_dataset',
                 computation_types.at_clients(
                     SequenceType(
                         StructType([
                             ('temp',
                              TensorType(dtype=tf.float32, shape=[None]))
                         ])))),
            ]),
            result=computation_types.at_server(
                collections.OrderedDict(
                    eval=collections.OrderedDict(num_over=tf.float32)))))

    def _temp_dict(temps):
      return {'temp': np.array(temps, dtype=np.float32)}

    result = evaluate(
        collections.OrderedDict(trainable=[5.0], non_trainable=[]), [
            [_temp_dict([1.0, 10.0, 2.0, 7.0]),
             _temp_dict([6.0, 11.0])],
            [_temp_dict([9.0, 12.0, 13.0])],
            [_temp_dict([1.0]), _temp_dict([22.0, 23.0])],
        ])
    self.assertEqual(
        result,
        collections.OrderedDict(
            eval=collections.OrderedDict(num_over=9.0),
        ))

  @test_utils.skip_test_for_multi_gpu
  def test_federated_evaluation_quantized_conservatively(self):
    # Set up a uniform quantization encoder as the broadcaster.
    broadcaster = (
        encoding_utils.build_encoded_broadcast_process_from_model(
            TestModelQuant, _build_simple_quant_encoder(12)))
    self.assert_types_equivalent(broadcaster.next.type_signature,
                                 _build_expected_broadcaster_next_signature())
    evaluate = federated_evaluation.build_federated_evaluation(
        TestModelQuant, broadcast_process=broadcaster)
    # Confirm that the type signature matches what is expected.
    self.assert_types_identical(
        evaluate.type_signature,
        _build_expected_test_quant_model_eval_signature())

    def _temp_dict(temps):
      return {'temp': np.array(temps, dtype=np.float32)}

    result = evaluate(
        collections.OrderedDict(
            trainable=[[5.0, 10.0, 5.0, 7.0]], non_trainable=[]), [
                [
                    _temp_dict([1.0, 10.0, 2.0, 7.0]),
                    _temp_dict([6.0, 11.0, 5.0, 8.0])
                ],
                [_temp_dict([9.0, 12.0, 13.0, 7.0])],
                [
                    _temp_dict([1.0, 22.0, 23.0, 24.0]),
                    _temp_dict([5.0, 10.0, 5.0, 7.0])
                ],
            ])
    # This conservative quantization should not be too lossy.
    # When comparing the data examples to trainable, there are 8 times
    # where the index and value match.
    self.assertEqual(
        result,
        collections.OrderedDict(eval=collections.OrderedDict(num_same=8.0)))

  @test_utils.skip_test_for_multi_gpu
  def test_federated_evaluation_quantized_aggressively(self):
    # Set up a uniform quantization encoder as the broadcaster.
    broadcaster = (
        encoding_utils.build_encoded_broadcast_process_from_model(
            TestModelQuant, _build_simple_quant_encoder(2)))
    self.assert_types_equivalent(broadcaster.next.type_signature,
                                 _build_expected_broadcaster_next_signature())
    evaluate = federated_evaluation.build_federated_evaluation(
        TestModelQuant, broadcast_process=broadcaster)
    # Confirm that the type signature matches what is expected.
    self.assert_types_identical(
        evaluate.type_signature,
        _build_expected_test_quant_model_eval_signature())

    def _temp_dict(temps):
      return {'temp': np.array(temps, dtype=np.float32)}

    result = evaluate(
        collections.OrderedDict(
            trainable=[[5.0, 10.0, 5.0, 7.0]], non_trainable=[]), [
                [
                    _temp_dict([1.0, 10.0, 2.0, 7.0]),
                    _temp_dict([6.0, 11.0, 5.0, 8.0])
                ],
                [_temp_dict([9.0, 12.0, 13.0, 7.0])],
                [
                    _temp_dict([1.0, 22.0, 23.0, 24.0]),
                    _temp_dict([5.0, 10.0, 5.0, 7.0])
                ],
            ])
    # This very aggressive quantization should be so lossy that some of the
    # data is changed during encoding so the number that are equal between
    # the original and the final result should not be 8 as it is in the
    # conservative quantization test above.
    self.assertEqual(list(result.keys()), ['eval'])
    self.assertContainsSubset(result['eval'].keys(), ['num_same'])
    self.assertLess(result['eval']['num_same'], 8.0)

  @test_utils.skip_test_for_multi_gpu
  def test_federated_evaluation_fails_stateful_broadcast(self):
    # Create a test stateful measured process that doesn't do anything useful.

    @computations.federated_computation
    def init_fn():
      return intrinsics.federated_eval(
          computations.tf_computation(
              lambda: tf.zeros(shape=[], dtype=tf.float32)), placements.SERVER)

    @computations.federated_computation(
        computation_types.at_server(tf.float32),
        computation_types.at_clients(tf.int32))
    def next_fn(state, value):
      return measured_process.MeasuredProcessOutput(state, value, state)

    broadcaster = measured_process.MeasuredProcess(init_fn, next_fn)
    with self.assertRaisesRegex(ValueError, 'stateful broadcast'):
      federated_evaluation.build_federated_evaluation(
          TestModelQuant, broadcast_process=broadcaster)

  @test_utils.skip_test_for_multi_gpu
  def test_federated_evaluation_fails_non_measured_process_broadcast(self):
    broadcaster = computations.tf_computation(lambda x: x)
    with self.assertRaisesRegex(ValueError, '`MeasuredProcess`'):
      federated_evaluation.build_federated_evaluation(
          TestModelQuant, broadcast_process=broadcaster)

  @parameterized.named_parameters(('non-simulation', False),
                                  ('simulation', True))
  @test_utils.skip_test_for_multi_gpu
  def test_federated_evaluation_with_keras(self, simulation):

    evaluate_comp = federated_evaluation.build_federated_evaluation(
        _model_fn_from_keras, use_experimental_simulation_loop=simulation)
    initial_weights = tf.nest.map_structure(
        lambda x: x.read_value(),
        model_utils.ModelWeights.from_model(_model_fn_from_keras()))

    def _input_dict(temps):
      return collections.OrderedDict(
          x=np.reshape(np.array(temps, dtype=np.float32), (-1, 1)),
          y=np.reshape(np.array(temps, dtype=np.float32), (-1, 1)))

    result = evaluate_comp(
        initial_weights,
        [[_input_dict([1.0, 10.0, 2.0, 7.0]),
          _input_dict([6.0, 11.0])], [_input_dict([9.0, 12.0, 13.0])],
         [_input_dict([1.0]), _input_dict([22.0, 23.0])]])
    # Expect 100% accuracy and no loss because we've constructed the identity
    # function and have the same x's and y's for training data.
    self.assertDictEqual(
        result,
        collections.OrderedDict(
            eval=collections.OrderedDict(
                accuracy=1.0, loss=0.0, num_examples=12, num_batches=5)))

  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn)
  @test_utils.skip_test_for_multi_gpu
  def test_federated_evaluation_dataset_reduce(self, mock_method):
    evaluate_comp = federated_evaluation.build_federated_evaluation(
        _model_fn_from_keras, use_experimental_simulation_loop=False)
    initial_weights = tf.nest.map_structure(
        lambda x: x.read_value(),
        model_utils.ModelWeights.from_model(_model_fn_from_keras()))

    def _input_dict(temps):
      return collections.OrderedDict(
          x=np.reshape(np.array(temps, dtype=np.float32), (-1, 1)),
          y=np.reshape(np.array(temps, dtype=np.float32), (-1, 1)))

    evaluate_comp(
        initial_weights,
        [[_input_dict([1.0, 10.0, 2.0, 7.0]),
          _input_dict([6.0, 11.0])], [_input_dict([9.0, 12.0, 13.0])],
         [_input_dict([1.0]), _input_dict([22.0, 23.0])]])

    mock_method.assert_called()

  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn)
  @test_utils.skip_test_for_gpu
  def test_federated_evaluation_simulation_loop(self, mock_method):
    evaluate_comp = federated_evaluation.build_federated_evaluation(
        _model_fn_from_keras, use_experimental_simulation_loop=True)
    initial_weights = tf.nest.map_structure(
        lambda x: x.read_value(),
        model_utils.ModelWeights.from_model(_model_fn_from_keras()))

    def _input_dict(temps):
      return collections.OrderedDict(
          x=np.reshape(np.array(temps, dtype=np.float32), (-1, 1)),
          y=np.reshape(np.array(temps, dtype=np.float32), (-1, 1)))

    evaluate_comp(
        initial_weights,
        [[_input_dict([1.0, 10.0, 2.0, 7.0]),
          _input_dict([6.0, 11.0])]])

    mock_method.assert_not_called()

  def test_construction_calls_model_fn(self):
    # Assert that the the process building does not call `model_fn` too many
    # times. `model_fn` can potentially be expensive (loading weights,
    # processing, etc).
    mock_model_fn = mock.Mock(side_effect=TestModel)
    federated_evaluation.build_federated_evaluation(mock_model_fn)
    # TODO(b/186451541): reduce the number of calls to model_fn.
    self.assertEqual(mock_model_fn.call_count, 2)

  def test_no_unsecure_aggregation_with_secure_metrics_finalizer(self):
    evaluate_comp = federated_evaluation.build_federated_evaluation(
        _model_fn_from_keras,
        metrics_aggregator=aggregator.secure_sum_then_finalize)
    static_assert.assert_not_contains_unsecure_aggregation(evaluate_comp)


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  test_case.main()
