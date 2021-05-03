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
from tensorflow_federated.python.learning import federated_evaluation
from tensorflow_federated.python.learning import keras_utils
from tensorflow_federated.python.learning import model
from tensorflow_federated.python.learning import model_utils
from tensorflow_federated.python.learning.framework import dataset_reduce
from tensorflow_federated.python.learning.framework import encoding_utils
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te


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

  def forward_pass(self, batch, training=True):
    assert not training
    num_over = tf.reduce_sum(
        tf.cast(
            tf.greater(batch['temp'], self._variables.max_temp), tf.float32))
    self._variables.num_over.assign_add(num_over)
    loss = tf.constant(0.0)
    predictions = tf.zeros_like(batch['temp'])
    return collections.OrderedDict([
        (model.ForwardPassKeys.LOSS, loss),
        (model.ForwardPassKeys.PREDICTIONS, predictions),
        (model.ForwardPassKeys.NUM_EXAMPLES, tf.shape(predictions)[0])
    ])

  def report_local_outputs(self):
    return collections.OrderedDict(num_over=self._variables.num_over)

  @property
  def federated_output_computation(self):

    def aggregate_metrics(client_metrics):
      return collections.OrderedDict(
          num_over=intrinsics.federated_sum(client_metrics.num_over))

    return computations.federated_computation(aggregate_metrics)


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
    predictions = tf.zeros_like(batch['temp'])
    return model.BatchOutput(
        loss=loss,
        predictions=predictions,
        num_examples=tf.shape(predictions)[0])

  def report_local_outputs(self):
    return collections.OrderedDict(num_same=self._variables.num_same)

  @property
  def federated_output_computation(self):

    def aggregate_metrics(client_metrics):
      return collections.OrderedDict(
          num_same=intrinsics.federated_sum(client_metrics.num_same))

    return computations.federated_computation(aggregate_metrics)


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
          y=tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
      ),
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
  return_type = collections.OrderedDict(
      num_same=computation_types.at_server(tf.float32))
  return computation_types.FunctionType(
      parameter=collections.OrderedDict(
          server_model_weights=weights_parameter_type,
          federated_dataset=data_parameter_type),
      result=return_type)


class FederatedEvaluationTest(test_case.TestCase, parameterized.TestCase):

  @test_utils.skip_test_for_multi_gpu
  def test_federated_evaluation(self):
    evaluate = federated_evaluation.build_federated_evaluation(TestModel)
    self.assertEqual(
        str(evaluate.type_signature),
        '(<server_model_weights=<trainable=<float32>,non_trainable=<>>@SERVER,'
        'federated_dataset={<temp=float32[?]>*}@CLIENTS> -> '
        '<num_over=float32@SERVER>)')

    def _temp_dict(temps):
      return {'temp': np.array(temps, dtype=np.float32)}

    result = evaluate(
        collections.OrderedDict([
            ('trainable', [5.0]),
            ('non_trainable', []),
        ]), [
            [_temp_dict([1.0, 10.0, 2.0, 7.0]),
             _temp_dict([6.0, 11.0])],
            [_temp_dict([9.0, 12.0, 13.0])],
            [_temp_dict([1.0]), _temp_dict([22.0, 23.0])],
        ])
    self.assertEqual(result, collections.OrderedDict(num_over=9.0))

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
        collections.OrderedDict([
            ('trainable', [[5.0, 10.0, 5.0, 7.0]]),
            ('non_trainable', []),
        ]), [
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
    self.assertEqual(result, collections.OrderedDict(num_same=8.0))

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
        collections.OrderedDict([
            ('trainable', [[5.0, 10.0, 5.0, 7.0]]),
            ('non_trainable', []),
        ]), [
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
    self.assertLess(result['num_same'], 8.0)

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
      return collections.OrderedDict([
          ('x', np.reshape(np.array(temps, dtype=np.float32), (-1, 1))),
          ('y', np.reshape(np.array(temps, dtype=np.float32), (-1, 1))),
      ])

    result = evaluate_comp(
        initial_weights,
        [[_input_dict([1.0, 10.0, 2.0, 7.0]),
          _input_dict([6.0, 11.0])], [_input_dict([9.0, 12.0, 13.0])],
         [_input_dict([1.0]), _input_dict([22.0, 23.0])]])
    # Expect 100% accuracy and no loss because we've constructed the identity
    # function and have the same x's and y's for training data.
    self.assertDictEqual(result,
                         collections.OrderedDict(accuracy=1.0, loss=0.0))

  @parameterized.named_parameters(('non-simulation', False),
                                  ('simulation', True))
  @mock.patch.object(
      dataset_reduce,
      '_dataset_reduce_fn',
      wraps=dataset_reduce._dataset_reduce_fn)
  @test_utils.skip_test_for_multi_gpu
  def test_federated_evaluation_dataset_reduce(self, simulation, mock_method):
    evaluate_comp = federated_evaluation.build_federated_evaluation(
        _model_fn_from_keras, use_experimental_simulation_loop=simulation)
    initial_weights = tf.nest.map_structure(
        lambda x: x.read_value(),
        model_utils.ModelWeights.from_model(_model_fn_from_keras()))

    def _input_dict(temps):
      return collections.OrderedDict([
          ('x', np.reshape(np.array(temps, dtype=np.float32), (-1, 1))),
          ('y', np.reshape(np.array(temps, dtype=np.float32), (-1, 1))),
      ])

    evaluate_comp(
        initial_weights,
        [[_input_dict([1.0, 10.0, 2.0, 7.0]),
          _input_dict([6.0, 11.0])], [_input_dict([9.0, 12.0, 13.0])],
         [_input_dict([1.0]), _input_dict([22.0, 23.0])]])

    if simulation:
      mock_method.assert_not_called()
    else:
      mock_method.assert_called()

  def test_construction_calls_model_fn(self):
    # Assert that the the process building does not call `model_fn` too many
    # times. `model_fn` can potentially be expensive (loading weights,
    # processing, etc).
    mock_model_fn = mock.Mock(side_effect=TestModel)
    federated_evaluation.build_federated_evaluation(mock_model_fn)
    # TODO(b/186451541): reduce the number of calls to model_fn.
    self.assertEqual(mock_model_fn.call_count, 2)


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test_case.main()
