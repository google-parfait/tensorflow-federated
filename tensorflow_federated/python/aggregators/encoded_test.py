# Copyright 2020, The TensorFlow Federated Authors.
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
import random

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.aggregators import encoded
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te


def _tff_spec_to_encoder(encoder, tff_type):
  assert tff_type.is_tensor()
  return te.encoders.as_gather_encoder(
      encoder, tf.TensorSpec(tff_type.shape, tff_type.dtype))


def _identity_encoder_fn(value_spec):
  return te.encoders.as_gather_encoder(te.encoders.identity(), value_spec)


def _uniform_encoder_fn(value_spec):
  return te.encoders.as_gather_encoder(
      te.encoders.uniform_quantization(8), value_spec)


def _hadamard_encoder_fn(value_spec):
  return te.encoders.as_gather_encoder(
      te.encoders.hadamard_quantization(8), value_spec)


def _one_over_n_encoder_fn(value_spec):
  return te.encoders.as_gather_encoder(
      te.core.EncoderComposer(te.testing.PlusOneOverNEncodingStage()).make(),
      value_spec)


def _state_update_encoder_fn(value_spec):
  return te.encoders.as_gather_encoder(
      te.core.EncoderComposer(StateUpdateTensorsEncodingStage()).make(),
      value_spec)


_test_struct_type = computation_types.to_type(((tf.float32, (20,)), tf.float32))


class EncodedSumFactoryComputationTest(test_case.TestCase,
                                       parameterized.TestCase):

  @parameterized.named_parameters(
      ('identity_from_encoder_fn', _identity_encoder_fn),
      ('uniform_from_encoder_fn', _uniform_encoder_fn),
      ('hadamard_from_encoder_fn', _hadamard_encoder_fn),
      ('one_over_n_from_encoder_fn', _one_over_n_encoder_fn),
      ('state_update_from_encoder_fn', _state_update_encoder_fn),
  )
  def test_type_properties(self, encoder_fn):
    encoded_f = encoded.EncodedSumFactory(encoder_fn)
    self.assertIsInstance(encoded_f, factory.UnweightedAggregationFactory)

    process = encoded_f.create(_test_struct_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    self.assertIsNone(process.initialize.type_signature.parameter)
    server_state_type = process.initialize.type_signature.result
    # State structure should have one element per tensor aggregated,
    self.assertLen(server_state_type.member, 2)
    self.assertEqual(placements.SERVER, server_state_type.placement)

    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.at_clients(_test_struct_type)),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.at_server(_test_struct_type),
            measurements=computation_types.at_server(())))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))

  def test_encoder_fn_not_callable_raises(self):
    encoder = te.encoders.as_gather_encoder(te.encoders.identity(),
                                            tf.TensorSpec((), tf.float32))
    with self.assertRaises(TypeError):
      encoded.EncodedSumFactory(encoder)

  def test_quantize_above_threshold_negative_threshold_raises(self):
    with self.assertRaises(ValueError):
      encoded.EncodedSumFactory.quantize_above_threshold(
          quantization_bits=8, threshold=-1)

  @parameterized.named_parameters(
      ('zero', 0),
      ('negative', -1),
      ('too_large', 17),
  )
  def test_quantize_above_threshold_quantization_bits_raises(
      self, quantization_bits):
    with self.assertRaises(ValueError):
      encoded.EncodedSumFactory.quantize_above_threshold(
          quantization_bits=quantization_bits, threshold=10000)


class EncodedSumFactoryExecutionTest(test_case.TestCase):

  def test_simple_sum(self):
    encoded_f = encoded.EncodedSumFactory(_identity_encoder_fn)
    process = encoded_f.create(computation_types.to_type(tf.float32))

    state = process.initialize()

    client_data = [1.0, 2.0, 3.0]
    for _ in range(3):
      output = process.next(state, client_data)
      self.assertAllClose(6.0, output.result)
      self.assertEqual((), output.measurements)
      state = output.state

  def test_structure_sum(self):
    encoded_f = encoded.EncodedSumFactory(_identity_encoder_fn)
    process = encoded_f.create(
        computation_types.to_type(((tf.float32, (2,)), tf.float32)))

    state = process.initialize()

    client_data = [
        [[1.0, -1.0], 2],
        [[2.0, 4.0], 3],
        [[3.0, 5.0], 5],
    ]
    for _ in range(3):
      output = process.next(state, client_data)
      self.assertAllClose([[6.0, 8.0], 10], output.result)
      self.assertEqual((), output.measurements)
      state = output.state

  def test_quantize_above_threshold_zero(self):
    encoded_f = encoded.EncodedSumFactory.quantize_above_threshold(
        quantization_bits=1, threshold=0)
    test_type = computation_types.to_type([(tf.float32, (3,)),
                                           (tf.float32, (5,))])
    process = encoded_f.create(test_type)

    single_client_data = [[[0.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0]]]
    state = process.initialize()
    output = process.next(state, single_client_data)
    # Both tensors are quantized to their min and max, independently.
    self.assertSetEqual(set([0.0, 2.0]), set(output.result[0]))
    self.assertSetEqual(set([1.0, 5.0]), set(output.result[1]))

  def test_quantize_above_threshold_positive(self):
    encoded_f = encoded.EncodedSumFactory.quantize_above_threshold(
        quantization_bits=1, threshold=4)
    test_type = computation_types.to_type([(tf.float32, (3,)),
                                           (tf.float32, (5,))])
    process = encoded_f.create(test_type)

    single_client_data = [[[0.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0]]]
    state = process.initialize()
    output = process.next(state, single_client_data)
    # The first tensor is not quantized.
    self.assertAllClose([0.0, 1.0, 2.0], output.result[0])
    # The second tensor is quantized to its min and max.
    self.assertSetEqual(set([1.0, 5.0]), set(output.result[1]))

  def test_quantize_above_threshold(self):
    encoded_f = encoded.EncodedSumFactory.quantize_above_threshold(
        quantization_bits=4, threshold=0)
    process = encoded_f.create(
        computation_types.to_type((tf.float32, (10000,))))

    # Creates random values in range [0., 15.] plus the bondaries exactly.
    # After randomized quantization, 16 unique values should be present.
    single_client_data = [[random.uniform(0.0, 15.0) for _ in range(9998)] +
                          [0.0, 15.0]]
    state = process.initialize()
    output = process.next(state, single_client_data)
    unique_values = sorted(list(set(output.result)))
    self.assertAllClose([float(i) for i in range(16)], unique_values)


@te.core.tf_style_adaptive_encoding_stage
class StateUpdateTensorsEncodingStage(te.core.AdaptiveEncodingStageInterface):
  """Test encoding stage using supported state aggregation modes.

  This implementation does not use `encoding_stage.StateAggregationMode.STACK`
  which is currently not supported by the implementation.
  """

  ENCODED_VALUES_KEY = 'state_update_tensors_identity'
  SUM_STATE_UPDATE_KEY = 'state_update_tensors_update_sum'
  MIN_STATE_UPDATE_KEY = 'state_update_tensors_update_min'
  MAX_STATE_UPDATE_KEY = 'state_update_tensors_update_max'
  LAST_SUM_STATE_KEY = 'state_update_tensors_state_sum'
  LAST_MIN_STATE_KEY = 'state_update_tensors_state_min'
  LAST_MAX_STATE_KEY = 'state_update_tensors_state_max'

  @property
  def name(self):
    """See base class."""
    return 'state_update_tensors'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [self.ENCODED_VALUES_KEY]

  @property
  def commutes_with_sum(self):
    """See base class."""
    return True

  @property
  def decode_needs_input_shape(self):
    """See base class."""
    return False

  @property
  def state_update_aggregation_modes(self):
    """See base class."""
    return {
        self.SUM_STATE_UPDATE_KEY: te.core.StateAggregationMode.SUM,
        self.MIN_STATE_UPDATE_KEY: te.core.StateAggregationMode.MIN,
        self.MAX_STATE_UPDATE_KEY: te.core.StateAggregationMode.MAX,
    }

  def initial_state(self):
    """See base class."""
    return {
        self.LAST_SUM_STATE_KEY: tf.constant(0.0),
        self.LAST_MIN_STATE_KEY: tf.constant(0.0),
        self.LAST_MAX_STATE_KEY: tf.constant(0.0),
    }

  def update_state(self, state, state_update_tensors):
    """See base class."""
    del state  # Unused.
    return {
        self.LAST_SUM_STATE_KEY:
            tf.reduce_sum(state_update_tensors[self.SUM_STATE_UPDATE_KEY]),
        self.LAST_MIN_STATE_KEY:
            tf.reduce_min(state_update_tensors[self.MIN_STATE_UPDATE_KEY]),
        self.LAST_MAX_STATE_KEY:
            tf.reduce_max(state_update_tensors[self.MAX_STATE_UPDATE_KEY])
    }

  def get_params(self, state):
    """See base class."""
    del state  # Unused.
    return {}, {}

  def encode(self, x, encode_params):
    """See base class."""
    del encode_params  # Unused.
    x = tf.identity(x)
    return {
        self.ENCODED_VALUES_KEY: x
    }, {
        self.SUM_STATE_UPDATE_KEY: tf.reduce_sum(x),
        self.MIN_STATE_UPDATE_KEY: tf.reduce_min(x),
        self.MAX_STATE_UPDATE_KEY: tf.reduce_max(x),
    }

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del decode_params, num_summands, shape  # Unused.
    return tf.identity(encoded_tensors[self.ENCODED_VALUES_KEY])


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test_case.main()
