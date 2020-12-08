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
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.aggregators import encoded_factory
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
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
    encoded_f = encoded_factory.EncodedSumFactory(encoder_fn)
    self.assertIsInstance(encoded_f, factory.UnweightedAggregationFactory)

    process = encoded_f.create_unweighted(_test_struct_type)
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
      encoded_factory.EncodedSumFactory(encoder)


class EncodedSumFactoryExecutionTest(test_case.TestCase):

  def test_simple_sum(self):
    encoded_f = encoded_factory.EncodedSumFactory(_identity_encoder_fn)
    process = encoded_f.create_unweighted(computation_types.to_type(tf.float32))

    state = process.initialize()

    client_data = [1.0, 2.0, 3.0]
    for _ in range(3):
      output = process.next(state, client_data)
      self.assertAllClose(6.0, output.result)
      self.assertEqual((), output.measurements)
      state = output.state

  def test_structure_sum(self):
    encoded_f = encoded_factory.EncodedSumFactory(_identity_encoder_fn)
    process = encoded_f.create_unweighted(
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
