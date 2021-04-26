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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning.framework import encoding_utils
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te


_bad_encoder_named_parameters = [('float', 1.0), ('string', 'str'),
                                 ('object', object),
                                 ('encoder', te.encoders.identity())]


class EncodedBroadcastProcessTest(test_case.TestCase, parameterized.TestCase):
  """Tests for build_encoded_broadcast_process method."""

  @parameterized.named_parameters(
      ('tf_constant_identity', tf.constant, te.encoders.identity),
      ('tf_constant_uniform_quantization', tf.constant,
       lambda: te.encoders.uniform_quantization(8)),
      ('numpy_identity', lambda x: x, te.encoders.identity),
      ('numpy_uniform_quantization', lambda x: x,
       lambda: te.encoders.uniform_quantization(8)),
  )
  def test_build_encoded_broadcast_process(self, value_constructor,
                                           encoder_constructor):
    value = value_constructor(np.random.rand(20))
    value_spec = tf.TensorSpec(value.shape, tf.dtypes.as_dtype(value.dtype))
    value_type = computation_types.to_type(value_spec)
    encoder = te.encoders.as_simple_encoder(encoder_constructor(), value_spec)
    broadcast_process = encoding_utils.build_encoded_broadcast_process(
        value_type, encoder)
    state_type = broadcast_process._initialize_fn.type_signature.result
    broadcast_signature = broadcast_process._next_fn.type_signature

    self.assertIsInstance(broadcast_process, measured_process.MeasuredProcess)
    self.assertEqual(state_type, broadcast_signature.result[0])
    self.assertEqual(placements.SERVER, broadcast_signature.result[0].placement)
    self.assertEqual(value_type, broadcast_signature.result[1].member)
    self.assertEqual(placements.CLIENTS,
                     broadcast_signature.result[1].placement)

  @parameterized.named_parameters(*_bad_encoder_named_parameters)
  def test_build_encoded_broadcast_process_raises_bad_encoder(
      self, bad_encoder):
    value_type = computation_types.TensorType(tf.float32, shape=[2])
    with self.assertRaises(TypeError):
      encoding_utils.build_encoded_broadcast_process(value_type, bad_encoder)

  def test_build_encoded_broadcast_process_raises_incompatible_encoder(self):
    value_type = computation_types.TensorType(tf.float32, shape=[2])
    incompatible_encoder = te.encoders.as_simple_encoder(
        te.encoders.identity(), tf.TensorSpec((3,)))
    with self.assertRaises(TypeError):
      encoding_utils.build_encoded_broadcast_process(value_type,
                                                     incompatible_encoder)

  def test_build_encoded_broadcast_process_raises_bad_structure(self):
    value_type = computation_types.StructType([
        computation_types.TensorType(tf.float32, shape=[2]),
        computation_types.TensorType(tf.float32, shape=[2])
    ])
    encoder = te.encoders.as_simple_encoder(te.encoders.identity(),
                                            tf.TensorSpec((2,)))
    with self.assertRaises(ValueError):
      encoding_utils.build_encoded_broadcast_process(value_type, encoder)


class EncodingUtilsTest(test_case.TestCase, parameterized.TestCase):

  def test_broadcast_process_from_model(self):
    model_fn = model_examples.LinearRegression
    broadcast_process = (
        encoding_utils.build_encoded_broadcast_process_from_model(
            model_fn, _test_encoder_fn()))
    self.assertIsInstance(broadcast_process, measured_process.MeasuredProcess)

  @parameterized.named_parameters(
      ('identity', te.encoders.identity),
      ('uniform', lambda: te.encoders.uniform_quantization(8)),
      ('hadamard', lambda: te.encoders.hadamard_quantization(8)),
      (
          'one_over_n',
          lambda: te.core.EncoderComposer(  # pylint: disable=g-long-lambda
              te.testing.PlusOneOverNEncodingStage()).make()),
      (
          'state_update',
          lambda: te.core.EncoderComposer(  # pylint: disable=g-long-lambda
              StateUpdateTensorsEncodingStage()).make()),
  )
  def test_build_encode_decode_tf_computations_for_broadcast(
      self, encoder_constructor):
    value_spec = tf.TensorSpec((20,), tf.float32)
    encoder = te.encoders.as_simple_encoder(encoder_constructor(), value_spec)

    _, state_type = encoding_utils._build_initial_state_tf_computation(encoder)
    value_type = computation_types.to_type(value_spec)
    encode_fn, decode_fn = (
        encoding_utils._build_encode_decode_tf_computations_for_broadcast(
            state_type, value_type, encoder))

    self.assertEqual(state_type, encode_fn.type_signature.parameter[0])
    self.assertEqual(state_type, encode_fn.type_signature.result[0])
    # Output of encode should be the input to decode.
    self.assertEqual(encode_fn.type_signature.result[1],
                     decode_fn.type_signature.parameter)
    # Decode should return the same type as input to encode - value_type.
    self.assertEqual(value_type, encode_fn.type_signature.parameter[1])
    self.assertEqual(value_type, decode_fn.type_signature.result)

  @parameterized.named_parameters(
      ('identity', te.encoders.identity),
      ('uniform', lambda: te.encoders.uniform_quantization(8)),
      ('hadamard', lambda: te.encoders.hadamard_quantization(8)),
      (
          'one_over_n',
          lambda: te.core.EncoderComposer(  # pylint: disable=g-long-lambda
              te.testing.PlusOneOverNEncodingStage()).make()),
      (
          'state_update',
          lambda: te.core.EncoderComposer(  # pylint: disable=g-long-lambda
              StateUpdateTensorsEncodingStage()).make()),
  )
  def test_build_tf_computations_for_sum(self, encoder_constructor):
    # Tests that the partial computations have matching relevant input-output
    # signatures.
    value_spec = tf.TensorSpec((20,), tf.float32)
    encoder = te.encoders.as_gather_encoder(encoder_constructor(), value_spec)

    _, state_type = encoding_utils._build_initial_state_tf_computation(encoder)
    value_type = computation_types.to_type(value_spec)
    nest_encoder = encoding_utils._build_tf_computations_for_gather(
        state_type, value_type, encoder)

    self.assertEqual(state_type,
                     nest_encoder.get_params_fn.type_signature.parameter)
    encode_params_type = nest_encoder.get_params_fn.type_signature.result[0]
    decode_before_sum_params_type = nest_encoder.get_params_fn.type_signature.result[
        1]
    decode_after_sum_params_type = nest_encoder.get_params_fn.type_signature.result[
        2]

    self.assertEqual(value_type,
                     nest_encoder.encode_fn.type_signature.parameter[0])
    self.assertEqual(encode_params_type,
                     nest_encoder.encode_fn.type_signature.parameter[1])
    self.assertEqual(decode_before_sum_params_type,
                     nest_encoder.encode_fn.type_signature.parameter[2])
    state_update_tensors_type = nest_encoder.encode_fn.type_signature.result[2]

    accumulator_type = nest_encoder.zero_fn.type_signature.result
    self.assertEqual(state_update_tensors_type,
                     accumulator_type.state_update_tensors)

    self.assertEqual(accumulator_type,
                     nest_encoder.accumulate_fn.type_signature.parameter[0])
    self.assertEqual(nest_encoder.encode_fn.type_signature.result,
                     nest_encoder.accumulate_fn.type_signature.parameter[1])
    self.assertEqual(accumulator_type,
                     nest_encoder.accumulate_fn.type_signature.result)
    self.assertEqual(accumulator_type,
                     nest_encoder.merge_fn.type_signature.parameter[0])
    self.assertEqual(accumulator_type,
                     nest_encoder.merge_fn.type_signature.parameter[1])
    self.assertEqual(accumulator_type,
                     nest_encoder.merge_fn.type_signature.result)
    self.assertEqual(accumulator_type,
                     nest_encoder.report_fn.type_signature.parameter)
    self.assertEqual(accumulator_type,
                     nest_encoder.report_fn.type_signature.result)

    self.assertEqual(
        accumulator_type.values,
        nest_encoder.decode_after_sum_fn.type_signature.parameter[0])
    self.assertEqual(
        decode_after_sum_params_type,
        nest_encoder.decode_after_sum_fn.type_signature.parameter[1])
    self.assertEqual(value_type,
                     nest_encoder.decode_after_sum_fn.type_signature.result)

    self.assertEqual(state_type,
                     nest_encoder.update_state_fn.type_signature.parameter[0])
    self.assertEqual(state_update_tensors_type,
                     nest_encoder.update_state_fn.type_signature.parameter[1])
    self.assertEqual(state_type,
                     nest_encoder.update_state_fn.type_signature.result)


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


def _test_encoder_fn():
  """Returns an example mapping of tensor to encoder, determined by shape."""
  encoder_constructor = te.encoders.as_simple_encoder
  identity_encoder = te.encoders.identity()
  test_encoder = te.core.EncoderComposer(
      te.testing.PlusOneOverNEncodingStage()).make()

  def encoder_fn(tensor):
    if np.prod(tensor.shape) > 1:
      encoder = encoder_constructor(test_encoder,
                                    tf.TensorSpec(tensor.shape, tensor.dtype))
    else:
      encoder = encoder_constructor(identity_encoder,
                                    tf.TensorSpec(tensor.shape, tensor.dtype))
    return encoder

  return encoder_fn


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test_case.main()
