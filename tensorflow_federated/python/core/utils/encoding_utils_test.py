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

import warnings

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates.measured_process import MeasuredProcess
from tensorflow_federated.python.core.utils import encoding_utils
from tensorflow_federated.python.core.utils.computation_utils import StatefulAggregateFn
from tensorflow_federated.python.core.utils.computation_utils import StatefulBroadcastFn
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te

_bad_encoder_named_parameters = [('float', 1.0), ('string', 'str'),
                                 ('object', object),
                                 ('encoder', te.encoders.identity())]


class EncodedBroadcastTest(test.TestCase, parameterized.TestCase):
  """Tests for build_encoded_broadcast method."""

  def test_build_encoded_broadcast_raise_warning(self):
    value = tf.constant(np.random.rand(20))
    value_spec = tf.TensorSpec(value.shape, tf.dtypes.as_dtype(value.dtype))
    encoder = te.encoders.as_simple_encoder(te.encoders.identity(), value_spec)

    with warnings.catch_warnings(record=True):
      warnings.simplefilter('error', DeprecationWarning)
      with self.assertRaisesRegex(DeprecationWarning,
                                  'tff.utils.build_encoded_broadcast()'):
        encoding_utils.build_encoded_broadcast(value, encoder)

  @parameterized.named_parameters(
      ('tf_constant_identity', tf.constant, te.encoders.identity),
      ('tf_constant_uniform_quantization', tf.constant,
       lambda: te.encoders.uniform_quantization(8)),
      ('numpy_identity', lambda x: x, te.encoders.identity),
      ('numpy_uniform_quantization', lambda x: x,
       lambda: te.encoders.uniform_quantization(8)),
  )
  def test_build_encoded_broadcast(self, value_constructor,
                                   encoder_constructor):
    value = value_constructor(np.random.rand(20))
    value_spec = tf.TensorSpec(value.shape, tf.dtypes.as_dtype(value.dtype))
    value_type = computation_types.to_type(value_spec)
    encoder = te.encoders.as_simple_encoder(encoder_constructor(), value_spec)
    broadcast_fn = encoding_utils.build_encoded_broadcast(value, encoder)
    state_type = broadcast_fn._initialize_fn.type_signature.result
    broadcast_signature = computations.federated_computation(
        broadcast_fn._next_fn,
        computation_types.FederatedType(state_type, placements.SERVER),
        computation_types.FederatedType(value_type,
                                        placements.SERVER)).type_signature

    self.assertIsInstance(broadcast_fn, StatefulBroadcastFn)
    self.assertEqual(state_type, broadcast_signature.result[0].member)
    self.assertEqual(placements.SERVER, broadcast_signature.result[0].placement)
    self.assertEqual(value_type, broadcast_signature.result[1].member)
    self.assertEqual(placements.CLIENTS,
                     broadcast_signature.result[1].placement)

  @parameterized.named_parameters(*_bad_encoder_named_parameters)
  def test_build_encoded_broadcast_raises_bad_encoder(self, bad_encoder):
    value = tf.constant([0.0, 1.0])
    with self.assertRaises(TypeError):
      encoding_utils.build_encoded_broadcast(value, bad_encoder)

  def test_build_encoded_broadcast_raises_incompatible_encoder(self):
    value = tf.constant([0.0, 1.0])
    incompatible_encoder = te.encoders.as_simple_encoder(
        te.encoders.identity(), tf.TensorSpec((3,)))
    with self.assertRaises(TypeError):
      encoding_utils.build_encoded_broadcast(value, incompatible_encoder)

  def test_build_encoded_broadcast_raises_bad_structure(self):
    value = [tf.constant([0.0, 1.0]), tf.constant([0.0, 1.0])]
    encoder = te.encoders.as_simple_encoder(te.encoders.identity(),
                                            tf.TensorSpec((2,)))
    with self.assertRaises(ValueError):
      encoding_utils.build_encoded_broadcast(value, encoder)


class EncodedBroadcastProcessTest(test.TestCase, parameterized.TestCase):
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

    self.assertIsInstance(broadcast_process, MeasuredProcess)
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


class EncodedSumTest(test.TestCase, parameterized.TestCase):
  """Tests for build_encoded_sum method."""

  def test_build_encoded_sum_raise_warning(self):
    value = tf.constant(np.random.rand(20))
    value_spec = tf.TensorSpec(value.shape, tf.dtypes.as_dtype(value.dtype))
    encoder = te.encoders.as_gather_encoder(te.encoders.identity(), value_spec)

    with warnings.catch_warnings(record=True):
      warnings.simplefilter('error', DeprecationWarning)
      with self.assertRaisesRegex(DeprecationWarning,
                                  'tff.utils.build_encoded_sum()'):
        encoding_utils.build_encoded_sum(value, encoder)

  @parameterized.named_parameters(
      ('tf_constant_identity', tf.constant, te.encoders.identity),
      ('tf_constant_uniform_quantization', tf.constant,
       lambda: te.encoders.uniform_quantization(8)),
      ('numpy_identity', lambda x: x, te.encoders.identity),
      ('numpy_uniform_quantization', lambda x: x,
       lambda: te.encoders.uniform_quantization(8)),
  )
  def test_build_encoded_sum(self, value_constructor, encoder_constructor):
    value = value_constructor(np.random.rand(20))
    value_spec = tf.TensorSpec(value.shape, tf.dtypes.as_dtype(value.dtype))
    value_type = computation_types.to_type(value_spec)
    encoder = te.encoders.as_gather_encoder(encoder_constructor(), value_spec)
    gather_fn = encoding_utils.build_encoded_sum(value, encoder)
    self.assertIsInstance(gather_fn, StatefulAggregateFn)

    state_type = gather_fn._initialize_fn.type_signature.result

    @computations.federated_computation(
        computation_types.FederatedType(state_type, placements.SERVER),
        computation_types.FederatedType(value_type, placements.CLIENTS),
        computation_types.FederatedType(
            computation_types.to_type(tf.float32), placements.CLIENTS))
    def gather_computation(state, value, weight):
      return gather_fn._next_fn(state, value, weight)

    gather_result_type = gather_computation.type_signature.result

    self.assertEqual(state_type, gather_result_type[0].member)
    self.assertEqual(placements.SERVER, gather_result_type[0].placement)
    self.assertEqual(value_type, gather_result_type[1].member)
    self.assertEqual(placements.SERVER, gather_result_type[1].placement)

  def test_run_encoded_sum(self):
    value = np.array([0.0, 1.0, 2.0, -1.0])
    value_spec = tf.TensorSpec(value.shape, tf.dtypes.as_dtype(value.dtype))
    value_type = computation_types.to_type(value_spec)
    encoder = te.encoders.as_gather_encoder(te.encoders.identity(), value_spec)
    gather_fn = encoding_utils.build_encoded_sum(value, encoder)
    initial_state = gather_fn.initialize()

    @computations.federated_computation(
        computation_types.FederatedType(
            gather_fn._initialize_fn.type_signature.result, placements.SERVER),
        computation_types.FederatedType(value_type, placements.CLIENTS))
    def call_gather(state, value):
      return gather_fn(state, value)

    _, value_sum = call_gather(initial_state, [value, value])
    self.assertAllClose(2 * value, value_sum)

    _, value_sum = call_gather(initial_state, [value, -value])
    self.assertAllClose(0 * value, value_sum)

    _, value_sum = call_gather(initial_state, [value, 2 * value])
    self.assertAllClose(3 * value, value_sum)

  @parameterized.named_parameters(*_bad_encoder_named_parameters)
  def test_build_encoded_sum_raises_bad_encoder(self, bad_encoder):
    value = tf.constant([0.0, 1.0])
    with self.assertRaises(TypeError):
      encoding_utils.build_encoded_sum(value, bad_encoder)

  def test_build_encoded_sum_raises_incompatible_encoder(self):
    value = tf.constant([0.0, 1.0])
    incompatible_encoder = te.encoders.as_gather_encoder(
        te.encoders.identity(), tf.TensorSpec((3,)))
    with self.assertRaises(TypeError):
      encoding_utils.build_encoded_sum(value, incompatible_encoder)

  def test_build_encoded_sum_raises_bad_structure(self):
    value = [tf.constant([0.0, 1.0]), tf.constant([0.0, 1.0])]
    encoder = te.encoders.as_gather_encoder(te.encoders.identity(),
                                            tf.TensorSpec((2,)))
    with self.assertRaises(ValueError):
      encoding_utils.build_encoded_sum(value, encoder)


class EncodedSumProcessTest(test.TestCase, parameterized.TestCase):
  """Tests for build_encoded_sum_process method."""

  @parameterized.named_parameters(
      ('tf_constant_identity', tf.constant, te.encoders.identity),
      ('tf_constant_uniform_quantization', tf.constant,
       lambda: te.encoders.uniform_quantization(8)),
      ('numpy_identity', lambda x: x, te.encoders.identity),
      ('numpy_uniform_quantization', lambda x: x,
       lambda: te.encoders.uniform_quantization(8)),
  )
  def test_build_encoded_sum_process(self, value_constructor,
                                     encoder_constructor):
    value = value_constructor(np.random.rand(20))
    value_spec = tf.TensorSpec(value.shape, tf.dtypes.as_dtype(value.dtype))
    value_type = computation_types.to_type(value_spec)
    encoder = te.encoders.as_gather_encoder(encoder_constructor(), value_spec)
    gather_process = encoding_utils.build_encoded_sum_process(
        value_type, encoder)
    state_type = gather_process._initialize_fn.type_signature.result
    gather_signature = gather_process._next_fn.type_signature

    self.assertIsInstance(gather_process, MeasuredProcess)
    self.assertEqual(state_type, gather_signature.result[0])
    self.assertEqual(placements.SERVER, gather_signature.result[0].placement)
    self.assertEqual(value_type, gather_signature.result[1].member)
    self.assertEqual(placements.SERVER, gather_signature.result[1].placement)

  def test_run_encoded_sum_process(self):
    value = np.array([0.0, 1.0, 2.0, -1.0])
    value_spec = tf.TensorSpec(value.shape, tf.dtypes.as_dtype(value.dtype))
    encoder = te.encoders.as_gather_encoder(te.encoders.identity(), value_spec)
    value_type = type_conversions.type_from_tensors(value)
    gather_process = encoding_utils.build_encoded_sum_process(
        value_type, encoder)
    initial_state = gather_process.initialize()
    call_gather = gather_process._next_fn

    output = call_gather(initial_state, [value, value])
    self.assertAllClose(2 * value, output.result)

    output = call_gather(initial_state, [value, -value])
    self.assertAllClose(0 * value, output.result)

    output = call_gather(initial_state, [value, 2 * value])
    self.assertAllClose(3 * value, output.result)

  @parameterized.named_parameters(*_bad_encoder_named_parameters)
  def test_build_encoded_sum_process_raises_bad_encoder(self, bad_encoder):
    value_type = computation_types.TensorType(tf.float32, shape=[2])
    with self.assertRaises(TypeError):
      encoding_utils.build_encoded_sum_process(value_type, bad_encoder)

  def test_build_encoded_sum_process_raises_incompatible_encoder(self):
    value_type = computation_types.TensorType(tf.float32, shape=[2])
    incompatible_encoder = te.encoders.as_gather_encoder(
        te.encoders.identity(), tf.TensorSpec((3,)))
    with self.assertRaises(TypeError):
      encoding_utils.build_encoded_sum_process(value_type, incompatible_encoder)

  def test_build_encoded_sum_process_raises_bad_structure(self):
    value_type = computation_types.StructType([
        computation_types.TensorType(tf.float32, shape=[2]),
        computation_types.TensorType(tf.float32, shape=[2])
    ])
    encoder = te.encoders.as_gather_encoder(te.encoders.identity(),
                                            tf.TensorSpec((2,)))
    with self.assertRaises(ValueError):
      encoding_utils.build_encoded_sum_process(value_type, encoder)


class EncodedMeanTest(test.TestCase, parameterized.TestCase):
  """Tests for build_encoded_mean method."""

  def test_build_encoded_mean_raise_warning(self):
    value = tf.constant(np.random.rand(20))
    value_spec = tf.TensorSpec(value.shape, tf.dtypes.as_dtype(value.dtype))
    encoder = te.encoders.as_gather_encoder(te.encoders.identity(), value_spec)

    with warnings.catch_warnings(record=True):
      warnings.simplefilter('error', DeprecationWarning)
      with self.assertRaisesRegex(DeprecationWarning,
                                  'tff.utils.build_encoded_mean()'):
        encoding_utils.build_encoded_mean(value, encoder)

  @parameterized.named_parameters(
      ('tf_constant_identity', tf.constant, te.encoders.identity),
      ('tf_constant_uniform_quantization', tf.constant,
       lambda: te.encoders.uniform_quantization(8)),
      ('numpy_identity', lambda x: x, te.encoders.identity),
      ('numpy_uniform_quantization', lambda x: x,
       lambda: te.encoders.uniform_quantization(8)),
  )
  def test_build_encoded_mean(self, value_constructor, encoder_constructor):
    value = value_constructor(np.random.rand(20))
    value_spec = tf.TensorSpec(value.shape, tf.dtypes.as_dtype(value.dtype))
    value_type = computation_types.to_type(value_spec)
    encoder = te.encoders.as_gather_encoder(encoder_constructor(), value_spec)
    gather_fn = encoding_utils.build_encoded_mean(value, encoder)
    state_type = gather_fn._initialize_fn.type_signature.result
    gather_signature = computations.federated_computation(
        gather_fn._next_fn,
        computation_types.FederatedType(state_type, placements.SERVER),
        computation_types.FederatedType(value_type, placements.CLIENTS),
        computation_types.FederatedType(
            computation_types.to_type(tf.float32),
            placements.CLIENTS)).type_signature

    self.assertIsInstance(gather_fn, StatefulAggregateFn)
    self.assertEqual(state_type, gather_signature.result[0].member)
    self.assertEqual(placements.SERVER, gather_signature.result[0].placement)
    self.assertEqual(value_type, gather_signature.result[1].member)
    self.assertEqual(placements.SERVER, gather_signature.result[1].placement)

  def test_run_encoded_mean(self):
    value = np.array([0.0, 1.0, 2.0, -1.0])
    value_spec = tf.TensorSpec(value.shape, tf.dtypes.as_dtype(value.dtype))
    value_type = computation_types.to_type(value_spec)
    encoder = te.encoders.as_gather_encoder(te.encoders.identity(), value_spec)
    gather_fn = encoding_utils.build_encoded_mean(value, encoder)
    initial_state = gather_fn.initialize()

    @computations.federated_computation(
        computation_types.FederatedType(
            gather_fn._initialize_fn.type_signature.result, placements.SERVER),
        computation_types.FederatedType(value_type, placements.CLIENTS),
        computation_types.FederatedType(
            computation_types.to_type(tf.float32), placements.CLIENTS))
    def call_gather(state, value, weight):
      return gather_fn(state, value, weight)

    _, value_mean = call_gather(initial_state, [value, value], [1.0, 1.0])
    self.assertAllClose(1 * value, value_mean)

    _, value_mean = call_gather(initial_state, [value, value], [0.3, 0.7])
    self.assertAllClose(1 * value, value_mean)

    _, value_mean = call_gather(initial_state, [value, 2 * value], [1.0, 2.0])
    self.assertAllClose(5 / 3 * value, value_mean)

  @parameterized.named_parameters(*_bad_encoder_named_parameters)
  def test_build_encoded_mean_raises_bad_encoder(self, bad_encoder):
    value = tf.constant([0.0, 1.0])
    with self.assertRaises(TypeError):
      encoding_utils.build_encoded_mean(value, bad_encoder)

  def test_build_encoded_mean_raises_incompatible_encoder(self):
    value = tf.constant([0.0, 1.0])
    incompatible_encoder = te.encoders.as_gather_encoder(
        te.encoders.identity(), tf.TensorSpec((3,)))
    with self.assertRaises(TypeError):
      encoding_utils.build_encoded_mean(value, incompatible_encoder)

  def test_build_encoded_mean_raises_bad_structure(self):
    value = [tf.constant([0.0, 1.0]), tf.constant([0.0, 1.0])]
    encoder = te.encoders.as_gather_encoder(te.encoders.identity(),
                                            tf.TensorSpec((2,)))
    with self.assertRaises(ValueError):
      encoding_utils.build_encoded_mean(value, encoder)


class EncodedMeanProcessTest(test.TestCase, parameterized.TestCase):
  """Tests for build_encoded_mean_process method."""

  @parameterized.named_parameters(
      ('tf_constant_identity', tf.constant, te.encoders.identity),
      ('tf_constant_uniform_quantization', tf.constant,
       lambda: te.encoders.uniform_quantization(8)),
      ('numpy_identity', lambda x: x, te.encoders.identity),
      ('numpy_uniform_quantization', lambda x: x,
       lambda: te.encoders.uniform_quantization(8)),
  )
  def test_build_encoded_mean_process(self, value_constructor,
                                      encoder_constructor):
    value = value_constructor(np.random.rand(20))
    value_spec = tf.TensorSpec(value.shape, tf.dtypes.as_dtype(value.dtype))
    value_type = computation_types.to_type(value_spec)
    encoder = te.encoders.as_gather_encoder(encoder_constructor(), value_spec)
    gather_process = encoding_utils.build_encoded_mean_process(
        value_type, encoder)
    state_type = gather_process._initialize_fn.type_signature.result
    gather_signature = gather_process._next_fn.type_signature

    self.assertIsInstance(gather_process, MeasuredProcess)
    self.assertEqual(state_type, gather_signature.result[0])
    self.assertEqual(placements.SERVER, gather_signature.result[0].placement)
    self.assertEqual(value_type, gather_signature.result[1].member)
    self.assertEqual(placements.SERVER, gather_signature.result[1].placement)

  def test_run_encoded_mean_process(self):
    value = np.array([0.0, 1.0, 2.0, -1.0])
    value_spec = tf.TensorSpec(value.shape, tf.dtypes.as_dtype(value.dtype))
    encoder = te.encoders.as_gather_encoder(te.encoders.identity(), value_spec)
    value_type = type_conversions.type_from_tensors(value)
    gather_process = encoding_utils.build_encoded_mean_process(
        value_type, encoder)
    initial_state = gather_process.initialize()
    call_gather = gather_process._next_fn

    output = call_gather(initial_state, [value, value], [1.0, 1.0])
    self.assertAllClose(1 * value, output.result)

    output = call_gather(initial_state, [value, value], [0.3, 0.7])
    self.assertAllClose(1 * value, output.result)

    output = call_gather(initial_state, [value, 2 * value], [1.0, 2.0])
    self.assertAllClose(5 / 3 * value, output.result)

  @parameterized.named_parameters(*_bad_encoder_named_parameters)
  def test_build_encoded_mean_process_raises_bad_encoder(self, bad_encoder):
    value_type = computation_types.TensorType(tf.float32, shape=[2])
    with self.assertRaises(TypeError):
      encoding_utils.build_encoded_mean_process(value_type, bad_encoder)

  def test_build_encoded_mean_process_raises_incompatible_encoder(self):
    value_type = computation_types.TensorType(tf.float32, shape=[2])
    incompatible_encoder = te.encoders.as_gather_encoder(
        te.encoders.identity(), tf.TensorSpec((3,)))
    with self.assertRaises(TypeError):
      encoding_utils.build_encoded_mean_process(value_type,
                                                incompatible_encoder)

  def test_build_encoded_mean_process_raises_bad_structure(self):
    value_type = computation_types.StructType([
        computation_types.TensorType(tf.float32, shape=[2]),
        computation_types.TensorType(tf.float32, shape=[2])
    ])
    encoder = te.encoders.as_gather_encoder(te.encoders.identity(),
                                            tf.TensorSpec((2,)))
    with self.assertRaises(ValueError):
      encoding_utils.build_encoded_mean_process(value_type, encoder)


class EncodingUtilsTest(test.TestCase, parameterized.TestCase):
  """Tests for utilities for building StatefulFns."""

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


if __name__ == '__main__':
  execution_contexts.set_local_execution_context()
  test.main()
