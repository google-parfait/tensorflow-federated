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
"""Tests for learning.framework.optimizer_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

import tensorflow_federated as tff
from tensorflow_federated.python.common_libs import test
from tensorflow_federated.python.core.utils import encoding_utils
from tensorflow_federated.python.learning import model_examples
from tensorflow_federated.python.learning.framework import optimizer_utils
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te


class EncodedBroadcastTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('tf_constant_identity', tf.constant, te.encoders.identity),
      ('tf_constant_uniform_quantization', tf.constant,
       lambda: te.encoders.uniform_quantization(8)),
      ('numpy_identity', lambda x: x, te.encoders.identity),
      ('numpy_uniform_quantization', lambda x: x,
       lambda: te.encoders.uniform_quantization(8)),
  )
  def test_build_broadcast(self, value_constructor, encoder_constructor):
    value = value_constructor(np.random.rand(20))
    value_spec = tf.TensorSpec(value.shape, tf.as_dtype(value.dtype))
    value_type = tff.to_type(value_spec)
    encoder = te.core.SimpleEncoder(encoder_constructor(), value_spec)
    broadcast_fn = encoding_utils.build_broadcast(value, encoder)
    broadcast_signature = broadcast_fn._next_fn.type_signature

    self.assertIsInstance(broadcast_fn, tff.utils.StatefulBroadcastFn)
    self.assertEqual(value_type, broadcast_signature.parameter[1].member)
    self.assertEqual(value_type, broadcast_signature.result[1].member)
    self.assertEqual(broadcast_signature.parameter[1].placement, tff.SERVER)
    self.assertEqual(broadcast_signature.result[1].placement, tff.CLIENTS)

  @parameterized.parameters([1.0, 'str', object, te.encoders.identity()])
  def test_build_broadcast_raises_bad_encoder(self, bad_encoder):
    value = tf.constant([0.0, 1.0])
    with self.assertRaises(TypeError):
      encoding_utils.build_broadcast(value, bad_encoder)

  def test_build_broadcast_raises_incompatible_encoder(self):
    value = tf.constant([0.0, 1.0])
    incompatible_encoder = te.core.SimpleEncoder(te.encoders.identity(),
                                                 tf.TensorSpec((3,)))
    with self.assertRaises(TypeError):
      encoding_utils.build_broadcast(value, incompatible_encoder)

  def test_build_broadcast_raises_bad_structure(self):
    value = [tf.constant([0.0, 1.0]), tf.constant([0.0, 1.0])]
    encoder = te.core.SimpleEncoder(te.encoders.identity(), tf.TensorSpec((2,)))
    with self.assertRaises(ValueError):
      encoding_utils.build_broadcast(value, encoder)


class EncodingUtilsTest(test.TestCase, parameterized.TestCase):
  """Tests for utilities for building StatefulFns."""

  # pyformat: disable
  @parameterized.parameters([
      te.encoders.identity,
      lambda: te.encoders.uniform_quantization(8),
      lambda: te.encoders.hadamard_quantization(8),
      lambda: te.core.EncoderComposer(PlusOneOverNEncodingStage()).make()
  ])
  # pyformat: enable
  def test_build_encode_decode_tf_computations_for_broadcast(
      self, encoder_constructor):
    value_spec = tf.TensorSpec((20,), tf.float32)
    encoder = te.core.SimpleEncoder(encoder_constructor(), value_spec)

    initial_state_fn = encoding_utils._build_initial_state_tf_computation(
        encoder)
    state_type = initial_state_fn.type_signature.result
    value_type = tff.to_type(value_spec)
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

  def test_broadcast_from_encoder_fn(self):
    values = [tf.constant(1.0), tf.constant([[1.0, 2.0], [3.0, 4.0]])]
    broadcast_fn = encoding_utils.broadcast_from_encoder_fn(
        values, _test_encoder_fn())
    self.assertIsInstance(broadcast_fn, tff.utils.StatefulBroadcastFn)

  def test_broadcast_from_model_fn_encoder_fn(self):
    model_fn = model_examples.TrainableLinearRegression
    broadcast_fn = encoding_utils.broadcast_from_model_fn_encoder_fn(
        model_fn, _test_encoder_fn())
    self.assertIsInstance(broadcast_fn, tff.utils.StatefulBroadcastFn)


class IterativeProcessTest(test.TestCase, parameterized.TestCase):
  """End-to-end tests using `tff.utils.IterativeProcess`."""

  def test_iterative_process_with_encoding(self):
    model_fn = model_examples.TrainableLinearRegression
    broadcast_fn = encoding_utils.broadcast_from_model_fn_encoder_fn(
        model_fn, _test_encoder_fn())
    iterative_process = optimizer_utils.build_model_delta_optimizer_process(
        model_fn=model_fn,
        model_to_client_delta_fn=DummyClientDeltaFn,
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        stateful_model_broadcast_fn=broadcast_fn)

    ds = tf.data.Dataset.from_tensor_slices({
        'x': [[1., 2.], [3., 4.]],
        'y': [[5.], [6.]]
    }).batch(2)
    federated_ds = [ds] * 3

    state = iterative_process.initialize()
    self.assertEqual(state.model_broadcast_state.trainable.a[0], 1)

    state, _ = iterative_process.next(state, federated_ds)
    self.assertEqual(state.model_broadcast_state.trainable.a[0], 2)


class DummyClientDeltaFn(optimizer_utils.ClientDeltaFn):

  def __init__(self, model_fn):
    self._model = model_fn()

  @property
  def variables(self):
    return []

  @tf.function
  def __call__(self, dataset, initial_weights):
    # Iterate over the dataset to get new metric values.
    def reduce_fn(dummy, batch):
      self._model.train_on_batch(batch)
      return dummy

    dataset.reduce(tf.constant(0.0), reduce_fn)

    # Create some fake weight deltas to send back.
    trainable_weights_delta = tf.nest.map_structure(lambda x: -tf.ones_like(x),
                                                    initial_weights.trainable)
    client_weight = tf.constant(1.0)
    return optimizer_utils.ClientOutput(
        trainable_weights_delta,
        weights_delta_weight=client_weight,
        model_output=self._model.report_local_outputs(),
        optimizer_output={'client_weight': client_weight})


# TODO(b/137613901): Remove this in next update of tfmot package, when
# te.testing is available.
@te.core.tf_style_adaptive_encoding_stage
class PlusOneOverNEncodingStage(te.core.AdaptiveEncodingStageInterface):
  """[Example] adaptive encoding stage, adding 1/N in N-th iteration.

  This is an example implementation of an `AdaptiveEncodingStageInterface` that
  modifies state, which controls the creation of params. This is also a simple
  example of how an `EncodingStageInterface` can be wrapped as an
  `AdaptiveEncodingStageInterface`, without modifying the wrapped encode and
  decode methods.
  """

  ENCODED_VALUES_KEY = 'pn_values'
  ADD_PARAM_KEY = 'pn_add'
  ITERATION_STATE_KEY = 'pn_iteration'

  @property
  def name(self):
    """See base class."""
    return 'plus_one_over_n'

  @property
  def compressible_tensors_keys(self):
    """See base class."""
    return [self.ENCODED_VALUES_KEY]

  @property
  def commutes_with_sum(self):
    """See base class."""
    return False

  @property
  def decode_needs_input_shape(self):
    """See base class."""
    return False

  @property
  def state_update_aggregation_modes(self):
    """See base class."""
    return {}

  def initial_state(self):
    """See base class."""
    return {self.ITERATION_STATE_KEY: tf.constant(1, dtype=tf.int32)}

  def update_state(self, state, state_update_tensors):
    """See base class."""
    del state_update_tensors  # Unused.
    return {
        self.ITERATION_STATE_KEY:
            state[self.ITERATION_STATE_KEY] + tf.constant(1, dtype=tf.int32)
    }

  def get_params(self, state):
    """See base class."""
    params = {
        self.ADD_PARAM_KEY: 1 / tf.to_float(state[self.ITERATION_STATE_KEY])
    }
    return params, params

  def encode(self, x, encode_params):
    """See base class."""
    return {self.ENCODED_VALUES_KEY: x + encode_params[self.ADD_PARAM_KEY]}, {}

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    """See base class."""
    del num_summands  # Unused.
    del shape  # Unused.
    decoded_x = (
        encoded_tensors[self.ENCODED_VALUES_KEY] -
        decode_params[self.ADD_PARAM_KEY])
    return decoded_x


def _test_encoder_fn():
  """Returns an example mapping of tensor to encoder, determined by shape."""
  identity_encoder = te.encoders.identity()
  test_encoder = te.core.EncoderComposer(PlusOneOverNEncodingStage()).make()

  def encoder_fn(tensor):
    if np.prod(tensor.shape) > 1:
      encoder = te.core.SimpleEncoder(test_encoder,
                                      tf.TensorSpec(tensor.shape, tensor.dtype))
    else:
      encoder = te.core.SimpleEncoder(identity_encoder,
                                      tf.TensorSpec(tensor.shape, tensor.dtype))
    return encoder

  return encoder_fn


if __name__ == '__main__':
  test.main()
