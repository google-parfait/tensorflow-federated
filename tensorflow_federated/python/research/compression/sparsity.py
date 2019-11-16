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
"""An example custom `te.core.Encoder` for `tff`."""

import tensorflow.compat.v2 as tf

from tensorflow_model_optimization.python.core.internal import tensor_encoding as te


@te.core.tf_style_adaptive_encoding_stage
class OddEvenSparseEncodingStage(te.core.AdaptiveEncodingStageInterface):
  """An example custom implementation of an `EncodingStageInterface`.

  Note: This is likely not what one would want to use in practice. Rather, this
  serves as an illustration of how a custom compression algorithm can be
  provided to `tff`.

  This encoding stage is expected to be run in an iterative manner, and
  alternatively zeroes out values corresponding to odd and even indices. Given
  the determinism of the non-zero indices selection, the encoded structure does
  not need to be represented as a sparse vector, but only the non-zero values
  are necessary. In the decode mehtod, the state (i.e., params derived from the
  state) is used to reconstruct the corresponding indices.

  Thus, this example encoding stage can realize representation saving of 2x.
  """

  ENCODED_VALUES_KEY = 'non_zero_floats'
  ODD_EVEN_PARAM_KEY = 'threshold'
  ODD_EVEN_STATE_KEY = 'odd_even'

  def encode(self, x, encode_params):
    indices = tf.range(
        encode_params[self.ODD_EVEN_PARAM_KEY], tf.size(x), delta=2)
    vals = tf.gather(tf.reshape(x, [-1]), indices)
    encoded_x = {self.ENCODED_VALUES_KEY: vals}
    state_update_tensors = {}
    return encoded_x, state_update_tensors

  def decode(self,
             encoded_tensors,
             decode_params,
             num_summands=None,
             shape=None):
    del num_summands  # Unused.
    indices = tf.range(
        decode_params[self.ODD_EVEN_PARAM_KEY], tf.reduce_prod(shape), delta=2)
    decoded_values = tf.scatter_nd(
        tf.expand_dims(indices, 1), encoded_tensors[self.ENCODED_VALUES_KEY],
        tf.expand_dims(tf.reduce_prod(shape), 0))
    return tf.reshape(decoded_values, shape)

  def initial_state(self):
    return {self.ODD_EVEN_STATE_KEY: tf.constant(0, dtype=tf.int32)}

  def update_state(self, state, state_update_tensors):
    del state_update_tensors  # Unused.
    return {
        self.ODD_EVEN_STATE_KEY:
            tf.math.floormod(state[self.ODD_EVEN_STATE_KEY] + 1, 2)
    }

  def get_params(self, state):
    encode_params = {self.ODD_EVEN_PARAM_KEY: state[self.ODD_EVEN_STATE_KEY]}
    decode_params = encode_params
    return encode_params, decode_params

  @property
  def name(self):
    return 'odd_even_sparse_encoding_stage'

  @property
  def compressible_tensors_keys(self):
    return [self.ENCODED_VALUES_KEY]

  @property
  def commutes_with_sum(self):
    return False

  @property
  def decode_needs_input_shape(self):
    return True

  @property
  def state_update_aggregation_modes(self):
    return {}


def sparse_quantizing_encoder(quantization_bits):
  """Constructor for the custom `te.core.Encoder`.

  This encoder first flattens the input (`FlattenEncodingStage`). Then, it
  zeroes out a half of the elements (`OddEvenSparseEncodingStage`), only
  returning the non-zero values. Then, it applies uniform quantization
  (`UniformQuantizationEncodingStage`), and simply packs the result into
  integers (`BitpackingEncodingStage`).

  Args:
    quantization_bits: An integer. Number of bits to use for uniform
      quantization.

  Returns:
    A `te.core.Encoder`.
  """
  encoder = te.core.EncoderComposer(
      te.stages.BitpackingEncodingStage(quantization_bits))
  encoder = encoder.add_parent(
      te.stages.UniformQuantizationEncodingStage(quantization_bits),
      te.stages.UniformQuantizationEncodingStage.ENCODED_VALUES_KEY)
  encoder = encoder.add_parent(OddEvenSparseEncodingStage(),
                               OddEvenSparseEncodingStage.ENCODED_VALUES_KEY)
  encoder = encoder.add_parent(
      te.stages.FlattenEncodingStage(),
      te.stages.FlattenEncodingStage.ENCODED_VALUES_KEY)
  return encoder.make()
