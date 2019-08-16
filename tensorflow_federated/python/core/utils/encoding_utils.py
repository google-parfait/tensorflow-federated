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
"""Utilities for building broadcast and aggregate functions with encoding.

This file contains utilities for building `StatefulBroadcastFn` and
`StatefulAggregateFn` utilizing `Encoder` class from `tensor_encoding` project,
to realize encoding (compression) of values being communicated between `SERVER`
and `CLIENTS`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_federated.python.core import api as tff
from tensorflow_federated.python.core import framework as tff_framework
from tensorflow_federated.python.core.utils.computation_utils import StatefulBroadcastFn
from tensorflow_federated.python.tensorflow_libs import nest as nest_contrib
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te


def build_encoded_broadcast(values, encoders):
  """Builds `StatefulBroadcastFn` for `values`, to be encoded by `encoders`.

  Args:
    values: Values to be broadcasted by the `StatefulBroadcastFn`. Must be
      convertible to `tff.Value`.
    encoders: A collection of `SimpleEncoder` objects to be used for encoding
      `values`. Must have the same structure as `values`.

  Returns:
    A `StatefulBroadcastFn` of which `next_fn` encodes the input at
    `tff.SERVER`, broadcasts the encoded representation and decodes the encoded
    representation at `tff.CLIENTS`.

  Raises:
    ValueError: If `values` and `encoders` do not have the same structure.
    TypeError: If `encoders` are not instances of `SimpleEncoder`, or if
      `values` are not compatible with the expected input of the `encoders`.
  """

  def validate_encoder(encoder, value):
    if not isinstance(encoder, te.core.SimpleEncoder):
      raise TypeError('Provided encoder must be an instance of SimpleEncoder.')
    if not encoder.input_tensorspec.is_compatible_with(
        tf.TensorSpec(value.shape, value.dtype)):
      raise TypeError('Provided encoder and value are not compatible.')

  tf.nest.assert_same_structure(values, encoders)
  tf.nest.map_structure(validate_encoder, encoders, values)

  initial_state_fn = _build_initial_state_tf_computation(encoders)

  fed_state_type = tff.FederatedType(initial_state_fn.type_signature.result,
                                     tff.SERVER)
  fed_value_type = tff.FederatedType(
      tff_framework.type_from_tensors(values), tff.SERVER)

  @tff.federated_computation(fed_state_type, fed_value_type)
  def encoded_broadcast_fn(state, value):
    """Broadcast function, to be wrapped as federated_computation."""

    state_type = state.type_signature.member
    value_type = value.type_signature.member

    encode_fn, decode_fn = _build_encode_decode_tf_computations_for_broadcast(
        state_type, value_type, encoders)

    new_state, encoded_value = tff.federated_apply(encode_fn, (state, value))
    client_encoded_value = tff.federated_broadcast(encoded_value)
    client_value = tff.federated_map(decode_fn, client_encoded_value)
    return new_state, client_value

  return StatefulBroadcastFn(
      initialize_fn=initial_state_fn, next_fn=encoded_broadcast_fn)


def _build_initial_state_tf_computation(encoders):
  """Utility for creating initial_state tf_computation."""

  @tff.tf_computation
  def initial_state_fn():
    return tf.nest.map_structure(lambda e: e.initial_state(), encoders)

  return initial_state_fn


# TODO(b/136219266): Remove dependency on tf.contrib.framework.nest.
def _build_encode_decode_tf_computations_for_broadcast(state_type, value_type,
                                                       encoders):
  """Utility for creating encode/decode tf_computations for broadcast."""

  @tff.tf_computation(state_type, value_type)
  def encode(state, value):
    """Encode tf_computation."""
    encoded_structure = nest_contrib.map_structure_up_to(
        encoders, lambda state, value, e: e.encode(value, state), state, value,
        encoders)
    encoded_value = nest_contrib.map_structure_up_to(encoders, lambda s: s[0],
                                                     encoded_structure)
    new_state = nest_contrib.map_structure_up_to(encoders, lambda s: s[1],
                                                 encoded_structure)
    return new_state, encoded_value

  @tff.tf_computation(encode.type_signature.result[1])
  def decode(encoded_value):
    """Decode tf_computation."""
    return nest_contrib.map_structure_up_to(encoders,
                                            lambda e, val: e.decode(val),
                                            encoders, encoded_value)

  return encode, decode
