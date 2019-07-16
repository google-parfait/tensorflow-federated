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

import tensorflow_federated as tff
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.learning import model_utils
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te


def build_broadcast(values, encoders):
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
      tff.framework.type_from_tensors(values), tff.SERVER)

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

  return tff.utils.StatefulBroadcastFn(
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
    encoded_structure = tf.contrib.framework.nest.map_structure_up_to(
        encoders, lambda state, value, e: e.encode(value, state), state, value,
        encoders)
    encoded_value = tf.contrib.framework.nest.map_structure_up_to(
        encoders, lambda s: s[0], encoded_structure)
    new_state = tf.contrib.framework.nest.map_structure_up_to(
        encoders, lambda s: s[1], encoded_structure)
    return new_state, encoded_value

  @tff.tf_computation(encode.type_signature.result[1])
  def decode(encoded_value):
    """Decode tf_computation."""
    return tf.contrib.framework.nest.map_structure_up_to(
        encoders, lambda e, val: e.decode(val), encoders, encoded_value)

  return encode, decode


def broadcast_from_encoder_fn(values, encoder_fn):
  """Builds `StatefulBroadcastFn` for `values`.

  This method creates a `SimpleEncoder` for every value in `values`, as
  returned by `encoder_fn`.

  Args:
    values: A possible nested structure of values to be broadcasted.
    encoder_fn: A Python callable with a single argument, which is expected to
      be a `tf.Tensor` of shape and dtype to be encoded.

  Returns:
    A `StatefulBroadcastFn` for encoding and broadcasting `values`.

  Raises:
    TypeError: If `encoder_fn` is not a callable object.
  """
  py_typecheck.check_callable(encoder_fn)
  encoders = tf.nest.map_structure(encoder_fn, values)
  return build_broadcast(values, encoders)


def broadcast_from_model_fn_encoder_fn(model_fn, encoder_fn):
  """Builds `StatefulBroadcastFn` for weights of model returned by `model_fn`.

  This

  Args:
    model_fn: A Python callable with no arguments function that returns a
      `tff.learning.Model`.
    encoder_fn: A Python callable with a single argument, which is expected to
      be a `tf.Tensor` of shape and dtype to be encoded.

  Returns:
    A `StatefulBroadcastFn` for encoding and broadcasting the weights of model
    created by `model_fn`.

  Raises:
    TypeError: If `model_fn` or `encoder_fn` are not callable objects.
  """
  py_typecheck.check_callable(encoder_fn)
  value = model_utils.enhance(model_fn()).weights
  return broadcast_from_encoder_fn(value, encoder_fn)
