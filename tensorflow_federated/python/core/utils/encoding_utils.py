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
to realize encoding (compression) of values being communicated between
`tff.SERVER` and `tff.CLIENTS`.
"""

import collections
import warnings

import attr
import tensorflow as tf
import tree

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.core.utils import computation_utils
from tensorflow_model_optimization.python.core.internal import tensor_encoding

_ALLOWED_ENCODERS = (tensor_encoding.core.SimpleEncoder,
                     tensor_encoding.core.GatherEncoder,
                     tensor_encoding.core.EncoderComposer)


@attr.s(eq=False, frozen=True)
class _NestGatherEncoder(object):
  """Structure for holding `tf_computations` needed for encoded_sum."""
  get_params_fn = attr.ib()
  encode_fn = attr.ib()
  decode_after_sum_fn = attr.ib()
  update_state_fn = attr.ib()
  zero_fn = attr.ib()
  accumulate_fn = attr.ib()
  merge_fn = attr.ib()
  report_fn = attr.ib()


# TODO(b/159836417): Depracate this function as part of the migration.
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
  warnings.warn(
      'Deprecation warning: tff.utils.build_encoded_broadcast() is deprecated, '
      'use tff.utils.build_encoded_broadcast_process() instead.',
      DeprecationWarning)

  tf.nest.assert_same_structure(values, encoders)
  tf.nest.map_structure(
      lambda e, v: _validate_encoder(e, v, tensor_encoding.core.SimpleEncoder),
      encoders, values)

  value_type = type_conversions.type_from_tensors(values)

  initial_state_fn, state_type = _build_initial_state_tf_computation(encoders)

  encode_fn, decode_fn = _build_encode_decode_tf_computations_for_broadcast(
      state_type, value_type, encoders)

  def encoded_broadcast_fn(state, value):
    """Encoded broadcast federated_computation."""
    new_state, encoded_value = intrinsics.federated_map(encode_fn,
                                                        (state, value))
    client_encoded_value = intrinsics.federated_broadcast(encoded_value)
    client_value = intrinsics.federated_map(decode_fn, client_encoded_value)
    return new_state, client_value

  return computation_utils.StatefulBroadcastFn(
      initialize_fn=initial_state_fn, next_fn=encoded_broadcast_fn)


def build_encoded_broadcast_process(value_type, encoders):
  """Builds `MeasuredProcess` for `value_type`, to be encoded by `encoders`.

  The returned `MeasuredProcess` has a next function with the TFF type
  signature:

  ```
  (<state_type@SERVER, {value_type}@CLIENTS> ->
   <state=state_type@SERVER, result=value_type@SERVER, measurements=()@SERVER>)
  ```

  Args:
    value_type: The type of values to be broadcasted by the `MeasuredProcess`.
      Either a `tff.TensorType` or a `tff.StructType`.
    encoders: A collection of `SimpleEncoder` objects to be used for encoding
      `values`. Must have the same structure as `values`.

  Returns:
    A `MeasuredProcess` of which `next_fn` encodes the input at `tff.SERVER`,
    broadcasts the encoded representation and decodes the encoded representation
    at `tff.CLIENTS`.

  Raises:
    ValueError: If `value_type` and `encoders` do not have the same structure.
    TypeError: If `encoders` are not instances of `SimpleEncoder`, or if
      `value_type` are not compatible with the expected input of the `encoders`.
  """
  py_typecheck.check_type(
      value_type, (computation_types.TensorType, computation_types.StructType))

  _validate_value_type_and_encoders(value_type, encoders,
                                    tensor_encoding.core.SimpleEncoder)

  initial_state_fn, state_type = _build_initial_state_tf_computation(encoders)

  @computations.federated_computation()
  def initial_state_comp():
    return intrinsics.federated_eval(initial_state_fn, placements.SERVER)

  encode_fn, decode_fn = _build_encode_decode_tf_computations_for_broadcast(
      state_type, value_type, encoders)

  @computations.federated_computation(initial_state_comp.type_signature.result,
                                      computation_types.FederatedType(
                                          value_type, placements.SERVER))
  def encoded_broadcast_comp(state, value):
    """Encoded broadcast federated_computation."""
    empty_metrics = intrinsics.federated_value((), placements.SERVER)
    new_state, encoded_value = intrinsics.federated_map(encode_fn,
                                                        (state, value))
    client_encoded_value = intrinsics.federated_broadcast(encoded_value)
    client_value = intrinsics.federated_map(decode_fn, client_encoded_value)
    return collections.OrderedDict(
        state=new_state, result=client_value, measurements=empty_metrics)

  return measured_process.MeasuredProcess(
      initialize_fn=initial_state_comp, next_fn=encoded_broadcast_comp)


def _build_encoded_sum_fn(nest_encoder):
  """Utility for creating encoded_sum based on _NestGatherEncoder."""

  def encoded_sum_fn(state, values, weight=None):
    """Encoded sum federated_computation."""
    del weight  # Unused.
    encode_params, decode_before_sum_params, decode_after_sum_params = (
        intrinsics.federated_map(nest_encoder.get_params_fn, state))
    encode_params = intrinsics.federated_broadcast(encode_params)
    decode_before_sum_params = intrinsics.federated_broadcast(
        decode_before_sum_params)

    encoded_values = intrinsics.federated_map(
        nest_encoder.encode_fn,
        [values, encode_params, decode_before_sum_params])

    aggregated_values = intrinsics.federated_aggregate(
        encoded_values, nest_encoder.zero_fn(), nest_encoder.accumulate_fn,
        nest_encoder.merge_fn, nest_encoder.report_fn)

    decoded_values = intrinsics.federated_map(
        nest_encoder.decode_after_sum_fn,
        [aggregated_values.values, decode_after_sum_params])

    updated_state = intrinsics.federated_map(
        nest_encoder.update_state_fn,
        [state, aggregated_values.state_update_tensors])
    return updated_state, decoded_values

  return encoded_sum_fn


# TODO(b/159836417): Depracate this function as part of the migration.
def build_encoded_sum(values, encoders):
  """Builds `StatefulAggregateFn` for `values`, to be encoded by `encoders`.

  Args:
    values: Values to be encoded by the `StatefulAggregateFn`. Must be
      convertible to `tff.Value`.
    encoders: A collection of `GatherEncoder` objects to be used for encoding
      `values`. Must have the same structure as `values`.

  Returns:
    A `StatefulAggregateFn` of which `next_fn` encodes the input at
    `tff.CLIENTS`, and computes their sum at `tff.SERVER`, automatically
    splitting the decoding part based on its commutativity with sum.

  Raises:
    ValueError: If `values` and `encoders` do not have the same structure.
    TypeError: If `encoders` are not instances of `GatherEncoder`, or if
      `values` are not compatible with the expected input of the `encoders`.
  """
  warnings.warn(
      'Deprecation warning: tff.utils.build_encoded_sum() is deprecated, use '
      'tff.utils.build_encoded_sum_process() instead.', DeprecationWarning)

  tf.nest.assert_same_structure(values, encoders)
  tf.nest.map_structure(
      lambda e, v: _validate_encoder(e, v, tensor_encoding.core.GatherEncoder),
      encoders, values)

  value_type = type_conversions.type_from_tensors(values)

  initial_state_fn, state_type = _build_initial_state_tf_computation(encoders)

  nest_encoder = _build_tf_computations_for_gather(state_type, value_type,
                                                   encoders)
  encoded_sum_fn = _build_encoded_sum_fn(nest_encoder)

  return computation_utils.StatefulAggregateFn(
      initialize_fn=initial_state_fn, next_fn=encoded_sum_fn)


def build_encoded_sum_process(value_type, encoders):
  """Builds `MeasuredProcess` for `value_type`, to be encoded by `encoders`.

  The returned `MeasuredProcess` has a next function with the TFF type
  signature:

  ```
  (<state_type@SERVER, {value_type}@CLIENTS> ->
   <state=state_type@SERVER, result=value_type@SERVER, measurements=()@SERVER>)
  ```

  Args:
    value_type: The type of values to be encoded by the `MeasuredProcess`.
      Either a `tff.TensorType` or a `tff.StructType`.
    encoders: A collection of `GatherEncoder` objects to be used for encoding
      `values`. Must have the same structure as `values`.

  Returns:
    A `MeasuredProcess` of which `next_fn` encodes the input at `tff.CLIENTS`,
    and computes their sum at `tff.SERVER`, automatically splitting the decoding
    part based on its commutativity with sum.

  Raises:
    ValueError: If `value_type` and `encoders` do not have the same structure.
    TypeError: If `encoders` are not instances of `GatherEncoder`, or if
      `value_type` are not compatible with the expected input of the `encoders`.
  """
  py_typecheck.check_type(
      value_type, (computation_types.TensorType, computation_types.StructType))

  _validate_value_type_and_encoders(value_type, encoders,
                                    tensor_encoding.core.GatherEncoder)

  initial_state_fn, state_type = _build_initial_state_tf_computation(encoders)

  @computations.federated_computation()
  def initial_state_comp():
    return intrinsics.federated_eval(initial_state_fn, placements.SERVER)

  nest_encoder = _build_tf_computations_for_gather(state_type, value_type,
                                                   encoders)
  encoded_sum_fn = _build_encoded_sum_fn(nest_encoder)

  @computations.federated_computation(initial_state_comp.type_signature.result,
                                      computation_types.FederatedType(
                                          value_type, placements.CLIENTS))
  def encoded_sum_comp(state, values):
    """Encoded sum federated_computation."""
    empty_metrics = intrinsics.federated_value((), placements.SERVER)
    state, result = encoded_sum_fn(state, values)
    return collections.OrderedDict(
        state=state, result=result, measurements=empty_metrics)

  return measured_process.MeasuredProcess(
      initialize_fn=initial_state_comp, next_fn=encoded_sum_comp)


# TODO(b/159836417): Depracate this function as part of the migration.
def build_encoded_mean(values, encoders):
  """Builds `StatefulAggregateFn` for `values`, to be encoded by `encoders`.

  Args:
    values: Values to be encoded by the `StatefulAggregateFn`. Must be
      convertible to `tff.Value`.
    encoders: A collection of `GatherEncoder` objects to be used for encoding
      `values`. Must have the same structure as `values`.

  Returns:
    A `StatefulAggregateFn` of which `next_fn` encodes the input at
    `tff.CLIENTS`, and computes their mean at `tff.SERVER`, automatically
    splitting the decoding part based on its commutativity with sum.

  Raises:
    ValueError: If `values` and `encoders` do not have the same structure.
    TypeError: If `encoders` are not instances of `GatherEncoder`, or if
      `values` are not compatible with the expected input of the `encoders`.
  """
  warnings.warn(
      'Deprecation warning: tff.utils.build_encoded_mean() is deprecated, use '
      'tff.utils.build_encoded_mean_process() instead.', DeprecationWarning)

  tf.nest.assert_same_structure(values, encoders)
  tf.nest.map_structure(
      lambda e, v: _validate_encoder(e, v, tensor_encoding.core.GatherEncoder),
      encoders, values)

  value_type = type_conversions.type_from_tensors(values)

  initial_state_fn, state_type = _build_initial_state_tf_computation(encoders)

  nest_encoder = _build_tf_computations_for_gather(state_type, value_type,
                                                   encoders)
  encoded_sum_fn = _build_encoded_sum_fn(nest_encoder)

  @computations.tf_computation(value_type, tf.float32)
  def multiply_fn(value, weight):
    return tf.nest.map_structure(lambda v: v * tf.cast(weight, v.dtype), value)

  @computations.tf_computation(value_type, tf.float32)
  def divide_fn(value, denominator):
    return tf.nest.map_structure(lambda v: v / tf.cast(denominator, v.dtype),
                                 value)

  def encoded_mean_fn(state, values, weight):
    weighted_values = intrinsics.federated_map(multiply_fn, [values, weight])
    updated_state, summed_decoded_values = encoded_sum_fn(
        state, weighted_values)
    summed_weights = intrinsics.federated_sum(weight)
    decoded_values = intrinsics.federated_map(
        divide_fn, [summed_decoded_values, summed_weights])
    return updated_state, decoded_values

  return computation_utils.StatefulAggregateFn(
      initialize_fn=initial_state_fn, next_fn=encoded_mean_fn)


def build_encoded_mean_process(value_type, encoders):
  """Builds `MeasuredProcess` for `value_type`, to be encoded by `encoders`.

  The returned `MeasuredProcess` has a next function with the TFF type
  signature:

  ```
  (<state_type@SERVER, {value_type}@CLIENTS> ->
   <state=state_type@SERVER, result=value_type@SERVER, measurements=()@SERVER>)
  ```

  Args:
    value_type: The type of values to be encoded by the `MeasuredProcess`.
      Either a `tff.TensorType` or a `tff.StructType`.
    encoders: A collection of `GatherEncoder` objects to be used for encoding
      `values`. Must have the same structure as `values`.

  Returns:
    A `MeasuredProcess` of which `next_fn` encodes the input at `tff.CLIENTS`,
    and computes their mean at `tff.SERVER`, automatically splitting the
    decoding part based on its commutativity with sum.

  Raises:
    ValueError: If `value_type` and `encoders` do not have the same structure.
    TypeError: If `encoders` are not instances of `GatherEncoder`, or if
      `value_type` are not compatible with the expected input of the `encoders`.
  """
  py_typecheck.check_type(
      value_type, (computation_types.TensorType, computation_types.StructType))

  _validate_value_type_and_encoders(value_type, encoders,
                                    tensor_encoding.core.GatherEncoder)

  initial_state_fn, state_type = _build_initial_state_tf_computation(encoders)

  @computations.federated_computation()
  def initial_state_comp():
    return intrinsics.federated_eval(initial_state_fn, placements.SERVER)

  nest_encoder = _build_tf_computations_for_gather(state_type, value_type,
                                                   encoders)
  encoded_sum_fn = _build_encoded_sum_fn(nest_encoder)

  @computations.tf_computation(value_type, tf.float32)
  def multiply_fn(value, weight):
    return tf.nest.map_structure(lambda v: v * tf.cast(weight, v.dtype), value)

  @computations.tf_computation(value_type, tf.float32)
  def divide_fn(value, denominator):
    return tf.nest.map_structure(lambda v: v / tf.cast(denominator, v.dtype),
                                 value)

  @computations.federated_computation(
      initial_state_comp.type_signature.result,
      computation_types.FederatedType(value_type, placements.CLIENTS),
      computation_types.FederatedType(tf.float32, placements.CLIENTS))
  def encoded_mean_comp(state, values, weight):
    """Encoded mean federated_computation."""
    empty_metrics = intrinsics.federated_value((), placements.SERVER)
    weighted_values = intrinsics.federated_map(multiply_fn, (values, weight))
    updated_state, summed_decoded_values = encoded_sum_fn(
        state, weighted_values)
    summed_weights = intrinsics.federated_sum(weight)
    decoded_values = intrinsics.federated_map(
        divide_fn, (summed_decoded_values, summed_weights))
    return collections.OrderedDict(
        state=updated_state, result=decoded_values, measurements=empty_metrics)

  return measured_process.MeasuredProcess(
      initialize_fn=initial_state_comp, next_fn=encoded_mean_comp)


def _build_initial_state_tf_computation(encoders):
  """Utility for creating initial_state tf_computation."""

  @computations.tf_computation
  def initial_state_fn():
    return tf.nest.map_structure(lambda e: e.initial_state(), encoders)

  return initial_state_fn, initial_state_fn.type_signature.result


def _slice(encoders, nested_value, idx):
  """Takes a slice of nested values.

  We use a collection of encoders to encode a collection of values. When a
  method of the encoder returns a tuple, e.g., encode / decode params of the
  get_params method, we need to recover the matching collection of encode params
  and collection of decode params. This method is a utility to achieve this.

  Args:
    encoders: A collection of encoders.
    nested_value: A collection of indexable values of the same structure as
      `encoders`.
    idx: An integer. Index of the values in `nested_value` along which to take
      the slice.

  Returns:
    A collection of values of the same structure as `encoders`.
  """
  return tree.map_structure_up_to(encoders, lambda t: t[idx], nested_value)


def _build_encode_decode_tf_computations_for_broadcast(state_type, value_type,
                                                       encoders):
  """Utility for creating encode/decode tf_computations for broadcast."""

  @computations.tf_computation(state_type, value_type)
  def encode(state, value):
    """Encode tf_computation."""
    encoded_structure = tree.map_structure_up_to(
        encoders, lambda state, value, e: e.encode(value, state), state, value,
        encoders)
    encoded_value = _slice(encoders, encoded_structure, 0)
    new_state = _slice(encoders, encoded_structure, 1)
    return new_state, encoded_value

  @computations.tf_computation(encode.type_signature.result[1])
  def decode(encoded_value):
    """Decode tf_computation."""
    return tree.map_structure_up_to(encoders, lambda e, val: e.decode(val),
                                    encoders, encoded_value)

  return encode, decode


def _build_tf_computations_for_gather(state_type, value_type, encoders):
  """Utility for creating tf_computations for encoded sum and mean.

  This method maps a collection of GatherEncoder objects to partial computations
  for encoding a collection of values jointly, and adds a logic for computing
  the number of summands in decode_before_sum, once for the entire collection,
  not on a per-value basis.

  Args:
    state_type: A `tff.Type` describing the collection of states handled by
      `encoders`.
    value_type: A `tff.Type` describing the collection of values to be encoded
      by `encoders`.
    encoders: A collection of `GatherEncoder` objects.

  Returns:
    A `_NestGatherEncoder` namedtuple holding the relevant tf_computations.
  """

  @computations.tf_computation(state_type)
  def get_params_fn(state):
    params = tree.map_structure_up_to(encoders, lambda e, s: e.get_params(s),
                                      encoders, state)
    encode_params = _slice(encoders, params, 0)
    decode_before_sum_params = _slice(encoders, params, 1)
    decode_after_sum_params = _slice(encoders, params, 2)
    return encode_params, decode_before_sum_params, decode_after_sum_params

  encode_params_type = get_params_fn.type_signature.result[0]
  decode_before_sum_params_type = get_params_fn.type_signature.result[1]
  decode_after_sum_params_type = get_params_fn.type_signature.result[2]

  # TODO(b/139844355): Get rid of decode_before_sum_params.
  # We pass decode_before_sum_params to the encode method, because TFF currently
  # does not have a mechanism to make a tff.SERVER placed value available inside
  # of intrinsics.federated_aggregate - in production, this could mean an
  # intermediary aggregator node. So currently, we send the params to clients,
  # and ask them to send them back as part of the encoded structure.
  @computations.tf_computation(value_type, encode_params_type,
                               decode_before_sum_params_type)
  def encode_fn(x, encode_params, decode_before_sum_params):
    encoded_structure = tree.map_structure_up_to(
        encoders, lambda e, *args: e.encode(*args), encoders, x, encode_params)
    encoded_x = _slice(encoders, encoded_structure, 0)
    state_update_tensors = _slice(encoders, encoded_structure, 1)
    return encoded_x, decode_before_sum_params, state_update_tensors

  state_update_tensors_type = encode_fn.type_signature.result[2]

  # This is not a @computations.tf_computation because it will be used below
  # when bulding the computations.tf_computations that will compose a
  # intrinsics.federated_aggregate...
  # @tf.function
  def decode_before_sum_tf_function(encoded_x, decode_before_sum_params):
    part_decoded_x = tree.map_structure_up_to(
        encoders, lambda e, *args: e.decode_before_sum(*args), encoders,
        encoded_x, decode_before_sum_params)
    one = tf.constant((1,), tf.int32)
    return part_decoded_x, one

  # ...however, result type is needed to build the subsequent tf_compuations.
  @computations.tf_computation(encode_fn.type_signature.result[0:2])
  def tmp_decode_before_sum_fn(encoded_x, decode_before_sum_params):
    return decode_before_sum_tf_function(encoded_x, decode_before_sum_params)

  part_decoded_x_type = tmp_decode_before_sum_fn.type_signature.result
  del tmp_decode_before_sum_fn  # Only needed for result type.

  @computations.tf_computation(part_decoded_x_type,
                               decode_after_sum_params_type)
  def decode_after_sum_fn(summed_values, decode_after_sum_params):
    part_decoded_aggregated_x, num_summands = summed_values
    return tree.map_structure_up_to(
        encoders,
        lambda e, x, params: e.decode_after_sum(x, params, num_summands),
        encoders, part_decoded_aggregated_x, decode_after_sum_params)

  @computations.tf_computation(state_type, state_update_tensors_type)
  def update_state_fn(state, state_update_tensors):
    return tree.map_structure_up_to(encoders,
                                    lambda e, *args: e.update_state(*args),
                                    encoders, state, state_update_tensors)

  # Computations for intrinsics.federated_aggregate.
  @computations.tf_computation
  def zero_fn():
    values = tf.nest.map_structure(
        lambda s: tf.zeros(s.shape, s.dtype),
        type_conversions.type_to_tf_tensor_specs(part_decoded_x_type))
    state_update_tensors = tf.nest.map_structure(
        lambda s: tf.zeros(s.shape, s.dtype),
        type_conversions.type_to_tf_tensor_specs(state_update_tensors_type))
    return _accumulator_value(values, state_update_tensors)

  accumulator_type = zero_fn.type_signature.result
  state_update_aggregation_modes = tf.nest.map_structure(
      lambda e: tuple(e.state_update_aggregation_modes), encoders)

  @computations.tf_computation(accumulator_type,
                               encode_fn.type_signature.result)
  def accumulate_fn(acc, encoded_x):
    """Internal accumulate function."""
    value, params, state_update_tensors = encoded_x
    part_decoded_value = decode_before_sum_tf_function(value, params)
    new_values = tf.nest.map_structure(tf.add, acc['values'],
                                       part_decoded_value)
    new_state_update_tensors = tf.nest.map_structure(
        _accmulate_state_update_tensor, acc['state_update_tensors'],
        state_update_tensors, state_update_aggregation_modes)
    return _accumulator_value(new_values, new_state_update_tensors)

  @computations.tf_computation(accumulator_type, accumulator_type)
  def merge_fn(acc1, acc2):
    new_values = tf.nest.map_structure(tf.add, acc1['values'], acc2['values'])
    new_state_update_tensors = tf.nest.map_structure(
        _accmulate_state_update_tensor, acc1['state_update_tensors'],
        acc2['state_update_tensors'], state_update_aggregation_modes)
    return _accumulator_value(new_values, new_state_update_tensors)

  @computations.tf_computation(accumulator_type)
  def report_fn(acc):
    return acc

  return _NestGatherEncoder(
      get_params_fn=get_params_fn,
      encode_fn=encode_fn,
      decode_after_sum_fn=decode_after_sum_fn,
      update_state_fn=update_state_fn,
      zero_fn=zero_fn,
      accumulate_fn=accumulate_fn,
      merge_fn=merge_fn,
      report_fn=report_fn)


def _validate_encoder(encoder, value, encoder_type):
  assert encoder_type in [
      tensor_encoding.core.SimpleEncoder, tensor_encoding.core.GatherEncoder
  ]
  if not isinstance(encoder, encoder_type):
    raise TypeError('Provided encoder must be an instance of %s.' %
                    encoder_type)
  if not encoder.input_tensorspec.is_compatible_with(
      tf.TensorSpec(value.shape, value.dtype)):
    raise TypeError('Provided encoder and value are not compatible.')


def _validate_value_type_and_encoders(value_type, encoders, encoder_type):
  """Validates if `value_type` and `encoders` are compatible."""
  if isinstance(encoders, _ALLOWED_ENCODERS):
    # If `encoders` is not a container, then `value_type` should be an instance
    # of `tff.TensorType.`
    if not isinstance(value_type, computation_types.TensorType):
      raise ValueError(
          '`value_type` and `encoders` do not have the same structure.')

    _validate_encoder(encoders, value_type, encoder_type)
  else:
    # If `encoders` is a container, then `value_type` should be an instance of
    # `tff.StructType.`
    if not isinstance(value_type, computation_types.StructType):
      raise TypeError('`value_type` is not compatible with the expected input '
                      'of the `encoders`.')
    structure.map_structure(lambda e, v: _validate_encoder(e, v, encoder_type),
                            structure.from_container(encoders, recursive=True),
                            value_type)


def _accmulate_state_update_tensor(a, b, mode):
  """Accumulates state_update_tensors according to aggregation mode."""
  if mode == tensor_encoding.core.StateAggregationMode.SUM:
    return a + b
  elif mode == tensor_encoding.core.StateAggregationMode.MIN:
    return tf.minimum(a, b)
  elif mode == tensor_encoding.core.StateAggregationMode.MAX:
    return tf.maximum(a, b)
  elif mode == tensor_encoding.core.StateAggregationMode.STACK:
    raise NotImplementedError(
        'StateAggregationMode.STACK is not supported yet.')
  else:
    raise ValueError('Not supported state aggregation mode: %s' % mode)


def _accumulator_value(values, state_update_tensors):
  return collections.OrderedDict([('values', values),
                                  ('state_update_tensors', state_update_tensors)
                                 ])
