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
"""Factory for aggregations parameterized by tensor_encoding Encoders."""

import collections
from typing import Callable

import tensorflow as tf
import tree

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te

_EncoderConstructor = Callable[[tf.TensorSpec], te.core.GatherEncoder]


# TODO(b/173001046): Enable parameterization by an inner aggregation factory.
# This will only be possible when the encoders fully commute with sum and
# requires different implementation of the core logic, not directly using
# `intrinsics.federated_aggregate`.
class EncodedSumFactory(factory.UnweightedAggregationFactory):
  """`UnweightedAggregationFactory` for encoded sum.

  The created `tff.templates.AggregationProcess` aggregates values placed at
  `CLIENTS` according to the provided `GatherEncoder` objects, and outputs the
  result placed at `SERVER`.

  A `GatherEncoder` defines encode and decode methods, which are applied by
  `CLIENTS` before aggregation and by `SERVER` after aggregation. Usually these
  are such that the encoded representation communicated from `CLIENTS` to
  `SERVER` is smaller than communicating the original value, possibly at a cost
  of communicating slightly inaccurate values. A common example is to apply
  quantization. In addition, `GatherEncoder` enables the decoding logic to be
  split based on which part of its decode part commutes with sum, which is
  automatically utilized by the implementation.

  NOTE: The current implementation does not allow for specification of an
  "inner" aggregation factory. This will be possible in a future version, when
  used together with a `GatherEncoder` of which the entire decode logic commutes
  with sum, as per its `fully_commutes_with_sum` property. Contributions
  welcome.
  """

  def __init__(self, encoder_fn: _EncoderConstructor):
    """Initializes `EncodedSumFactory`.

    This class is initialized with an `encoder_fn` function, which given a
    `tf.TensorSpec`, returns an instance of `GatherEncoder` which is to be used
    for encoding and aggregating a value matching the provided spec.

    An example where this pattern is practically useful, is when encoding a
    collection of values, such as all variables of a model, it is usually best
    to only encode variables with larger number of elements, as those with small
    number of elements are often more sensitive to inaccuracy, and can provide
    only relatively small gain in terms of compression.

    The `encoder_fn` will be used during the call to `create` of the factory,
    and applied based on the provided `value_type`.

    Args:
      encoder_fn: A one-arg callable, mapping a `tf.TensorSpec`, to a
        `GatherEncoder`.
    """
    py_typecheck.check_callable(encoder_fn)
    self._encoder_fn = encoder_fn

  @classmethod
  def quantize_above_threshold(cls, quantization_bits=8, threshold=20000):
    """Quantization of values with at least `threshold` elements.

    Given a `value_type` in the `create` method, this classmethod configures the
    `EncodedSumFactory` to apply uniform quantization to all instances of
    `tff.TensorType` in the `value_type` which have more than `threshold`
    elements.

    Precisely, for each tensor `t`, this operation corresponds to
    `t = round((t - min(t)) / (max(t) - min(t)) * (2**quantizaton_bits - 1))`.

    If a type does not have more than `threshold` elements, it is summed
    directly without being modified.

    Args:
      quantization_bits: A integer specifying the quantization bitwidth.
      threshold: A non-negative integer. Only tensors with more than this number
        of elements are quantized.

    Returns:
      An `EncodedSumFactory`.
    """
    _check_quantization_bits(quantization_bits)
    _check_threshold(threshold)

    def encoder_fn(value_spec):
      if value_spec.shape.num_elements() > threshold:
        return te.encoders.as_gather_encoder(
            te.encoders.uniform_quantization(quantization_bits), value_spec)
      return te.encoders.as_gather_encoder(te.encoders.identity(), value_spec)

    return cls(encoder_fn)

  def create(
      self,
      value_type: factory.ValueType) -> aggregation_process.AggregationProcess:
    py_typecheck.check_type(value_type, factory.ValueType.__args__)
    encoders = self._encoders_for_value_type(value_type)
    init_fn = _encoded_init_fn(encoders)
    next_fn = _encoded_next_fn(init_fn.type_signature.result, value_type,
                               encoders)
    return aggregation_process.AggregationProcess(init_fn, next_fn)

  def _encoders_for_value_type(self, value_type):
    encoders = None
    # Creates unused tf_computation to manipulate `value_type` without TFF
    # type system, for compatibility with tree package, used later.
    @computations.tf_computation(value_type)
    def unused_fn(value):
      nonlocal encoders
      value_specs = tf.nest.map_structure(
          lambda t: tf.TensorSpec(t.shape, t.dtype), value)
      encoders = tf.nest.map_structure(self._encoder_fn, value_specs)
      return value

    return encoders


def _encoded_init_fn(encoders):
  """Creates `init_fn` for the process returned by `EncodedSumFactory`.

  The state for the `EncodedSumFactory` is directly derived from the state of
  the `GatherEncoder` objects that parameterize the functionality.

  Args:
    encoders: A collection of `GatherEncoder` objects.

  Returns:
    A no-arg `tff.Computation` returning initial state for `EncodedSumFactory`.
  """
  init_fn_tf = computations.tf_computation(
      lambda: tf.nest.map_structure(lambda e: e.initial_state(), encoders))
  init_fn = computations.federated_computation(
      lambda: intrinsics.federated_eval(init_fn_tf, placements.SERVER))
  return init_fn


def _encoded_next_fn(server_state_type, value_type, encoders):
  """Creates `next_fn` for the process returned by `EncodedSumFactory`.

  The structure of the implementation is roughly as follows:
  * Extract params for encoding/decoding from state (`get_params_fn`).
  * Encode values to be aggregated, placed at clients (`encode_fn`).
  * Call `federated_aggregate` operator, with decoding of the part which does
    not commute with sum, placed in its `accumulate_fn` arg.
  * Finish decoding the summed value placed at server (`decode_after_sum_fn`).
  * Update the state placed at server (`update_state_fn`).

  Args:
    server_state_type: A `tff.Type` of the expected state placed at server.
    value_type: An unplaced `tff.Type` of the value to be aggregated.
    encoders: A collection of `GatherEncoder` objects.

  Returns:
    A `tff.Computation` for `EncodedSumFactory`, with the type signature of
    `(server_state_type, value_type@CLIENTS) ->
    MeasuredProcessOutput(server_state_type, value_type@SERVER, ()@SERVER)`
  """

  @computations.tf_computation(server_state_type.member)
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

  @computations.tf_computation(server_state_type.member,
                               state_update_tensors_type)
  def update_state_fn(state, state_update_tensors):
    return tree.map_structure_up_to(encoders,
                                    lambda e, *args: e.update_state(*args),
                                    encoders, state, state_update_tensors)

  # Computations for intrinsics.federated_aggregate.
  def _accumulator_value(values, state_update_tensors):
    return collections.OrderedDict(
        values=values, state_update_tensors=state_update_tensors)

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

  @computations.federated_computation(server_state_type,
                                      computation_types.at_clients(value_type))
  def next_fn(state, value):
    encode_params, decode_before_sum_params, decode_after_sum_params = (
        intrinsics.federated_map(get_params_fn, state))
    encode_params = intrinsics.federated_broadcast(encode_params)
    decode_before_sum_params = intrinsics.federated_broadcast(
        decode_before_sum_params)

    encoded_values = intrinsics.federated_map(
        encode_fn, [value, encode_params, decode_before_sum_params])

    aggregated_values = intrinsics.federated_aggregate(encoded_values,
                                                       zero_fn(), accumulate_fn,
                                                       merge_fn, report_fn)

    decoded_values = intrinsics.federated_map(
        decode_after_sum_fn,
        [aggregated_values.values, decode_after_sum_params])

    updated_state = intrinsics.federated_map(
        update_state_fn, [state, aggregated_values.state_update_tensors])

    empty_metrics = intrinsics.federated_value((), placements.SERVER)
    return measured_process.MeasuredProcessOutput(
        state=updated_state, result=decoded_values, measurements=empty_metrics)

  return next_fn


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


def _accmulate_state_update_tensor(a, b, mode):
  """Accumulates state_update_tensors according to aggregation mode."""
  if mode == te.core.StateAggregationMode.SUM:
    return a + b
  elif mode == te.core.StateAggregationMode.MIN:
    return tf.minimum(a, b)
  elif mode == te.core.StateAggregationMode.MAX:
    return tf.maximum(a, b)
  elif mode == te.core.StateAggregationMode.STACK:
    raise NotImplementedError(
        'StateAggregationMode.STACK is not supported yet.')
  else:
    raise ValueError('Not supported state aggregation mode: %s' % mode)


def _check_threshold(threshold):
  py_typecheck.check_type(threshold, int)
  if threshold < 0:
    raise ValueError(f'The threshold must be nonnegative. '
                     f'Provided threshold: {threshold}')


def _check_quantization_bits(quantization_bits):
  py_typecheck.check_type(quantization_bits, int)
  if not 1 <= quantization_bits <= 16:
    raise ValueError(f'The quantization_bits must be in range [1, 16]. '
                     f'Provided quantization_bits: {quantization_bits}')
