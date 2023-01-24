# Copyright 2022, The TensorFlow Federated Authors.
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
"""Factory for string aggregation using IBLT."""

import collections
from typing import Optional

import attr
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.analytics import data_processing
from tensorflow_federated.python.analytics.heavy_hitters.iblt import chunkers
from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_tensor
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process

# Convenience Aliases
_CharacterEncoding = chunkers.CharacterEncoding

DATASET_KEY = 'key'
DATASET_VALUE = 'value'


@attr.s(eq=False, frozen=True)
class ServerOutput:
  output_strings = attr.ib()
  string_values = attr.ib()
  num_not_decoded = attr.ib()


@tf.function
def _parse_client_dict(
    dataset: tf.data.Dataset, string_max_bytes: int
) -> tuple[tf.Tensor, tf.Tensor]:
  """Parses the dictionary in the input `dataset` to key and value lists.

  Args:
    dataset: A `tf.data.Dataset` that yields `OrderedDict`. In each
      `OrderedDict` there are two key, value pairs: `DATASET_KEY`: A `tf.string`
      representing a string in the dataset. `DATASET_VALUE`: A rank 1
      `tf.Tensor` with `dtype` `tf.int64` representing the value associate with
      the string.
    string_max_bytes: The maximum length of the strings in bytes. If any string
      is longer than `string_max_bytes`, a `ValueError` will be raised.

  Returns:
    input_strings: A rank 1 `tf.Tensor` containing the list of strings in
      `dataset`.
    string_values: A rank 2 `tf.Tensor` containing the values of
      `input_strings`.
  Raises:
    ValueError: If any string in `dataset` is longer than string_max_bytes.
  """
  parsed_dict = data_processing.to_stacked_tensor(dataset)
  input_strings = parsed_dict[DATASET_KEY]
  string_values = parsed_dict[DATASET_VALUE]
  tf.debugging.Assert(
      tf.math.logical_not(
          tf.math.reduce_any(
              tf.greater(tf.strings.length(input_strings), string_max_bytes)
          )
      ),
      data=[
          (
              'IbltFactory: Input strings must be truncated to'
              f' {string_max_bytes=}'
          ),
          input_strings,
      ],
      name='CHECK_STRING_LENGTH',
  )
  return input_strings, string_values


class IbltFactory(factory.UnweightedAggregationFactory):
  """Factory for string and values aggregation by IBLT."""

  def __init__(
      self,
      *,
      capacity: int,
      string_max_bytes: int,
      encoding: _CharacterEncoding = _CharacterEncoding.UTF8,
      repetitions: int,
      seed: int = 0,
      sketch_agg_factory: Optional[factory.UnweightedAggregationFactory] = None,
      value_tensor_agg_factory: Optional[
          factory.UnweightedAggregationFactory
      ] = None,
  ) -> None:
    """Initializes IbltFactory.

    Args:
      capacity: The capacity of the IBLT sketch. Must be positive.
      string_max_bytes: The maximum length in bytes of a string in the IBLT.
        Must be positive.
      encoding: The character encoding of the string data to encode. For
        non-character binary data or strings with unknown encoding, specify
        `CharacterEncoding.UNKNOWN`.
      repetitions: The number of repetitions in IBLT data structure (must be >=
        3). Must be at least `3`.
      seed: An integer seed for hash functions. Defaults to 0.
      sketch_agg_factory: (Optional) A `UnweightedAggregationFactory` specifying
        the value aggregation to sum IBLT sketches. Defaults to
        `tff.aggregators.SumFactory`. If `sketch_agg_factory` is set to a
        `tff.aggregators.SecureSumFactory`, then the `upper_bound_threshold`
        should be at least 2 ** 32 - 1.
      value_tensor_agg_factory: (Optional) A `UnweightedAggregationFactory`
        specifying the value aggregation to sum value tensors. Defaults to
        `tff.aggregators.SumFactory`. Note that when using `sketch_agg_factory`
        is set to a `tff.aggregators.SecureSumFactory`, the value to be summed
        might be clipped depends on the choices of  `upper_bound_threshold` and
        `lower_bound_threshold` parameters in `SecureSumFactory`.

    Raises:
      ValueError: if parameters don't meet expectations.
    """
    if capacity < 1:
      raise ValueError(f'capacity should be at least 1, got {capacity}')
    if string_max_bytes < 1:
      raise ValueError(
          f'string_max_bytes should be at least 1, got {string_max_bytes}'
      )
    if repetitions < 3:
      raise ValueError(f'repetitions should be at least 3, got {repetitions}')

    self._sketch_agg_factory = (
        sum_factory.SumFactory()
        if sketch_agg_factory is None
        else sketch_agg_factory
    )
    self._value_tensor_agg_factory = (
        sum_factory.SumFactory()
        if value_tensor_agg_factory is None
        else value_tensor_agg_factory
    )
    self._capacity = capacity
    self._string_max_bytes = string_max_bytes
    self._encoding = encoding
    self._repetitions = repetitions
    self._seed = seed

  def create(
      self, value_type: computation_types.SequenceType
  ) -> aggregation_process.AggregationProcess:
    """Creates an AggregationProcess using IBLT to aggregate strings.

    Args:
      value_type: A `tff.SequenceType` representing the type of the input
        dataset, must be compatible with the following `tff.Type`:
        tff.SequenceType( collections.OrderedDict([ (DATASET_KEY, tf.string),
        (DATASET_VALUE, tff.TensorType(shape=[None], dtype=tf.int64)), ]))

    Raises:
      ValueError: If `value_type` is not as expected.

    Returns:
      A `tff.templates.AggregationProcess` to aggregate strings and values
        associate with the strings.
    """
    expected_value_type = computation_types.SequenceType(
        collections.OrderedDict([
            (DATASET_KEY, tf.string),
            (
                DATASET_VALUE,
                computation_types.TensorType(shape=[None], dtype=tf.int64),
            ),
        ])
    )
    if not expected_value_type.is_assignable_from(value_type):
      raise ValueError(
          'value_shape must be compatible with '
          f'{expected_value_type}. Found {value_type} instead.'
      )
    self._value_shape = tuple(value_type.element[DATASET_VALUE].shape)

    @tensorflow_computation.tf_computation(value_type)
    def encode_iblt(dataset):
      """The TF computation to compute the IBLT frequency sketches."""
      input_strings, string_values = _parse_client_dict(
          dataset, self._string_max_bytes
      )
      iblt_encoder = iblt_tensor.IbltTensorEncoder(
          capacity=self._capacity,
          string_max_bytes=self._string_max_bytes,
          encoding=self._encoding,
          repetitions=self._repetitions,
          value_shape=self._value_shape,
          seed=self._seed,
      )
      return iblt_encoder.compute_iblt(input_strings, string_values)

    @tensorflow_computation.tf_computation(encode_iblt.type_signature.result)
    @tf.function
    def decode_iblt(sketch, value_tensor):
      """The TF computation to decode the strings and values from IBLT."""
      iblt_decoder = iblt_tensor.IbltTensorDecoder(
          iblt=sketch,
          iblt_values=value_tensor,
          value_shape=self._value_shape,
          capacity=self._capacity,
          string_max_bytes=self._string_max_bytes,
          encoding=self._encoding,
          repetitions=self._repetitions,
          seed=self._seed,
      )
      (output_strings, _, string_values, num_not_decoded) = (
          iblt_decoder.get_freq_estimates_tf()
      )

      return (output_strings, string_values, num_not_decoded)

    inner_aggregator_sketch = self._sketch_agg_factory.create(
        encode_iblt.type_signature.result[0]
    )
    inner_aggregator_value_tensor = self._value_tensor_agg_factory.create(
        encode_iblt.type_signature.result[1]
    )

    @federated_computation.federated_computation
    def init_fn():
      sketch_state = inner_aggregator_sketch.initialize()
      value_tensor_state = inner_aggregator_value_tensor.initialize()
      return intrinsics.federated_zip((sketch_state, value_tensor_state))

    @federated_computation.federated_computation(
        init_fn.type_signature.result, computation_types.at_clients(value_type)
    )
    def next_fn(state, dataset):
      sketch_state, value_tensor_state = state
      sketch, value_tensor = intrinsics.federated_map(encode_iblt, dataset)
      sketch_output = inner_aggregator_sketch.next(sketch_state, sketch)
      value_tensor_output = inner_aggregator_value_tensor.next(
          value_tensor_state, value_tensor
      )
      summed_sketch = sketch_output.result
      summed_value_tensor = value_tensor_output.result
      (output_strings, string_values, num_not_decoded) = (
          intrinsics.federated_map(
              decode_iblt, (summed_sketch, summed_value_tensor)
          )
      )
      result = intrinsics.federated_zip(
          ServerOutput(
              output_strings=output_strings,
              string_values=string_values,
              num_not_decoded=num_not_decoded,
          )
      )
      updated_state = intrinsics.federated_zip(
          (sketch_output.state, value_tensor_output.state)
      )
      updated_measurements = intrinsics.federated_zip(
          collections.OrderedDict(
              num_not_decoded=num_not_decoded,
              sketch=sketch_output.measurements,
              value_tensor=value_tensor_output.measurements,
          )
      )
      return measured_process.MeasuredProcessOutput(
          updated_state, result, updated_measurements
      )

    return aggregation_process.AggregationProcess(init_fn, next_fn)
