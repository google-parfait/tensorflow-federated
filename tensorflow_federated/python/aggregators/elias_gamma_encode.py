# Copyright 2022, Google LLC.
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
"""A library for building aggregators that apply Elias gamma codes to client updates."""

import collections
from typing import Optional

import tensorflow as tf
import tensorflow_compression as tfc

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


@tensorflow_computation.tf_computation
def _get_bits(value):
  """Return size (in bits) of an encoded tensor."""
  return 8 * tf.strings.length(value, unit="BYTE")


def _is_int32_or_structure_of_int32s(type_spec: computation_types.Type) -> bool:
  """Tests whether or not `type_spec` only contains elements of type int32.

  Args:
    type_spec: `value_type` to test for compatibility with the
      `EliasGammaEncodedSumFactory`.

  Returns:
    `True` if `type_spec` is an int32 or a structure containing only other
    structures of int32s, otherwise `False`.
  """
  if type_spec.is_tensor():
    py_typecheck.check_type(type_spec.dtype, tf.dtypes.DType)
    return type_spec.dtype == tf.int32
  elif type_spec.is_struct():
    return all(
        _is_int32_or_structure_of_int32s(v)
        for _, v in structure.iter_elements(type_spec)
    )
  else:
    return False


class EliasGammaEncodedSumFactory(factory.UnweightedAggregationFactory):
  """`UnweightedAggregationFactory` encoding values using Elias Gamma code.

  The created `tff.templates.AggregationProcess` encodes the input tensor as a
  bitstring. The `value_type` is expected to be a Type of integer tensors,
  with the expecataion that it should have relatively large amount of zeros.
  Each value is encoded according to the following protocol.

  First, one more than the number of zeros preceding the first non-zero integer
  in the tensor is encoded using the Elias Gamma code, a universal code which
  maps positive integers to a bitstring representation. Next, the sign of the
  non-zero integer is encoded with a single bit. The magnitude of the integer is
  encoded using the Elias Gamma code. This process is repeated for the remaining
  elements of the integer tensor and the substrings are concatenated into a
  single bitstring.

  Information about the Elias Gamma code can be found here:
  https://ieeexplore.ieee.org/document/1055349. Notably, the Elias Gamma code
  is used to compress positive integers whose values are unbounded but for which
  smaller values are more likely to occur than larger values.

  The bitstrings are aggregated at `tff.SERVER` and decoded to the same shape as
  the original input integer tensors. This aggregator computes the sum over
  decoded client values at `tff.SERVER` and outputs the sum placed at
  `tff.SERVER`.

  The process returns an empty `state`, the summed client values in `result` and
  optionally records the average number of encoded bits sent from `tff.CLIENT`
  to `tff.SERVER` in `measurements` as a dictionary with key `avg_bitrate`
  using the `bitrate_mean_factory`, if provided.
  """

  def __init__(
      self,
      bitrate_mean_factory: Optional[
          factory.UnweightedAggregationFactory
      ] = None,
  ):
    """Constructor for EliasGammaEncodedSumFactory.

    Args:
      bitrate_mean_factory: A `tff.aggregators.UnweightedAggregationFactory`
        that is used to compute the average bitrate across clients in each
        round. Note that the aggregator `state` does not change between rounds.
        If this is set to `None`, then the `AggregationProcess` created through
        this factory returns empty `measurements`.
    """
    if bitrate_mean_factory is not None:
      py_typecheck.check_type(
          bitrate_mean_factory, factory.UnweightedAggregationFactory
      )
    self._bitrate_mean_factory = bitrate_mean_factory

  def create(self, value_type):
    if not _is_int32_or_structure_of_int32s(value_type):
      raise ValueError(
          "Expect value_type to be an int32 tensor or a structure "
          "containing only other structures of int32 tensors, "
          f"found {value_type}."
      )

    if self._bitrate_mean_factory is not None:
      bitrate_mean_process = self._bitrate_mean_factory.create(
          computation_types.to_type(tf.float64)
      )

    @tensorflow_computation.tf_computation(value_type)
    def encode(value):
      return tf.nest.map_structure(
          lambda x: tfc.run_length_gamma_encode(data=x), value
      )

    def sum_encoded_value(value):
      @tensorflow_computation.tf_computation
      def get_accumulator():
        return type_conversions.structure_from_tensor_type_tree(
            lambda x: tf.zeros(shape=x.shape, dtype=tf.int32), value_type
        )

      @tensorflow_computation.tf_computation
      def decode_accumulate_values(accumulator, encoded_value):
        shapes = type_conversions.structure_from_tensor_type_tree(
            lambda x: x.shape, value_type
        )
        return tf.nest.map_structure(
            lambda a, x, y: a + tfc.run_length_gamma_decode(code=x, shape=y),
            accumulator,
            encoded_value,
            shapes,
        )

      @tensorflow_computation.tf_computation
      def merge_decoded_values(decoded_value_1, decoded_value_2):
        return tf.nest.map_structure(
            tensorflow_computation.tf_computation(lambda x, y: x + y),
            decoded_value_1,
            decoded_value_2,
        )

      @tensorflow_computation.tf_computation
      def report_decoded_summation(summed_decoded_values):
        return summed_decoded_values

      return intrinsics.federated_aggregate(
          value,
          zero=get_accumulator(),
          accumulate=decode_accumulate_values,
          merge=merge_decoded_values,
          report=report_decoded_summation,
      )

    @federated_computation.federated_computation()
    def init_fn():
      return intrinsics.federated_value((), placements.SERVER)

    @federated_computation.federated_computation(
        init_fn.type_signature.result, computation_types.at_clients(value_type)
    )
    def next_fn(state, value):
      measurements = ()
      encoded_value = intrinsics.federated_map(encode, value)
      if self._bitrate_mean_factory is not None:

        @tensorflow_computation.tf_computation
        def get_num_elements():
          num_elements = type_conversions.structure_from_tensor_type_tree(
              lambda x: x.shape.num_elements(), value_type
          )
          return tf.cast(sum(tf.nest.flatten(num_elements)), tf.float64)

        num_elements = intrinsics.federated_eval(
            get_num_elements, placements.CLIENTS
        )

        @tensorflow_computation.tf_computation
        def struct_get_bits(x):
          return tf.cast(
              sum([_get_bits(t) for t in tf.nest.flatten(x)]), tf.float64
          )

        total_bits = intrinsics.federated_map(struct_get_bits, encoded_value)

        bitrates = intrinsics.federated_map(
            tensorflow_computation.tf_computation(
                lambda x, y: tf.math.divide_no_nan(x=x, y=y, name="divide")
            ),
            (total_bits, num_elements),
        )
        avg_bitrate = bitrate_mean_process.next(
            bitrate_mean_process.initialize(), bitrates
        ).result
        measurements = intrinsics.federated_zip(
            collections.OrderedDict(elias_gamma_code_avg_bitrate=avg_bitrate)
        )
      decoded_value = sum_encoded_value(encoded_value)
      return measured_process.MeasuredProcessOutput(
          state=state, result=decoded_value, measurements=measurements
      )

    return aggregation_process.AggregationProcess(init_fn, next_fn)
