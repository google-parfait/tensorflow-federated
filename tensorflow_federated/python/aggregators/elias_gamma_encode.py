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
import tensorflow as tf
import tensorflow_compression as tfc

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


def get_bitstring_length(value):
  """Return size (in bits) of encoded value."""
  return 8. * tf.cast(tf.strings.length(value, unit="BYTE"), dtype=tf.float64)


class EliasGammaEncodedSumFactory(factory.UnweightedAggregationFactory):
  """Aggregator that encodes input integer tensor elements.

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
  records the average number of encoded bits sent from `tff.CLIENT` to
  `tff.SERVER` in `measurements` as a dictionary with key `avg_bitrate`.
  """

  def create(self, value_type):
    if not type_analysis.is_structure_of_integers(
        value_type) or not value_type.is_tensor():
      raise ValueError("Expect value_type to be an integer tensor, "
                       f"found {value_type}.")

    def sum_encoded_value(value):

      @computations.tf_computation
      def get_accumulator():
        return tf.zeros(shape=value_type.shape, dtype=tf.int32)

      @computations.tf_computation
      def decode_accumulate_values(accumulator, encoded_value):
        decoded_value = tfc.run_length_gamma_decode(
            code=encoded_value, shape=value_type.shape)
        return accumulator + decoded_value

      @computations.tf_computation
      def merge_decoded_values(decoded_value_1, decoded_value_2):
        return decoded_value_1 + decoded_value_2

      @computations.tf_computation
      def report_decoded_summation(summed_decoded_values):
        return summed_decoded_values

      return intrinsics.federated_aggregate(
          value,
          zero=get_accumulator(),
          accumulate=decode_accumulate_values,
          merge=merge_decoded_values,
          report=report_decoded_summation)

    @computations.federated_computation()
    def init_fn():
      return intrinsics.federated_value((), placements.SERVER)

    @computations.federated_computation(init_fn.type_signature.result,
                                        computation_types.at_clients(value_type)
                                       )
    def next_fn(state, value):
      encoded_value = intrinsics.federated_map(
          computations.tf_computation(
              lambda x: tfc.run_length_gamma_encode(data=x)), value)
      bitstring_lengths = intrinsics.federated_map(
          computations.tf_computation(get_bitstring_length), encoded_value)
      avg_bitstring_length = intrinsics.federated_mean(bitstring_lengths)
      num_elements = intrinsics.federated_mean(
          intrinsics.federated_map(
              computations.tf_computation(
                  lambda x: tf.size(x, out_type=tf.float64)), value))
      avg_bitrate = intrinsics.federated_map(
          computations.tf_computation(
              lambda x, y: tf.math.divide_no_nan(x, y, name="tff_divide")),
          (avg_bitstring_length, num_elements))
      decoded_value = sum_encoded_value(encoded_value)
      return measured_process.MeasuredProcessOutput(
          state=state,
          result=decoded_value,
          measurements=intrinsics.federated_zip(
              collections.OrderedDict(avg_bitrate=avg_bitrate)))

    return aggregation_process.AggregationProcess(init_fn, next_fn)
