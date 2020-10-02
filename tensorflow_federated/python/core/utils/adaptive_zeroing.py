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
"""Measured Process for adaptive zeroing."""

from typing import Union

import attr
import tensorflow as tf
import tensorflow_privacy
import tree

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.templates import measured_process

ValueType = Union[computation_types.TensorType, computation_types.StructType]


@attr.s(auto_attribs=True, eq=False, frozen=True)
class AdaptiveZeroingMetrics(object):
  """Structure metrics returned by adaptive zeroing mean prcoess.

  Attributes:
    zeroing_threshold: A float. The value of the norm over which updates will
      be zeroed.
    num_zeroed: An integer. The number of updates that were zeroed this round.
  """
  zeroing_threshold: float
  num_zeroed: int


def build_adaptive_zeroing_mean_process(
    value_type: ValueType,
    initial_quantile_estimate: float,
    target_quantile: float,
    multiplier: float,
    increment: float,
    learning_rate: float,
    norm_order: float,
):
  """Builds `tff.templates.MeasuredProcess` for averaging with adaptive zeroing.

  The returned `MeasuredProcess` averages values after zeroing out any values
  whose norm is greater than `C * r + i` where C is adapted to approximate the
  q'th quantile of the distribution of value norms. Its `next` function has the
  following type signature:

  (<{state_type}@SERVER,{value_type}@CLIENTS,{float32}@CLIENTS> ->
   <state={state_type}@SERVER,result={value_type}@SERVER,
    measurements=AdaptiveZeroingMetrics@SERVER>)

  Args:
    value_type: The type of values to be averaged by the `MeasuredProcess`. Can
      be a `tff.TensorType` or a nested structure of `tff.StructType` that
      bottoms out in `tff.TensorType`.
    initial_quantile_estimate: The initial value of `C`.
    target_quantile: The target quantile `q`. The adaptive process ensures that
      `C` will approximate the `q`'th quantile of the distribution of value
      norms.
    multiplier: The multiplier `r` of the quantile estimate `C`.
    increment: The increment `i` in the computation of the zeroing threshold.
    learning_rate: The learning rate `l` for the adaptive process. If the
      observed fraction of values whose norm is less than `C` on a given round
      is `p`, then `C` will be updated according to `C *= exp(l * (q - p))`. It
      follows that the maximum possible update is multiplying or dividing by a
      factor of `exp(l)`.
    norm_order: The order of the norm. May be 1, 2, or np.inf.

  Returns:
    A `MeasuredProcess` implementing averaging values with adaptive zeroing with
    the type signature described above.
  """
  # Actually value_type can be any nested structure of StructType bottoming
  # out in TensorType, but we'll just verify this much here.
  py_typecheck.check_type(
      value_type, (computation_types.TensorType, computation_types.StructType))

  if isinstance(value_type, computation_types.StructType):
    if not value_type:
      raise ValueError('value_type cannot be empty.')

  quantile_query = tensorflow_privacy.NoPrivacyQuantileEstimatorQuery(
      initial_estimate=initial_quantile_estimate,
      target_quantile=target_quantile,
      learning_rate=learning_rate,
      geometric_update=True)

  assert isinstance(quantile_query, tensorflow_privacy.SumAggregationDPQuery)

  @computations.tf_computation
  def initial_state_fn():
    return quantile_query.initial_global_state()

  @computations.federated_computation()
  def initial_state_comp():
    return intrinsics.federated_eval(initial_state_fn, placements.SERVER)

  global_state_type = initial_state_fn.type_signature.result

  @computations.tf_computation(global_state_type)
  def derive_sample_params(global_state):
    return quantile_query.derive_sample_params(global_state)

  @computations.tf_computation(derive_sample_params.type_signature.result,
                               value_type, tf.float32)
  def preprocess_value(params, value, weight):
    vectors = tree.map_structure(lambda v: tf.reshape(v, [-1]), value)
    norm = tf.norm(tf.concat(tree.flatten(vectors), axis=0), ord=norm_order)
    quantile_record = quantile_query.preprocess_record(params, norm)

    threshold = params.current_estimate * multiplier + increment
    too_large = (norm > threshold)
    adj_weight = tf.cond(too_large, lambda: tf.constant(0.0), lambda: weight)

    weighted_value = tree.map_structure(
        lambda v: tf.math.multiply_no_nan(v, adj_weight), value)

    too_large = tf.cast(too_large, tf.int32)
    return weighted_value, adj_weight, quantile_record, too_large

  quantile_record_type = preprocess_value.type_signature.result[2]

  @computations.tf_computation(quantile_record_type, global_state_type)
  def next_quantile(quantile_sum, global_state):
    new_estimate, new_global_state = quantile_query.get_noised_result(
        quantile_sum, global_state)
    new_threshold = new_estimate * multiplier + increment
    return new_threshold, new_global_state

  @computations.tf_computation(value_type, tf.float32)
  def divide_no_nan(value_sum, total_weight):
    return tree.map_structure(lambda v: tf.math.divide_no_nan(v, total_weight),
                              value_sum)

  @computations.federated_computation(
      initial_state_comp.type_signature.result,
      computation_types.FederatedType(value_type, placements.CLIENTS),
      computation_types.FederatedType(tf.float32, placements.CLIENTS))
  def next_fn(global_state, value, weight):
    sample_params = intrinsics.federated_broadcast(
        intrinsics.federated_map(derive_sample_params, global_state))
    weighted_value, adj_weight, quantile_record, too_large = (
        intrinsics.federated_map(preprocess_value,
                                 (sample_params, value, weight)))

    value_sum = intrinsics.federated_sum(weighted_value)
    total_weight = intrinsics.federated_sum(adj_weight)
    quantile_sum = intrinsics.federated_sum(quantile_record)
    num_zeroed = intrinsics.federated_sum(too_large)

    mean_value = intrinsics.federated_map(divide_no_nan,
                                          (value_sum, total_weight))

    new_threshold, new_global_state = intrinsics.federated_map(
        next_quantile, (quantile_sum, global_state))

    measurements = intrinsics.federated_zip(
        AdaptiveZeroingMetrics(new_threshold, num_zeroed))

    return measured_process.MeasuredProcessOutput(
        state=new_global_state, result=mean_value, measurements=measurements)

  return measured_process.MeasuredProcess(
      initialize_fn=initial_state_comp, next_fn=next_fn)
