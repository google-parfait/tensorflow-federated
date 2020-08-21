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
"""Implement Various Defense Methods against Targeted Attack."""

import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_privacy


def build_stateless_mean():
  """Just tff.federated_mean with empty state, to use as a default."""
  return tff.utils.StatefulAggregateFn(
      initialize_fn=lambda: (),
      next_fn=lambda state, value, weight=None: (  # pylint: disable=g-long-lambda
          state, tff.federated_mean(value, weight=weight)))


def build_aggregate_and_clip(norm_bound):
  """Build a 'tff.utils.StatefulAggregateFn' that clip the aggregated update."""

  def aggregate_and_clip(global_state, value, weight=None):
    """Compute the weighted mean of values@CLIENTS and clip it and return an aggregated value @SERVER."""

    round_model_delta = tff.federated_mean(value, weight)
    value_type = value.type_signature.member

    @tff.tf_computation(value_type)
    @tf.function
    def clip_by_norm(gradient):
      """Clip the gradient by a certain l_2 norm."""

      delta_norm = tf.linalg.global_norm(tf.nest.flatten(gradient))

      if delta_norm < tf.cast(norm_bound, tf.float32):
        return gradient
      else:
        delta_mul_factor = tf.math.divide_no_nan(
            tf.cast(norm_bound, tf.float32), delta_norm)
        return tf.nest.map_structure(lambda g: g * delta_mul_factor, gradient)

    return global_state, tff.federated_map(clip_by_norm, round_model_delta)

  return tff.utils.StatefulAggregateFn(
      initialize_fn=lambda: (), next_fn=aggregate_and_clip)


def build_dp_aggregate(l2_norm, mul_factor, num_clients):
  """Build a 'tff.utils.StatefulAggregateFn' that aggregates the model deltas differentially privately."""

  query = tensorflow_privacy.GaussianAverageQuery(l2_norm, mul_factor,
                                                  num_clients)
  dp_aggregate_fn, _ = tff.utils.build_dp_aggregate(query)
  return dp_aggregate_fn
