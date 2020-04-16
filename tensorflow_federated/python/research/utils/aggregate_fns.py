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
"""General useful `StatefulAggregateFn`."""

import attr
import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_federated.python.common_libs import anonymous_tuple


@attr.s(auto_attribs=True, eq=False, frozen=True)
class ClipNormAggregateState(object):
  """Structure for `tff.utils.StatefulBroadcastFn`.

  Fields:
  -   `clip_norm`: A float. The clipping ratio.
  -   `max_norm`: A float. The maximum client global norm observed in a round.
  """
  clip_norm: float
  max_norm: float


def build_clip_norm_aggregate_fn(
    clip_norm: float) -> tff.utils.StatefulAggregateFn:
  """Returns `tff.utils.StatefulAggregateFn` that clips client deltas."""

  @tff.tf_computation
  def initialize_fn():
    return ClipNormAggregateState(
        clip_norm=tf.constant(clip_norm, tf.float32),
        max_norm=tf.zeros((), tf.float32))

  def next_fn(state, deltas, weights=None):

    @tff.tf_computation(deltas.type_signature.member, tf.float32)
    def clip_by_global_norm(delta, clip_norm):
      # TODO(b/123092620): Replace anonymous_tuple with tf.nest.
      delta = anonymous_tuple.from_container(delta)
      clipped, global_norm = tf.clip_by_global_norm(
          anonymous_tuple.flatten(delta), clip_norm)
      return anonymous_tuple.pack_sequence_as(delta, clipped), global_norm

    client_clip_norm = tff.federated_broadcast(state.clip_norm)
    clipped_deltas, client_norms = tff.federated_map(clip_by_global_norm,
                                                     (deltas, client_clip_norm))
    # clip_norm no-op update here but could be set using max_norm.
    next_state = ClipNormAggregateState(
        clip_norm=state.clip_norm,
        max_norm=tff.utils.federated_max(client_norms))
    return next_state, tff.federated_mean(clipped_deltas, weight=weights)

  return tff.utils.StatefulAggregateFn(initialize_fn, next_fn)
