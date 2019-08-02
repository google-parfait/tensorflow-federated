# Lint as: python3
# Copyright 2019, Krishna Pillutla and Sham M. Kakade and Zaid Harchaoui.
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

"""Simple implementation of the RFA Algorithm for robust aggregation."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from six.moves import range

import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.common_libs import py_typecheck


def build_stateless_robust_aggregation(
    model_type, num_communication_passes=5, tolerance=1e-6
):
  """ Create TFF function for robust aggregation.
  The robust aggregate is an approximate geometric median
  computed via the smoothed Weiszfeld algorithm.

  Args:
    model_type: tff typespec of quantity to be aggregated.
    num_communication_passes: number of communication rounds in the
      smoothed Weiszfeld algorithm (min. 1).
    tolerance: smoothing parameter of smoothed Weiszfeld algorithm.
      Default 1e-6.

  Return:
    An instance of `tff.utils.StatefulAggregateFn` which implements a
    (stateless) robust aggregate.
  """
  pytype_check.check_type(num_communication_passes, int)
  if num_communication_passes < 1:
    raise ValueError('Aggregation requires num_communication_passes >= 1')
  # TODO: client weights have been hardcoded as float32

  @tff.tf_computation(tf.float32, model_type, model_type)
  def update_weight_fn(weight, server_model, client_model):
    sqnorms = tf.nest.map_structure(lambda a, b: tf.norm(a-b)**2, server_model, client_model)
    sqnorm = tf.reduce_sum(list(six.itervalues(sqnorms)))
    return weight / tf.math.maximum(tolerance, tf.math.sqrt(sqnorm))


  client_model_type = tff.FederatedType(model_type, tff.CLIENTS)
  client_weight_type = tff.FederatedType(tf.float32, tff.CLIENTS)

  @tff.federated_computation(client_model_type, client_weight_type)
  def robust_aggregation_fn(value, weight):
    aggregate = tff.federated_mean(value, weight=weight)
    for _ in range(num_communication_passes-1):
      aggregate_at_client = tff.federated_broadcast(aggregate)
      updated_weight = tff.federated_map(
          update_weight_fn,
          (weight, aggregate_at_client, value)
      )
      aggregate = tff.federated_mean(value, weight=updated_weight)
    return aggregate

  return tff.utils.StatefulAggregateFn(
      initialize_fn=lambda: (),
      next_fn=lambda state, value, weight: (state, robust_aggregation_fn(value, weight))
  )


def build_robust_federated_aggregation_process(
        model_fn,
        num_communication_passes=5,
        tolerance=1e-6):
  """Builds the TFF computations for robust federated aggregation using the RFA Algorithm.
  Args:
    model_fn: A no-arg function that returns a `tff.learning.TrainableModel`.
    num_communication_passes: Number of communication passes for the smoothed
      Weiszfeld algorithm to compute the approximate geometric median.
      The default is 5 and it has to be an interger at least 1.
    tolerance: Tolerance for the smoothed Weiszfeld algorithm. Default 1e-6.
 
  Returns:
    A `tff.utils.IterativeProcess`.
  """
  # build throwaway model simply to infer types
  with tf.Graph().as_default():
    model_type = tff.framework.type_from_tensors(model_fn().weights.trainable)
  robust_aggregation_fn = build_stateless_robust_aggregation(
      model_type,
      num_communication_passes=num_communication_passes,
      tolerance=tolerance
  )
  return tff.learning.build_federated_averaging_process(
      model_fn, stateful_delta_aggregate_fn=robust_aggregation_fn)
