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
"""Iterative process for quantile estimation."""

import tensorflow as tf
import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.templates import iterative_process


class PrivateQuantileEstimatorProcess(iterative_process.IterativeProcess):
  """A `tff.templates.IterativeProcess for estimating private quantiles.

  This iterative process uses a `tensorflow_privacy.QuantileEstimatorQuery` to
  maintain an estimate of the target quantile that is updated after each round.
  The `next` function has the following type signature:

  (<{state_type}@SERVER,{float32}@CLIENTS> -> {state_type}@SERVER)

  Given a `state` of type `state_type`, the most recent estimate of the target
  quantile can be retrieved using `get_current_estimate(state)`.
  """

  def __init__(
      self,
      quantile_estimator_query: tfp.QuantileEstimatorQuery,
      record_aggregation_factory: factory.AggregationProcessFactory = None):
    """Initializes `PrivateQuantileEstimatorProcess`.

    Args:
      quantile_estimator_query: A `tfp.QuantileEstimatorQuery` for estimating
        quantiles with differential privacy.
      record_aggregation_factory: A `tff.aggregators.AggregationProcessFactory`
        to aggregate counts of values below the current estimate. If `None`,
        defaults to `tff.aggregators.SumFactory`.
    """
    py_typecheck.check_type(quantile_estimator_query,
                            tfp.QuantileEstimatorQuery)
    if record_aggregation_factory is None:
      record_aggregation_factory = sum_factory.SumFactory()
    else:
      py_typecheck.check_type(record_aggregation_factory,
                              factory.AggregationProcessFactory)

    # 1. Define tf_computations.
    initial_state_fn = computations.tf_computation(
        quantile_estimator_query.initial_global_state)
    quantile_state_type = initial_state_fn.type_signature.result
    derive_sample_params = computations.tf_computation(
        quantile_estimator_query.derive_sample_params, quantile_state_type)
    get_quantile_record = computations.tf_computation(
        quantile_estimator_query.preprocess_record,
        derive_sample_params.type_signature.result, tf.float32)
    quantile_record_type = get_quantile_record.type_signature.result
    get_noised_result = computations.tf_computation(
        quantile_estimator_query.get_noised_result, quantile_record_type,
        quantile_state_type)
    quantile_agg_process = record_aggregation_factory.create(
        quantile_record_type)

    # 2. Define federated_computations.
    @computations.federated_computation()
    def init_fn():
      return intrinsics.federated_zip(
          (intrinsics.federated_eval(initial_state_fn, placements.SERVER),
           quantile_agg_process.initialize()))

    @computations.federated_computation(init_fn.type_signature.result,
                                        computation_types.FederatedType(
                                            tf.float32, placements.CLIENTS))
    def next_fn(state, value):
      quantile_query_state, agg_state = state

      params = intrinsics.federated_broadcast(
          intrinsics.federated_map(derive_sample_params, quantile_query_state))
      quantile_record = intrinsics.federated_map(get_quantile_record,
                                                 (params, value))

      (new_agg_state, agg_result,
       agg_measurements) = quantile_agg_process.next(agg_state, quantile_record)

      # We expect the quantile record aggregation process to be something simple
      # like basic sum, so we won't surface its measurements.
      del agg_measurements

      _, new_quantile_query_state = intrinsics.federated_map(
          get_noised_result, (agg_result, quantile_query_state))

      return intrinsics.federated_zip((new_quantile_query_state, new_agg_state))

    super().__init__(init_fn, next_fn)

    self._current_estimate_fn = computations.tf_computation(
        lambda state: state[0].current_estimate,
        init_fn.type_signature.result.member)

  @property
  def get_current_estimate(self):
    """A `tff.Computation` that computes the current estimate from `state`.

    Given a `state` controlled by this process, computes and returns the most
    recent estimate of the given quantile.

    Note that this computation operates on types without placements, and thus
    can be used with `state` residing either on `SERVER` or `CLIENTS`.

    Returns:
      A `tff.Computation`.
    """
    return self._current_estimate_fn
