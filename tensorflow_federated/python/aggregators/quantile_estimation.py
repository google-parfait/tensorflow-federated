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

from typing import Optional

import numpy as np
import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import secure
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import estimation_process


class PrivateQuantileEstimationProcess(estimation_process.EstimationProcess):
  """A `tff.templates.EstimationProcess` for estimating private quantiles.

  This iterative process uses a `tensorflow_privacy.QuantileEstimatorQuery` to
  maintain an estimate of the target quantile that is updated after each round.
  The `next` function has the following type signature:

  (<{state_type}@SERVER,{float32}@CLIENTS> -> {state_type}@SERVER)

  Given a `state` of type `state_type`, the most recent estimate of the target
  quantile can be retrieved using `report(state)`.
  """

  @classmethod
  def no_noise(
      cls,
      initial_estimate: float,
      target_quantile: float,
      learning_rate: float,
      multiplier: float = 1.0,
      increment: float = 0.0,
      secure_estimation: bool = False,
  ):
    """No-noise estimator for affine function of value at quantile.

    Estimates value `C` at `q`'th quantile of input distribution and reports
    `rC + i` for multiplier `r` and increment `i`. The quantile `C` is estimated
    using the geometric method described in Andrew, Thakkar et al. 2021,
    "Differentially Private Learning with Adaptive Clipping"
    (https://arxiv.org/abs/1905.03871) without noise added.

    Note that this estimator does not add noise, so it does not guarantee
    differential privacy. It is useful for estimating quantiles in contexts
    where rigorous privacy guarantees are not needed.

    Args:
      initial_estimate: The initial estimate of `C`.
      target_quantile: The quantile `q` to which `C` will be adapted.
      learning_rate: The learning rate for the adaptive algorithm.
      multiplier: The multiplier `r` of the affine transform.
      increment: The increment `i` of the affine transform.
      secure_estimation: Whether to perform the aggregation for estimation using
        `tff.aggregators.SumFactory` (if `False`; default) or
        `tff.aggregators.SecureSumFactory` (if `True`).

    Returns:
      An `EstimationProcess` whose `report` function returns `rC + i`.
    """
    _check_float_positive(initial_estimate, 'initial_estimate')
    _check_float_probability(target_quantile, 'target_quantile')
    _check_float_positive(learning_rate, 'learning_rate')
    _check_float_positive(multiplier, 'multiplier')
    _check_float_nonnegative(increment, 'increment')

    if secure_estimation:
      # NoPrivacyQuantileEstimatorQuery aggregates +/-0.5 values as the record,
      # and a constant 1.0 as weights for the average, thus the bound of 1.0.
      record_aggregation_factory = secure.SecureSumFactory(1.0)
    else:
      record_aggregation_factory = sum_factory.SumFactory()

    quantile = cls(
        tfp.NoPrivacyQuantileEstimatorQuery(
            initial_estimate=initial_estimate,
            target_quantile=target_quantile,
            learning_rate=learning_rate,
            geometric_update=True,
        ),
        record_aggregation_factory,
    )
    if multiplier == 1.0 and increment == 0.0:
      return quantile
    else:
      return quantile.map(_affine_transform(multiplier, increment))

  def __init__(
      self,
      quantile_estimator_query: tfp.QuantileEstimatorQuery,
      record_aggregation_factory: Optional[
          factory.UnweightedAggregationFactory
      ] = None,
  ):
    """Initializes `PrivateQuantileEstimationProcess`.

    Args:
      quantile_estimator_query: A `tfp.QuantileEstimatorQuery` for estimating
        quantiles with differential privacy.
      record_aggregation_factory: An optional
        `tff.aggregators.UnweightedAggregationFactory` to aggregate counts of
        values below the current estimate. If `None`, defaults to
        `tff.aggregators.SumFactory`.
    """
    py_typecheck.check_type(
        quantile_estimator_query, tfp.QuantileEstimatorQuery
    )
    if record_aggregation_factory is None:
      record_aggregation_factory = sum_factory.SumFactory()
    else:
      py_typecheck.check_type(
          record_aggregation_factory, factory.UnweightedAggregationFactory
      )

    # 1. Define tf_computations.
    initial_state_fn = tensorflow_computation.tf_computation(
        quantile_estimator_query.initial_global_state
    )
    quantile_state_type = initial_state_fn.type_signature.result
    derive_sample_params = tensorflow_computation.tf_computation(
        quantile_estimator_query.derive_sample_params, quantile_state_type
    )
    get_quantile_record = tensorflow_computation.tf_computation(
        quantile_estimator_query.preprocess_record,
        derive_sample_params.type_signature.result,
        np.float32,
    )
    quantile_record_type = get_quantile_record.type_signature.result
    get_noised_result = tensorflow_computation.tf_computation(
        quantile_estimator_query.get_noised_result,
        quantile_record_type,
        quantile_state_type,
    )
    quantile_agg_process = record_aggregation_factory.create(
        quantile_record_type
    )

    # 2. Define federated_computations.
    @federated_computation.federated_computation()
    def init_fn():
      return intrinsics.federated_zip((
          intrinsics.federated_eval(initial_state_fn, placements.SERVER),
          quantile_agg_process.initialize(),
      ))

    @federated_computation.federated_computation(
        init_fn.type_signature.result,
        computation_types.FederatedType(np.float32, placements.CLIENTS),
    )
    def next_fn(state, value):
      quantile_query_state, agg_state = state

      params = intrinsics.federated_broadcast(
          intrinsics.federated_map(derive_sample_params, quantile_query_state)
      )
      quantile_record = intrinsics.federated_map(
          get_quantile_record, (params, value)
      )

      quantile_agg_output = quantile_agg_process.next(
          agg_state, quantile_record
      )

      _, new_quantile_query_state, _ = intrinsics.federated_map(
          get_noised_result, (quantile_agg_output.result, quantile_query_state)
      )

      return intrinsics.federated_zip(
          (new_quantile_query_state, quantile_agg_output.state)
      )

    report_fn = federated_computation.federated_computation(
        lambda state: state[0].current_estimate, init_fn.type_signature.result
    )

    super().__init__(init_fn, next_fn, report_fn)


def _affine_transform(multiplier, increment):
  transform_tf_comp = tensorflow_computation.tf_computation(
      lambda value: multiplier * value + increment, np.float32
  )
  return federated_computation.federated_computation(
      lambda value: intrinsics.federated_map(transform_tf_comp, value),
      computation_types.FederatedType(np.float32, placements.SERVER),
  )


def _check_float_positive(value, label):
  py_typecheck.check_type(value, float, label)
  if value <= 0:
    raise ValueError(f'{label} must be positive. Found {value}.')


def _check_float_nonnegative(value, label):
  py_typecheck.check_type(value, float, label)
  if value < 0:
    raise ValueError(f'{label} must be nonnegative. Found {value}.')


def _check_float_probability(value, label):
  py_typecheck.check_type(value, float, label)
  if not 0 <= value <= 1:
    raise ValueError(
        f'{label} must be between 0 and 1 (inclusive). Found {value}.'
    )
