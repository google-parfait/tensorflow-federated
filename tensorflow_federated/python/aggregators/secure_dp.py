# Copyright 2021, Google LLC.
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
"""A wrapper tff.aggregator for secure central and distributed DP."""

import collections
import enum
import functools
import math
from typing import Optional, Union
import warnings

import attr
from attr import converters
from attr import validators
import numpy as np
import tensorflow as tf
import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import concat
from tensorflow_federated.python.aggregators import differential_privacy
from tensorflow_federated.python.aggregators import discretization
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import modular_clipping
from tensorflow_federated.python.aggregators import quantile_estimation
from tensorflow_federated.python.aggregators import robust
from tensorflow_federated.python.aggregators import rotation
from tensorflow_federated.python.aggregators import secure
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


class DDPMechanism(enum.Enum):
  GAUSSIAN = 'distributed_dgauss'
  SKELLAM = 'skellam'


class RotationType(enum.Enum):
  DFT = 'dft'
  HD = 'hd'


@attr.s(auto_attribs=True)
class StaticClipArgs:
  """Arguments for specifying Static Tree Clipping in DP.

  Attributes:
    l2_clip: A float specifying the fixed l2 clipping norm.
  """

  # The fixed clipping norm.
  l2_clip: float = attr.ib(converter=float, validator=attr.validators.gt(0))


def _check_in_range(value, label, left, right, left_inclusive, right_inclusive):
  """Checks that a scalar value is in specified range."""
  _check_scalar(value, label)
  _check_bool(left_inclusive, 'left_inclusive')
  _check_bool(right_inclusive, 'right_inclusive')
  if left > right:
    raise ValueError(f'left must be smaller than right; found {left}, {right}.')
  left_cond = value >= left if left_inclusive else value > left
  right_cond = value <= right if right_inclusive else value < right
  if left_inclusive:
    left_bracket = '['
  else:
    left_bracket = '('

  if right_inclusive:
    right_bracket = ']'
  else:
    right_bracket = ')'

  if not left_cond or not right_cond:
    raise ValueError(f'{label} should be between {left_bracket}{left}, '
                     f'{right}{right_bracket}. Got {value}.')


def _numeric_range_validator(left,
                             right,
                             left_inclusive=False,
                             right_inclusive=False):
  """Wraps _check_in_range to make it compatible with attr validators."""
  validator = functools.partial(
      _check_in_range,
      left=left,
      right=right,
      left_inclusive=left_inclusive,
      right_inclusive=right_inclusive)
  return lambda instance, attribute, value: validator(value, attribute.name)


@attr.s(auto_attribs=True)
class AdaptiveClipArgs:
  """Arguments for specifying Adaptive Tree Clipping in DP.

  Attributes:
    l2_clip: A float specifying the initial l2 clipping norm. This may change
      adaptively.
    restart_warmup: (Optional) int specifying the first tree restart after
      `restart_warmup` times of calling `next`. After the restart the estimated
      clip norm will be adopted.
    restart_frequency: (Optional) int specfying the number of times `next` will
      be called before a new estimated clip norm will be adopted.
    target_unclipped_quantile: A float specifying the desired quantile of
      updates that should be unclipped.
    learning_rate: A float specifying the learning rate for adapting the
      clipping norm adaptation. This uses geometric updating which means that
      the clipping norm will change by a maximum of exp(`learning_rate`) at each
      update.
    l2_clip_count_stddev: (Optional) float specfying the stddev of the noise
      added to the clipped counts in the adaptive clipping algorithm. If None,
      defaults to `0.05 * expected_clients_per_round` (unless `noise_multiplier`
      is 0, in which case it is also 0).
  """
  l2_clip: float = attr.ib(
      default=0.1, converter=float, validator=attr.validators.gt(0))
  restart_warmup: Optional[int] = attr.ib(
      default=None,
      converter=converters.optional(int),
      validator=validators.optional(validators.gt(0)))
  restart_frequency: Optional[int] = attr.ib(
      default=None,
      converter=converters.optional(int),
      validator=validators.optional(validators.gt(0)))
  target_unclipped_quantile: float = attr.ib(
      default=0.5,
      converter=float,
      validator=_numeric_range_validator(0, 1, right_inclusive=True))
  learning_rate: float = attr.ib(
      default=0.2, converter=float, validator=validators.gt(0))
  l2_clip_count_stddev: Optional[float] = attr.ib(
      default=None,
      converter=converters.optional(float),
      validator=validators.optional(validators.ge(0)))


@attr.s(auto_attribs=True)
class DistributedDPArgs:
  """Arguments for specfying Distrubted DP.

  Attributes:
    noise_multiplier: A float specifying the noise multiplier (central noise
      stddev / L2 clip norm) for model updates. Note that this is with respect
      to the initial L2 clip norm, and the quantization procedure as part of the
      DDP algorithm may inflate the L2 sensitivity. The specified noise will be
      split into `expected_clients_per_round` noise shares to be added locally
      on the clients. A value of 1.0 or higher may be needed for strong privacy.
      Must be nonnegative. A value of 0.0 means no noise will be added.
    bits: An int specifying the communication bit-width (B) which must be in the
      inclusive range [1, 22]. The field size for SecAgg is thus 2^B. Note that
      this is for the noisy quantized aggregate at the server and thus should
      account for the number of clients. Should be at least as large as
      log_2(expected_clients_per_round).
    mechanism: A DDPMechanism specifying the underlying distributed
      differentially private query mechanism to use.
    rotation_type: A RotationType specifying the rotation operation, used to
      spread out input values across vector dimensions.
    beta: A float representing the conditional randomized rounding bias. Must be
      in the range [0.0, 1.0). The larger the value, the less post-round L2
      sensitivity inflation. For a defailted explanation, see Section 4 of
      https://arxiv.org/pdf/2102.06387.pdf.
    modclip_prob: A float specifying the target probability in the exclusive
      range (0, 1) for modular wrapping due to SecAgg's modulo operations. The
      default corresponds to 0.01% or roughly 3.9 standard deviations of the
      mean, assuming normally distributed aggregates at the server.
  """
  noise_multiplier: float = attr.ib(
      converter=float, validator=attr.validators.ge(0))
  bits: int = attr.ib(converter=int, validator=_numeric_range_validator(0, 23))
  mechanism: DDPMechanism = attr.ib(
      default=DDPMechanism.SKELLAM,
      converter=DDPMechanism,
      validator=attr.validators.in_(DDPMechanism))
  rotation_type: RotationType = attr.ib(
      default=RotationType.DFT,
      converter=RotationType,
      validator=attr.validators.in_(RotationType))
  beta: float = attr.ib(
      default=math.exp(-0.5),
      converter=float,
      validator=_numeric_range_validator(0, 1, left_inclusive=True))
  modclip_prob: float = attr.ib(
      default=1e-4, converter=float, validator=_numeric_range_validator(0, 1))


_ClipOptionsType = Union[StaticClipArgs, AdaptiveClipArgs]

# The maximum possible scaling factor before rounding is applied. This is needed
# because when the number of clients per round is small (<10) and/or the noise
# multiplier is very small (say < 0.001) and/or the number of bits is large (say
# > 18), the scale factor computed by `_heuristic_scale_factor(...)` becomes
# very large. For example, for a noise multiplier = 0, number of bits = 20, and
# 10 clients per round, the scaling factor is on the order of 1e8. Such a large
# scaling factor leads to overflows when computing the inflated and scaled l1/l2
# norm bounds using int32 bit representations. In practice, however, such very
# scaling factors do not offer any added value (in minimizing rounding errors
# and/or minimizing the inflation of the l1/l2 norms upon rounding). Capping the
# scaling factor to 1e6 avoids these overflow issues without compromising
# utility.
# TODO(b/223427213): Adapt the bitwidth whenever the scale is too high.
_MAX_SCALE_FACTOR = 1e6


class SecureDPFactory(factory.UnweightedAggregationFactory):
  """An `UnweightedAggregationFactory` for central/distributed DP with SecAgg.

  The created `tff.templates.AggregationProcess` serves as a wrapper that
  encapsulates several component tff.aggregators into an implementation of
  a central and/or distributed differential privacy (DP) algorithm with secure
  aggregation. This implemenation is streamlined so as to only clip once and
  efficiently implement all three configurations (only central,
  only distributed, and both).

  For central DP, we use DP-FTRL (see
  https://www.tensorflow.org/federated/api_docs/python/tff/aggregators/DifferentiallyPrivateFactory#tree_aggregation).
  This achieves state-of-the-art guarantees but assumes higher trust in the
  server.

  Instead, Distributed DP algorithms aim to "distribute trust" away from the
  central server by allowing clients to add their own noise for DP while
  allowing comparable privacy/utility to be achieved compared to most central DP
  models. See
  https://www.tensorflow.org/federated/api_docs/python/tff/learning/ddp_secure_aggregator.

  To obtain concrete (epsilon, delta) guarantees, one could use the analysis
  tools provided in tensorflow_privacy on the metrics generated in each round.
  """

  def __init__(self, expected_clients_per_round: int,
               distributed_dp_options: DistributedDPArgs,
               clip_options: _ClipOptionsType):
    """Initializes the `SecureDPFactory`.

    Note that the `create` method of this factory needs to be executed in TF
    eager mode.

    Args:
      expected_clients_per_round: An integer specifying the expected number of
        clients to participate in this round. This number dictates how much
        noise is added locally. In particular, local noise stddev = central
        noise stddev / `expected_clients_per_round`. Must be a positive integer.
      distributed_dp_options: Specifies the distributed dp options, which by
        default includes small (0.01 * central_dp_noise_multiplier) DDP
        guarantees. See documentation in DistributedDPArgs
      clip_options: Specifies the parameters for either fixed or adaptive
        clipping. See StaticClipArgs or AdaptiveClipArgs for documentation.

    Raises:
      TypeError: If arguments have the wrong type(s).
      ValueError: If arguments have invalid value(s).
    """
    _check_scalar(expected_clients_per_round, 'expected_clients_per_round')
    py_typecheck.check_type(expected_clients_per_round, int,
                            'expected_clients_per_round')
    _check_positive(expected_clients_per_round, 'expected_clients_per_round')
    self._num_clients = expected_clients_per_round

    self._distributed_dp_options = distributed_dp_options
    self._clip_options = clip_options

    bits = distributed_dp_options.bits
    self._k_stddevs = _clip_prob_to_num_stddevs(
        self._distributed_dp_options.modclip_prob)

    # Value range checks based on the client count and the clip probability.
    if bits < math.log2(expected_clients_per_round):
      raise ValueError('bits should be >= log2(expected_clients_per_round). '
                       f'Found 2^b = 2^{bits} < {expected_clients_per_round}.')
    if 2**(2 * bits) < expected_clients_per_round * self._k_stddevs**2:
      raise ValueError(f'The selected bit-width ({bits}) is too small for the '
                       f'given parameters (expected_clients_per_round = '
                       f'{expected_clients_per_round}, modclip_prob = '
                       f'{distributed_dp_options.modclip_prob}). You must'
                       f' decrease the `expected_clients_per_round`, increase'
                       f' `bits`, or increase `modclip_prob`.')

    if isinstance(self._clip_options, AdaptiveClipArgs):
      self._l2_clip, self._value_noise_mult = self._build_auto_l2_clip_process(
          self._clip_options)
    else:
      self._l2_clip = self._clip_options.l2_clip
      self._value_noise_mult = self._distributed_dp_options.noise_multiplier

  def _build_auto_l2_clip_process(self, clip_options: AdaptiveClipArgs):
    """Builds a `tff.templates.EstimationProcess` for adaptive L2 clipping.

    Specifically, we use the private quantile estimation algorithm described in
    https://arxiv.org/abs/1905.03871 for choosing the adaptive L2 clip norm.
    The default noise level for the procedure follows the paper and the
    implementation of `tff.aggregators.DifferentiallyPrivateFactory`.

    Note that for consistency with the use of secure aggregation for the client
    values, the binary flags as part of the quantile estimation procedure
    indicating whether client L2 norms are below the current estimate are also
    securely aggregated.

    Args:
      clip_options: Specifies the parameters for adaptive clipping. See
        AdaptiveClipArgs for documentation.

    Returns:
      The `EstimationProcess` for adaptive L2 clipping and the required noise
      multiplier for the record aggregation.
    """
    value_noise_mult, clip_count_stddev = (
        differential_privacy.adaptive_clip_noise_params(
            self._distributed_dp_options.noise_multiplier, self._num_clients,
            clip_options.l2_clip_count_stddev))

    estimator_query = tfp.QuantileEstimatorQuery(
        initial_estimate=clip_options.l2_clip,
        target_quantile=clip_options.target_unclipped_quantile,
        learning_rate=clip_options.learning_rate,
        below_estimate_stddev=clip_count_stddev,
        expected_num_records=self._num_clients,
        geometric_update=True)
    # Note also that according to https://arxiv.org/abs/1905.03871, the binary
    # flags for quantile estimation are shifted from [0, 1] to [-0.5, 0.5], so
    # we set the SecAgg input bounds accordingly.
    estimator_process = quantile_estimation.PrivateQuantileEstimationProcess(
        quantile_estimator_query=estimator_query,
        record_aggregation_factory=secure.SecureSumFactory(
            upper_bound_threshold=0.5, lower_bound_threshold=-0.5))

    return estimator_process, value_noise_mult

  def _build_aggregation_factory(self):
    bits = self._distributed_dp_options.bits
    central_stddev = self._value_noise_mult * self._clip_options.l2_clip
    local_stddev = central_stddev / math.sqrt(self._num_clients)

    # Ensure dim is at least 1 only for computing DDP parameters.
    self._client_dim = max(1, self._client_dim)
    if self._distributed_dp_options.rotation_type == RotationType.HD:
      # Hadamard transform requires dimension to be powers of 2.
      self._padded_dim = 2**math.ceil(math.log2(self._client_dim))
      rotation_factory = rotation.HadamardTransformFactory
    else:
      # DFT pads at most 1 zero.
      self._padded_dim = math.ceil(self._client_dim / 2.0) * 2
      rotation_factory = rotation.DiscreteFourierTransformFactory

    scale = _heuristic_scale_factor(local_stddev, self._clip_options.l2_clip,
                                    bits, self._num_clients, self._padded_dim,
                                    self._k_stddevs).numpy()

    # Very large scales could lead to overflows and are not as helpful for
    # utility. See comment above for more details.
    scale = min(scale, _MAX_SCALE_FACTOR)

    if scale <= 1:
      warnings.warn(f'The selected scale_factor {scale} <= 1. This may lead to'
                    f'substantial quantization errors. Consider increasing'
                    f'the bit-width (currently {bits}) or decreasing the'
                    f'expected number of clients per round (currently '
                    f'{self._num_clients}).')

    # The procedure for obtaining inflated L2 bound assumes eager TF execution
    # and can be rewritten with NumPy if needed.
    inflated_l2 = discretization.inflated_l2_norm_bound(
        l2_norm_bound=self._clip_options.l2_clip,
        gamma=1.0 / scale,
        beta=self._distributed_dp_options.beta,
        dim=self._padded_dim).numpy()

    # Add small leeway on norm bounds to gracefully allow numerical errors.
    # Specifically, the norm thresholds are computed directly from the specified
    # parameters in Python and will be checked right before noising; on the
    # other hand, the actual norm of the record (to be measured at noising time)
    # can possibly be (negligibly) higher due to the float32 arithmetic after
    # the conditional rounding (thus failing the check). While we have mitigated
    # this by sharing the computation for the inflated norm bound from
    # quantization, adding a leeway makes the execution more robust (it does not
    # need to abort should any precision issues happen) while not affecting the
    # correctness if privacy accounting is done based on the norm bounds at the
    # DPQuery/DPFactory (which incorporates the leeway).
    scaled_inflated_l2 = (inflated_l2 + 1e-5) * scale
    # Since values are scaled and rounded to integers, we have L1 <= L2^2
    # on top of the general of L1 <= sqrt(d) * L2.
    scaled_l1 = math.ceil(scaled_inflated_l2 *
                          min(math.sqrt(self._padded_dim), scaled_inflated_l2))

    # Build nested aggregtion factory.
    # 1. Secure Aggregation. In particular, we have 4 modular clips from
    #    nesting two modular clip aggregators:
    #    #1. outer-client: clips to [-2^(b-1), 2^(b-1)]
    #        Bounds the client values (with limited effect as scaling was
    #        chosen such that `num_clients` is taken into account).
    #    #2. inner-client: clips to [0, 2^b]
    #        Similar to applying a two's complement to the values such that
    #        frequent values (post-rotation) are now near 0 (representing small
    #        positives) and 2^b (small negatives). 0 also always map to 0, and
    #        we do not require another explicit value range shift from
    #        [-2^(b-1), 2^(b-1)] to [0, 2^b] to make sure that values are
    #        compatible with SecAgg's mod m = 2^b. This can be reverted at #4.
    #    #3. inner-server: clips to [0, 2^b]
    #        Ensures the aggregated value range does not grow by log_2(n).
    #        NOTE: If underlying SecAgg is implemented using the new
    #        `tff.federated_secure_modular_sum()` operator with the same
    #        modular clipping range, then this would correspond to a no-op.
    #    #4. outer-server: clips to [-2^(b-1), 2^(b-1)]
    #        Keeps aggregated values centered near 0 out of the logical SecAgg
    #        black box for outer aggregators.
    #    Note that the scaling factor and the bit-width are chosen such that
    #    the number of clients to aggregate is taken into account.
    nested_factory = secure.SecureSumFactory(
        upper_bound_threshold=2**bits - 1, lower_bound_threshold=0)
    nested_factory = modular_clipping.ModularClippingSumFactory(
        clip_range_lower=0,
        clip_range_upper=2**bits,
        inner_agg_factory=nested_factory)
    nested_factory = modular_clipping.ModularClippingSumFactory(
        clip_range_lower=-(2**(bits - 1)),
        clip_range_upper=2**(bits - 1),
        inner_agg_factory=nested_factory)

    # 2. DP operations. DP params are in the scaled domain (post-quantization).
    if self._distributed_dp_options.mechanism == DDPMechanism.GAUSSIAN:
      dp_query = tfp.DistributedDiscreteGaussianSumQuery(
          l2_norm_bound=scaled_inflated_l2, local_stddev=local_stddev * scale)
    else:
      dp_query = tfp.DistributedSkellamSumQuery(
          l1_norm_bound=scaled_l1,
          l2_norm_bound=scaled_inflated_l2,
          local_stddev=local_stddev * scale)

    nested_factory = differential_privacy.DifferentiallyPrivateFactory(
        query=dp_query, record_aggregation_factory=nested_factory)

    # 3. Discretization operations. This appropriately quantizes the inputs.
    nested_factory = discretization.DiscretizationFactory(
        inner_agg_factory=nested_factory,
        scale_factor=scale,
        stochastic=True,
        beta=self._distributed_dp_options.beta,
        prior_norm_bound=self._clip_options.l2_clip)

    # 4. L2 clip, possibly adaptively with a `tff.templates.EstimationProcess`.
    nested_factory = robust.clipping_factory(
        clipping_norm=self._l2_clip,
        inner_agg_factory=nested_factory,
        clipped_count_sum_factory=secure.SecureSumFactory(
            upper_bound_threshold=1, lower_bound_threshold=0))

    # 5. Flattening to improve quantization and reduce modular wrapping.
    nested_factory = rotation_factory(inner_agg_factory=nested_factory)

    # 6. Concat the input structure into a single vector.
    nested_factory = concat.concat_factory(inner_agg_factory=nested_factory)
    return nested_factory

  def _unpack_state(self, agg_state):
    # Note: `agg_state` has a nested structure similar to the composed
    # aggregator. Please print it to figure out how to correctly unpack the
    # needed states. This is especially needed when you add, remove, or change
    # any of the core composed aggregators.
    # TODO(b/222162205): Simplify how we compose states of nested aggregators.
    rotation_state = agg_state  # Concat has no states.
    l2_clip_state, _ = rotation_state
    discrete_state = l2_clip_state['inner_agg']
    dp_state = discrete_state['inner_agg_process']
    return l2_clip_state, discrete_state, dp_state

  def _unpack_measurements(self, agg_measurements):
    rotate_metrics = agg_measurements  # Concat has no measurements.
    l2_clip_metrics = rotate_metrics[
        self._distributed_dp_options.rotation_type.value]
    discrete_metrics = l2_clip_metrics['clipping']
    dp_metrics = discrete_metrics['discretize']
    return l2_clip_metrics, discrete_metrics, dp_metrics

  def _autotune_component_states(self, agg_state):
    """Updates the nested aggregator state in-place.

    This procedure makes the following assumptions: (1) this wrapper aggregator
    has knowledge about the states of the component aggregators and their
    Python containers, and can thus make in-place modifications directly; (2)
    this aggregator has knowledge about the state of the `DPQuery` objects
    (types and members) that are used by the `DifferentiallyPrivateFactory`, and
    can thus update the members directly. Both assumptions should be revisited.

    Args:
      agg_state: The state of this aggregator, which is a nested object
        containing the states of the component aggregators.

    Returns:
      The updated agg_state.
    """

    @tensorflow_computation.tf_computation
    def _update_scale(agg_state, new_l2_clip):
      _, discrete_state, _ = self._unpack_state(agg_state)
      new_central_stddev = new_l2_clip * self._value_noise_mult
      new_local_stddev = new_central_stddev / math.sqrt(self._num_clients)
      new_scale = _heuristic_scale_factor(new_local_stddev, new_l2_clip,
                                          self._distributed_dp_options.bits,
                                          self._num_clients, self._padded_dim,
                                          self._k_stddevs)

      # Very large scales could lead to overflows and are not as helpful for
      # utility. See comment above for more details.
      new_scale = tf.math.minimum(
          new_scale, tf.constant(_MAX_SCALE_FACTOR, dtype=tf.float64))

      discrete_state['scale_factor'] = tf.cast(new_scale, tf.float32)
      return agg_state

    @tensorflow_computation.tf_computation
    def _update_dp_params(agg_state, new_l2_clip):
      _, discrete_state, dp_state = self._unpack_state(agg_state)
      new_scale = discrete_state['scale_factor']
      new_inflated_l2 = discretization.inflated_l2_norm_bound(
          l2_norm_bound=new_l2_clip,
          gamma=1.0 / new_scale,
          beta=self._distributed_dp_options.beta,
          dim=self._padded_dim)
      # Similarly include a norm bound leeway. See inline comment in
      # `_build_aggregation_factory()` for more details.
      new_scaled_inflated_l2 = (new_inflated_l2 + 1e-5) * new_scale
      l1_fac = tf.minimum(math.sqrt(self._padded_dim), new_scaled_inflated_l2)
      new_scaled_l1 = tf.math.ceil(new_scaled_inflated_l2 * l1_fac)
      new_scaled_l1 = tf.cast(new_scaled_l1, tf.int32)
      # Recompute noise stddevs.
      new_central_stddev = new_l2_clip * self._value_noise_mult
      new_local_stddev = new_central_stddev / math.sqrt(self._num_clients)
      # Update DP params: norm bounds (uninflated/inflated) and local stddev.
      dp_query_state, dp_inner_agg_state = dp_state
      if self._distributed_dp_options.mechanism == DDPMechanism.GAUSSIAN:
        new_dp_query_state = dp_query_state._replace(
            l2_norm_bound=new_scaled_inflated_l2,
            local_stddev=new_local_stddev * new_scale)
      else:
        new_dp_query_state = dp_query_state._replace(
            l1_norm_bound=new_scaled_l1,
            l2_norm_bound=new_scaled_inflated_l2,
            local_stddev=new_local_stddev * new_scale)
      new_dp_state = (new_dp_query_state, dp_inner_agg_state)
      discrete_state['inner_agg_process'] = new_dp_state
      discrete_state['prior_norm_bound'] = new_l2_clip
      return agg_state

    l2_clip_state, _, _ = self._unpack_state(agg_state)
    # NOTE(b/170893510): Explicitly declaring Union[float, EstimationProcess]
    # for _l2_clip or doing isinstance() check still triggers attribute-error.
    new_l2_clip = self._l2_clip.report(l2_clip_state['clipping_norm'])  # pytype: disable=attribute-error
    agg_state = intrinsics.federated_map(_update_scale,
                                         (agg_state, new_l2_clip))
    agg_state = intrinsics.federated_map(_update_dp_params,
                                         (agg_state, new_l2_clip))
    return agg_state

  def _derive_measurements(self, agg_state, agg_measurements):
    _, discrete_state, dp_state = self._unpack_state(agg_state)
    l2_clip_metrics, _, dp_metrics = self._unpack_measurements(agg_measurements)
    dp_query_state, _ = dp_state

    actual_num_clients = intrinsics.federated_secure_sum_bitwidth(
        intrinsics.federated_value(1, placements.CLIENTS), bitwidth=1)
    padded_dim = intrinsics.federated_value(
        int(self._padded_dim), placements.SERVER)

    measurements = collections.OrderedDict(
        l2_clip=l2_clip_metrics['clipping_norm'],
        scale_factor=discrete_state['scale_factor'],
        scaled_inflated_l2=dp_query_state.l2_norm_bound,
        scaled_local_stddev=dp_query_state.local_stddev,
        actual_num_clients=actual_num_clients,
        padded_dim=padded_dim,
        dp_query_metrics=dp_metrics['dp_query_metrics'])

    return intrinsics.federated_zip(measurements)

  def create(self, value_type):
    # Checks value_type and compute client data dimension.
    if (value_type.is_struct_with_python() and
        type_analysis.is_structure_of_tensors(value_type)):
      num_elements_struct = type_conversions.structure_from_tensor_type_tree(
          lambda x: x.shape.num_elements(), value_type)
      self._client_dim = sum(tf.nest.flatten(num_elements_struct))
    elif value_type.is_tensor():
      self._client_dim = value_type.shape.num_elements()
    else:
      raise TypeError('Expected `value_type` to be `TensorType` or '
                      '`StructWithPythonType` containing only `TensorType`. '
                      f'Found type: {repr(value_type)}')
    # Checks that all values are integers or floats.
    if not (type_analysis.is_structure_of_floats(value_type) or
            type_analysis.is_structure_of_integers(value_type)):
      raise TypeError('Component dtypes of `value_type` must all be integers '
                      f'or floats. Found {repr(value_type)}.')

    ddp_agg_process = self._build_aggregation_factory().create(value_type)
    init_fn = ddp_agg_process.initialize

    @federated_computation.federated_computation(
        init_fn.type_signature.result, computation_types.at_clients(value_type))
    def next_fn(state, value):
      agg_output = ddp_agg_process.next(state, value)
      new_measurements = self._derive_measurements(agg_output.state,
                                                   agg_output.measurements)
      new_state = agg_output.state
      if isinstance(self._clip_options, AdaptiveClipArgs):
        new_state = self._autotune_component_states(agg_output.state)

      return measured_process.MeasuredProcessOutput(
          state=new_state,
          result=agg_output.result,
          measurements=new_measurements)

    return aggregation_process.AggregationProcess(init_fn, next_fn)


def _clip_prob_to_num_stddevs(clip_prob):
  """Computes the number of stddevs for the target clipping probability.

  This function assumes (approximately) normal distributions. It is implemented
  using TensorFlow to avoid depending on SciPy's `stats.norm.ppf` and it thus
  assumes eager TF execution. This can be replaced with the `statistics` package
  from Python >= 3.8.

  Args:
    clip_prob: A float for clipping probability in the exclusive range (0, 1).

  Returns:
    The number of standard deviations corresponding to the clip prob.
  """
  return math.sqrt(2) * tf.math.erfcinv(clip_prob).numpy()


def _heuristic_scale_factor(local_stddev,
                            l2_clip,
                            bits,
                            num_clients,
                            dim,
                            k_stddevs,
                            rho=1.0):
  """Selects a scaling factor by assuming subgaussian aggregates.

  Selects scale_factor = 1 / gamma such that k stddevs of the noisy, quantized,
  aggregated client values are bounded within the bit-width. The aggregate at
  the server is assumed to follow a subgaussian distribution. Note that the
  DDP algorithm is correct for any reasonable scaling factor, thus even if the
  subgaussian assumption does not hold (e.g. in the case of distributed Skellam
  which has sub-exponential tails), this function still provides a useful
  heuristic. See Section 4.2 and 4.4 of https://arxiv.org/pdf/2102.06387.pdf
  for more details.

  Specifically, the implementation is solving for gamma using the following
  expression:

    2^b = 2k * sqrt(rho / dim * (cn)^2 + (gamma^2 / 4 + sigma^2) * n) / gamma.

  Args:
    local_stddev: The local noise standard deviation.
    l2_clip: The initial L2 clip norm. See the __init__ docstring.
    bits: The bit-width. See the __init__ docstring.
    num_clients: The expected number of clients. See the __init__ docstring.
    dim: The dimension of the client vector that includes any necessary padding.
    k_stddevs: The number of standard deviations of the noisy and quantized
      aggregate values to bound within the bit-width.
    rho: (Optional) The subgaussian flatness parameter of the random orthogonal
      transform as part of the DDP procedure. See Section 4.2 of the above paper
      for more details.

  Returns:
    The selected scaling factor in tf.float64.
  """
  bits = tf.cast(bits, tf.float64)
  c = tf.cast(l2_clip, tf.float64)
  dim = tf.cast(dim, tf.float64)
  k_stddevs = tf.cast(k_stddevs, tf.float64)
  n = tf.cast(num_clients, tf.float64)
  sigma = tf.cast(local_stddev, tf.float64)

  numer = tf.sqrt(2.0**(2.0 * bits) - n * k_stddevs**2)
  denom = 2.0 * k_stddevs * tf.sqrt(rho / dim * c**2 * n**2 + n * sigma**2)
  scale_factor = numer / denom
  return scale_factor


def _check_scalar(value, label):
  is_bool = isinstance(value, bool)
  is_py_scalar = isinstance(value, (int, float))
  is_np_scalar = np.isscalar(value)
  if is_bool or not (is_py_scalar or is_np_scalar):
    raise TypeError(f'{label} must be a scalar. Got {repr(value)}.')


def _check_bool(value, label):
  if not isinstance(value, bool):
    raise TypeError(f'{label} must be a bool. Found {repr(value)}.')


def _check_positive(value, label):
  _check_scalar(value, label)
  if value <= 0:
    raise ValueError(f'{label} must be positive. Found {repr(value)}.')
