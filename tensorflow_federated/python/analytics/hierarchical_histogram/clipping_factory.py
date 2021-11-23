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
"""The private clipping factory for hierarchical histogram computation."""
from typing import Optional

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.templates import aggregation_process

# Supported clip mechanisms.
CLIP_MECHANISMS = [
    'sub-sampling',
    'distinct',
]


class HistogramClippingSumFactory(factory.UnweightedAggregationFactory):
  """An `UnweightedAggregationFactory` for clipping client-side histogram.

  Supports two types of clipping on local histograms to bound client-side
  contribution. Only to be used in hierarchical histogram computation.
  (1) Sub-sampling: Takes in a histogram represented by an integer tensor,
  uniformly samples up to `max_records_per_user` records without replacement
  from it, and returns the histogram on the sub-sampled records.

  For example:
      Input  = [1, 2, 3, 5], max_records_per_user = 5;
      Example Outputs = [1, 2, 1, 1], [0, 0, 0, 5], [0, 2, 3, 0]

  (2) Distinct Sub-sampling: Takes in a histogram represented by an integer
  tensor, uniquifies it, uniformly samples up to `max_records_per_user`
  records without replacement from it, and returns the histogram on the
  sub-sampled records.

  For example:
      Input  = [0, 2, 3, 5], max_records_per_user = 2;
      Example Outputs = [0, 1, 1, 0], [0, 0, 1, 1], [0, 1, 0, 1]
  """

  def __init__(
      self,
      clip_mechanism: str = 'sub-sampling',
      max_records_per_user: int = 10,
      inner_agg_factory: Optional[factory.UnweightedAggregationFactory] = None,
      cast_to_float: bool = False):
    """Initializes a `HistogramClippingSumFactory` instance.

    Args:
      clip_mechanism: A `str` representing the clipping mechanism. Currently
        supported mechanisms are
      - 'sub-sampling': (Default) Uniformly sample up to `max_records_per_user`
        records without replacement from the client dataset.
      - 'distinct': Uniquify client dataset and uniformly sample up to
        `max_records_per_user` records without replacement from it.
      max_records_per_user: An `int` representing the maximum of records each
        user can include in their local histogram. Defaults to 10.
      inner_agg_factory: (Optional) An `UnweightedAggregationFactory` specifying
        the value aggregation to be wrapped by `HistogramClippingSumFactory`.
        Defaults to `tff.aggregators.SumFactory`.
      cast_to_float: A boolean specifying the data type of the clipped
        histogram. If set to `False` (default), tensor with the same integer
        dtype will be passed to `inner_agg_factory`. If set to `True`, the
        clipped histogram will be casted to `tf.float32` before being passed to
        `inner_agg_factory`.

    Raises:
      TypeError: If arguments have the wrong type(s).
      ValueError: If arguments have invalid value(s).
    """
    _check_membership(clip_mechanism, CLIP_MECHANISMS, 'clip_mechanism')
    _check_positive(max_records_per_user, 'max_records_per_user')

    self._clip_mechanism = clip_mechanism
    self._max_records_per_user = max_records_per_user
    if inner_agg_factory is None:
      self._inner_agg_factory = sum_factory.SumFactory()
    else:
      self._inner_agg_factory = inner_agg_factory
    self._cast_to_float = cast_to_float

  def create(self, value_type):
    _check_is_tensor_type(value_type, 'value_type')
    _check_is_integer_struct(value_type, 'value_type')

    if self._clip_mechanism == 'sub-sampling':
      clip_fn = _sub_sample_clip
    elif self._clip_mechanism == 'distinct':
      clip_fn = _distinct_clip

    inner_value_type = value_type
    if self._cast_to_float:
      inner_value_type = computation_types.to_type(
          (tf.float32, value_type.shape))
    inner_agg_process = self._inner_agg_factory.create(inner_value_type)

    init_fn = inner_agg_process.initialize

    tff_clip_fn = computations.tf_computation(clip_fn)
    tff_cast_fn = computations.tf_computation(
        lambda x: tf.cast(x, inner_value_type.dtype))

    @computations.federated_computation(init_fn.type_signature.result,
                                        computation_types.at_clients(value_type)
                                       )
    def next_fn(state, value):
      # Clip values before aggregation.
      clipped_value = intrinsics.federated_map(
          tff_clip_fn, (value,
                        intrinsics.federated_value(self._max_records_per_user,
                                                   placements.CLIENTS)))
      clipped_value = intrinsics.federated_map(tff_cast_fn, clipped_value)

      return inner_agg_process.next(state, clipped_value)

    return aggregation_process.AggregationProcess(init_fn, next_fn)


def _sub_sample_clip(histogram, sample_num):
  """Clips `histogram` by sub-sampling.

  Uniformly samples up to `max_records_per_user` records without replacement
  from the client histogram.

  Args:
    histogram: A `tf.Tensor` with `dtype=tf.int32` representing a histogram.
    sample_num: The number of samples to draw from the histogram. If histogram
      contains fewer elements than `sample_num`, returns the original histogram.

  Returns:
    A `tf.Tensor` with `dtype=tf.int32` representing the sub-sampled histogram.
  """

  def sub_sample():
    indices = tf.repeat(tf.range(tf.shape(histogram)[0]), histogram)
    seed = tf.cast(
        tf.stack([tf.timestamp() * 1e6,
                  tf.timestamp() * 1e6]), dtype=tf.int64)
    samples = tf.random.stateless_uniform(tf.shape(indices), seed)
    _, sampled_idx = tf.math.top_k(samples, k=sample_num, sorted=False)
    ind = tf.expand_dims(tf.gather(indices, sampled_idx), axis=1)
    upd = tf.ones(tf.shape(sampled_idx), dtype=tf.int32)
    return tf.scatter_nd(indices=ind, updates=upd, shape=tf.shape(histogram))

  l1_norm = tf.norm(histogram, ord=1)
  result = tf.cond(
      tf.greater(l1_norm, sample_num), sub_sample, lambda: histogram)
  result.set_shape(histogram.shape.as_list())
  # Ensure shape as TF shape inference may fail due to custom sampling.
  return result


def _distinct_clip(histogram, sample_num):
  """Clips `histogram` by distinct sub-sampling.

  Uniquifies client dataset and uniformly samples up to `max_records_per_user`
  records without replacement from it.

  Args:
    histogram: A `tf.Tensor` with `dtype=tf.int32` representing a histogram.
    sample_num: The number of samples to draw from the histogram. If histogram
      contains fewer elements than `sample_num`, returns the uniquified
      histogram.

  Returns:
    A `tf.Tensor` with `dtype=tf.int32` representing the uniquified and
    sub-sampled histogram.
  """

  def distinct():
    indices = tf.squeeze(
        tf.cast(tf.where(tf.not_equal(histogram, 0)), tf.int32))
    seed = tf.cast(
        tf.stack([tf.timestamp() * 1e6,
                  tf.timestamp() * 1e6]), dtype=tf.int64)
    samples = tf.random.stateless_uniform(tf.shape(indices), seed)
    _, sampled_idx = tf.math.top_k(samples, k=sample_num, sorted=False)
    ind = tf.expand_dims(tf.gather(indices, sampled_idx), axis=1)
    upd = tf.ones(tf.shape(sampled_idx), dtype=tf.int32)
    return tf.scatter_nd(indices=ind, updates=upd, shape=tf.shape(histogram))

  l0_norm = tf.math.count_nonzero(histogram, dtype=tf.int32)
  result = tf.cond(
      tf.greater(l0_norm, sample_num), distinct,
      lambda: tf.minimum(histogram, 1))
  # Ensure shape as TF shape inference may fail due to custom sampling.
  return tf.reshape(result, histogram.shape)


def _check_is_integer_struct(value_type, label):
  if not type_analysis.is_structure_of_integers(value_type):
    raise TypeError(f'Component dtypes of `{label}` must all be integers. '
                    f'Found {repr(value_type)}.')


def _check_is_tensor_type(value, label):
  if not value.is_tensor():
    raise TypeError(f'Expected `{label}` to be `TensorType`. '
                    f'Found type: {repr(value)}')


def _check_positive(value, label):
  if value <= 0:
    raise ValueError(f'{label} must be positive. Found {value}.')


def _check_membership(value, valid_set, label):
  if value not in valid_set:
    raise ValueError(f'`{label}` must be one of {valid_set}. '
                     f'Found {value}.')
