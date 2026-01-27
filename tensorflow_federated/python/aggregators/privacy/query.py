# Copyright 2020, The TensorFlow Authors.
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
"""An interface for differentially private query mechanisms."""

import abc
import collections

import dp_accounting
import tensorflow as tf


class DPQuery(metaclass=abc.ABCMeta):
  """Interface for differentially private query mechanisms.

  Differential privacy is achieved by processing records to bound sensitivity,
  accumulating the processed records (usually by summing them) and then
  adding noise to the aggregated result. The process can be repeated to compose
  applications of the same mechanism, possibly with different parameters.

  The DPQuery interface specifies a functional approach to this process. A
  global state maintains state that persists across applications of the
  mechanism. For each application, the following steps are performed:

  1. Use the global state to derive parameters to use for the next sample of
     records.
  2. Initialize a sample state that will accumulate processed records.
  3. For each record:
     a. Process the record.
     b. Accumulate the record into the sample state.
  4. Get the result of the mechanism, possibly updating the global state to use
     in the next application.
  5. Derive metrics from the global state.

  Here is an example using the GaussianSumQuery. Assume there is some function
  records_for_round(round) that returns an iterable of records to use on some
  round.

  ```
  dp_query = tensorflow_privacy.GaussianSumQuery(
      l2_norm_clip=1.0, stddev=1.0)
  global_state = dp_query.initial_global_state()

  for round in range(num_rounds):
    sample_params = dp_query.derive_sample_params(global_state)
    sample_state = dp_query.initial_sample_state()
    for record in records_for_round(round):
      sample_state = dp_query.accumulate_record(
          sample_params, sample_state, record)

    result, global_state = dp_query.get_noised_result(
        sample_state, global_state)
    metrics = dp_query.derive_metrics(global_state)

    # Do something with result and metrics...
  ```
  """

  def initial_global_state(self):
    """Returns the initial global state for the DPQuery.

    The global state contains any state information that changes across
    repeated applications of the mechanism. The default implementation returns
    just an empty tuple for implementing classes that do not have any persistent
    state.

    This object must be processable via tf.nest.map_structure.

    Returns:
      The global state.
    """
    return ()

  def derive_sample_params(self, global_state):
    """Given the global state, derives parameters to use for the next sample.

    For example, if the mechanism needs to clip records to bound the norm,
    the clipping norm should be part of the sample params. In a distributed
    context, this is the part of the state that would be sent to the workers
    so they can process records.

    Args:
      global_state: The current global state.

    Returns:
      Parameters to use to process records in the next sample.
    """
    del global_state  # unused.
    return ()

  @abc.abstractmethod
  def initial_sample_state(self, template=None):
    """Returns an initial state to use for the next sample.

    For typical `DPQuery` classes that are aggregated by summation, this should
    return a nested structure of zero tensors of the appropriate shapes, to
    which processed records will be aggregated.

    Args:
      template: A nested structure of tensors, TensorSpecs, or numpy arrays used
        as a template to create the initial sample state. It is assumed that the
        leaves of the structure are python scalars or some type that has
        properties `shape` and `dtype`.

    Returns: An initial sample state.
    """

  def preprocess_record(self, params, record):
    """Preprocesses a single record.

    This preprocessing is applied to one client's record, e.g. selecting vectors
    and clipping them to a fixed L2 norm. This method can be executed in a
    separate TF session, or even on a different machine, so it should not depend
    on any TF inputs other than those provided as input arguments. In
    particular, implementations should avoid accessing any TF tensors or
    variables that are stored in self.

    Args:
      params: The parameters for the sample. In standard DP-SGD training, the
        clipping norm for the sample's microbatch gradients (i.e., a maximum
        norm magnitude to which each gradient is clipped)
      record: The record to be processed. In standard DP-SGD training, the
        gradient computed for the examples in one microbatch, which may be the
        gradient for just one example (for size 1 microbatches).

    Returns:
      A structure of tensors to be aggregated.
    """
    del params  # unused.
    return record

  @abc.abstractmethod
  def accumulate_preprocessed_record(self, sample_state, preprocessed_record):
    """Accumulates a single preprocessed record into the sample state.

    This method is intended to only do simple aggregation, typically just a sum.
    In the future, we might remove this method and replace it with a way to
    declaratively specify the type of aggregation required.

    Args:
      sample_state: The current sample state. In standard DP-SGD training, the
        accumulated sum of previous clipped microbatch gradients.
      preprocessed_record: The preprocessed record to accumulate.

    Returns:
      The updated sample state.
    """

  def accumulate_record(self, params, sample_state, record):
    """Accumulates a single record into the sample state.

    This is a helper method that simply delegates to `preprocess_record` and
    `accumulate_preprocessed_record` for the common case when both of those
    functions run on a single device. Typically this will be a simple sum.

    Args:
      params: The parameters for the sample. In standard DP-SGD training, the
        clipping norm for the sample's microbatch gradients (i.e., a maximum
        norm magnitude to which each gradient is clipped)
      sample_state: The current sample state. In standard DP-SGD training, the
        accumulated sum of previous clipped microbatch gradients.
      record: The record to accumulate. In standard DP-SGD training, the
        gradient computed for the examples in one microbatch, which may be the
        gradient for just one example (for size 1 microbatches).

    Returns:
      The updated sample state. In standard DP-SGD training, the set of
      previous microbatch gradients with the addition of the record argument.
    """
    preprocessed_record = self.preprocess_record(params, record)
    return self.accumulate_preprocessed_record(
        sample_state, preprocessed_record
    )

  @abc.abstractmethod
  def merge_sample_states(self, sample_state_1, sample_state_2):
    """Merges two sample states into a single state.

    This can be useful if aggregation is performed hierarchically, where
    multiple sample states are used to accumulate records and then
    hierarchically merged into the final accumulated state. Typically this will
    be a simple sum.

    Args:
      sample_state_1: The first sample state to merge.
      sample_state_2: The second sample state to merge.

    Returns:
      The merged sample state.
    """

  @abc.abstractmethod
  def get_noised_result(self, sample_state, global_state):
    """Gets the query result after all records of sample have been accumulated.

    The global state can also be updated for use in the next application of the
    DP mechanism.

    Args:
      sample_state: The sample state after all records have been accumulated. In
        standard DP-SGD training, the accumulated sum of clipped microbatch
        gradients (in the special case of microbatches of size 1, the clipped
        per-example gradients).
      global_state: The global state, storing long-term privacy bookkeeping.

    Returns:
      A tuple `(result, new_global_state, event)` where:
        * `result` is the result of the query,
        * `new_global_state` is the updated global state, and
        * `event` is the `DpEvent` that occurred.
      In standard DP-SGD training, the result is a gradient update comprising a
      noised average of the clipped gradients in the sample state---with the
      noise and averaging performed in a manner that guarantees differential
      privacy.
    """

  def derive_metrics(self, global_state):
    """Derives metric information from the current global state.

    Any metrics returned should be derived only from privatized quantities.

    Args:
      global_state: The global state from which to derive metrics.

    Returns:
      A `collections.OrderedDict` mapping string metric names to tensor values.
    """
    del global_state
    return collections.OrderedDict()


def _zeros_like(arg):
  """A `zeros_like` function that also works for `tf.TensorSpec`s."""
  try:
    arg = tf.convert_to_tensor(value=arg)
  except (TypeError, ValueError):
    pass
  return tf.zeros(arg.shape, arg.dtype)


def _safe_add(x, y):
  """Adds x and y but if y is None, simply returns x."""
  return x if y is None else tf.add(x, y)


class SumAggregationDPQuery(DPQuery):
  """Base class for DPQueries that aggregate via sum."""

  def initial_sample_state(self, template=None):
    """Implements `tensorflow_privacy.DPQuery.initial_sample_state`."""
    return tf.nest.map_structure(_zeros_like, template)

  def accumulate_preprocessed_record(self, sample_state, preprocessed_record):
    """Implements `tensorflow_privacy.DPQuery.accumulate_preprocessed_record`."""
    return tf.nest.map_structure(_safe_add, sample_state, preprocessed_record)

  def merge_sample_states(self, sample_state_1, sample_state_2):
    """Implements `tensorflow_privacy.DPQuery.merge_sample_states`."""
    return tf.nest.map_structure(tf.add, sample_state_1, sample_state_2)


class GaussianSumQuery(SumAggregationDPQuery):
  """Implements DPQuery interface for Gaussian sum queries.

  Clips records to bound the L2 norm, then adds Gaussian noise to the sum.
  """

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple(
      '_GlobalState', ['l2_norm_clip', 'stddev']
  )

  def __init__(self, l2_norm_clip, stddev):
    """Initializes the GaussianSumQuery.

    Args:
      l2_norm_clip: The clipping norm to apply to the global norm of each
        record.
      stddev: The stddev of the noise added to the sum.
    """
    self._l2_norm_clip = l2_norm_clip
    self._stddev = stddev

  def make_global_state(self, l2_norm_clip, stddev):
    """Creates a global state from the given parameters."""
    return self._GlobalState(
        tf.cast(l2_norm_clip, tf.float32), tf.cast(stddev, tf.float32)
    )

  def initial_global_state(self):
    """Implements `tensorflow_privacy.DPQuery.initial_global_state`."""
    return self.make_global_state(self._l2_norm_clip, self._stddev)

  def derive_sample_params(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_sample_params`."""
    return global_state.l2_norm_clip

  def preprocess_record_impl(self, params, record):
    """Clips the l2 norm, returning the clipped record and the l2 norm.

    Args:
      params: The parameters for the sample.
      record: The record to be processed.

    Returns:
      A tuple (preprocessed_records, l2_norm) where `preprocessed_records` is
        the structure of preprocessed tensors, and l2_norm is the total l2 norm
        before clipping.
    """
    l2_norm_clip = params
    record_as_list = tf.nest.flatten(record)
    clipped_as_list, norm = tf.clip_by_global_norm(record_as_list, l2_norm_clip)
    return tf.nest.pack_sequence_as(record, clipped_as_list), norm

  def preprocess_record(self, params, record):
    """Implements `tensorflow_privacy.DPQuery.preprocess_record`."""
    preprocessed_record, _ = self.preprocess_record_impl(params, record)
    return preprocessed_record

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`."""
    random_normal = tf.random_normal_initializer(stddev=global_state.stddev)

    def add_noise(v):
      return v + tf.cast(random_normal(tf.shape(input=v)), dtype=v.dtype)

    result = tf.nest.map_structure(add_noise, sample_state)
    noise_multiplier = global_state.stddev / global_state.l2_norm_clip
    event = dp_accounting.GaussianDpEvent(noise_multiplier)

    return result, global_state, event


class NoPrivacySumQuery(SumAggregationDPQuery):
  """Implements DPQuery interface for a sum query with no privacy.

  Accumulates vectors without clipping or adding noise.
  """

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`."""
    return sample_state, global_state, dp_accounting.NonPrivateDpEvent()


class NoPrivacyAverageQuery(SumAggregationDPQuery):
  """Implements DPQuery interface for an average query with no privacy.

  Accumulates vectors and normalizes by the total number of accumulated vectors.
  Under some sampling schemes, such as Poisson subsampling, the number of
  records in a sample is a private quantity, so we lose all privacy guarantees
  by using the number of records directly to normalize.

  Also allows weighted accumulation, unlike the base class DPQuery. In a private
  implementation of weighted average, the weight would have to be itself
  privatized.
  """

  def initial_sample_state(self, template=None):
    """Implements `tensorflow_privacy.DPQuery.initial_sample_state`."""
    return super().initial_sample_state(template), tf.constant(0.0)

  def preprocess_record(self, params, record, weight=1):
    """Implements `tensorflow_privacy.DPQuery.preprocess_record`.

    Optional `weight` argument allows weighted accumulation.

    Args:
      params: The parameters for the sample.
      record: The record to accumulate.
      weight: Optional weight for the record.

    Returns:
      The preprocessed record.
    """
    weighted_record = tf.nest.map_structure(lambda t: weight * t, record)
    return (weighted_record, tf.cast(weight, tf.float32))

  def accumulate_record(self, params, sample_state, record, weight=1):
    """Implements `tensorflow_privacy.DPQuery.accumulate_record`.

    Optional `weight` argument allows weighted accumulation.

    Args:
      params: The parameters for the sample.
      sample_state: The current sample state.
      record: The record to accumulate.
      weight: Optional weight for the record.

    Returns:
      The updated sample state.
    """
    weighted_record = tf.nest.map_structure(lambda t: weight * t, record)
    return self.accumulate_preprocessed_record(
        sample_state, (weighted_record, tf.cast(weight, tf.float32))
    )

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`."""
    sum_state, denominator = sample_state

    result = tf.nest.map_structure(lambda t: t / denominator, sum_state)
    return result, global_state, dp_accounting.NonPrivateDpEvent()


class NormalizedQuery(SumAggregationDPQuery):
  """`DPQuery` for queries with a `DPQuery` numerator and fixed denominator.

  If the number of records per round is a public constant R, `NormalizedQuery`
  could be used with a sum query as the numerator and R as the denominator to
  implement an average. Under some sampling schemes, such as Poisson
  subsampling, the actual number of records in a sample is a private quantity,
  so we cannot use it directly. Using this class with the expected number of
  records as the denominator gives an unbiased estimate of the average.
  """

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple(
      '_GlobalState', ['numerator_state', 'denominator']
  )

  def __init__(self, numerator_query, denominator):
    """Initializes the NormalizedQuery.

    Args:
      numerator_query: A SumAggregationDPQuery for the numerator.
      denominator: A value for the denominator. May be None if it will be
        supplied via the set_denominator function before get_noised_result is
        called.
    """
    self._numerator = numerator_query
    self._denominator = denominator

    assert isinstance(self._numerator, SumAggregationDPQuery)

  def initial_global_state(self):
    """Implements `tensorflow_privacy.DPQuery.initial_global_state`."""
    denominator = tf.cast(self._denominator, tf.float32)
    return self._GlobalState(
        self._numerator.initial_global_state(), denominator
    )

  def derive_sample_params(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_sample_params`."""
    return self._numerator.derive_sample_params(global_state.numerator_state)

  def initial_sample_state(self, template=None):
    """Implements `tensorflow_privacy.DPQuery.initial_sample_state`."""
    # NormalizedQuery has no sample state beyond the numerator state.
    return self._numerator.initial_sample_state(template)

  def preprocess_record(self, params, record):
    """Implements `tensorflow_privacy.DPQuery.preprocess_record`."""
    return self._numerator.preprocess_record(params, record)

  def get_noised_result(self, sample_state, global_state):
    """Implements `tensorflow_privacy.DPQuery.get_noised_result`."""
    noised_sum, new_sum_global_state, event = self._numerator.get_noised_result(
        sample_state, global_state.numerator_state
    )

    def normalize(v):
      return tf.truediv(v, tf.cast(global_state.denominator, v.dtype))

    # The denominator is constant so the privacy cost comes from the numerator.
    return (
        tf.nest.map_structure(normalize, noised_sum),
        self._GlobalState(new_sum_global_state, global_state.denominator),
        event,
    )

  def derive_metrics(self, global_state):
    """Implements `tensorflow_privacy.DPQuery.derive_metrics`."""
    return self._numerator.derive_metrics(global_state.numerator_state)
