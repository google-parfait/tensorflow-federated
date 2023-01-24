# Copyright 2021, The TensorFlow Federated Authors.
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
"""Factory for clipping client strings before aggregation via IBLT."""

import collections
from typing import Optional

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.analytics import data_processing
from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_factory
from tensorflow_federated.python.core.impl.federated_context import federated_computation
from tensorflow_federated.python.core.impl.federated_context import intrinsics
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.templates import aggregation_process


@tf.function
def get_clipped_elements_with_counts(
    dataset: tf.data.Dataset,
    max_words_per_user: Optional[int] = None,
    multi_contribution: bool = True,
    batch_size: int = 1,
    string_max_bytes: int = 10,
    unique_counts: bool = False,
) -> tf.data.Dataset:
  """Gets elements and corresponding clipped counts from the input `dataset`.

  Returns a dataset that yields `OrderedDict`s with two keys: `key` with
  `tf.string` scalar value, and `value' with list of tf.int64 scalar values.
  The list is of length one or two, with each entry representing the (clipped)
  count for a given word and (if unique_counts=True) the constant 1.  The
  primary intended use case for this function is to preprocess client-data
  before sending it through `tff.analytics.IbltFactory` for heavy hitter
  calculations.

  Args:
    dataset: The input `tf.data.Dataset` whose elements are to be counted.
    max_words_per_user: The maximum total count each client is allowed to
      contribute across all words. If not `None`, must be a positive integer.
      Defaults to `None`, which means all the clients contribute all their
      words. Note that this does not cap the count of each individual word each
      client can contribute. Set `multi_contirbution=False` to restrict the
      per-client count for each word.
    multi_contribution: Whether each client is allowed to contribute multiple
      instances of each string, or only a count of one for each unique word.
      Defaults to `True` meaning clients contribute the full count for each
      contributed string. Note that this doesn't limit the total number of
      strings each client can contribute. Set `max_words_per_user` to limit the
      total number of strings per client.
    batch_size: The number of elements in each batch of the dataset. Batching is
      an optimization for pulling multiple inputs at a time from the input
      `tf.data.Dataset`, amortizing the overhead cost of each read to the
      `batch_size`. Consider batching if you observe poor client execution
      performance or reading inputs is particularly expsensive. Defaults to `1`,
      means the input dataset is processed by `tf.data.Dataset.batch(1)`. Must
      be positive.
    string_max_bytes: The maximum length in bytes of a string in the IBLT.
      Strings longer than `string_max_bytes` will be truncated. Defaults to
      `10`. Must be positive.
    unique_counts: If True, the value for every element is the array [count, 1].

  Returns:
    A dataset containing an OrderedDict of elements and corresponding counts.
  """
  if max_words_per_user is not None:
    if multi_contribution:
      k_words, counts = data_processing.get_capped_elements_with_counts(
          dataset,
          max_words_per_user,
          batch_size=batch_size,
          string_max_bytes=string_max_bytes,
      )
    else:
      # `tff.analytics.data_processing.get_top_elements_with_counts` returns the
      # top `max_words_per_user` words in client's local histogram. Each element
      # appears at most once in the list.
      k_words, counts = data_processing.get_top_elements_with_counts(
          dataset, max_words_per_user, string_max_bytes=string_max_bytes
      )
      counts = tf.ones_like(counts)
  else:
    k_words, counts = data_processing.get_unique_elements_with_counts(
        dataset, string_max_bytes=string_max_bytes
    )
    if not multi_contribution:
      counts = tf.ones_like(counts)
  if unique_counts:
    values = tf.stack([counts, tf.ones_like(counts)], axis=1)
  else:
    values = counts
  client = collections.OrderedDict([('key', k_words), ('value', values)])
  return tf.data.Dataset.from_tensor_slices(client)


class ClippingIbltFactory(factory.UnweightedAggregationFactory):
  """Factory for clipping client data before aggregation."""

  def __init__(
      self,
      inner_iblt_agg: iblt_factory.IbltFactory,
      max_words_per_user: Optional[int] = None,
      multi_contribution: bool = True,
      batch_size: int = 1,
      string_max_bytes: int = 10,
      unique_counts: bool = False,
  ):
    """Initializes ClientPreprocessingAggregationFactory.

    Args:
      inner_iblt_agg: An instance of IbltFactory.
      max_words_per_user: The maximum total count each client is allowed to
        contribute across all words. If not `None`, must be a positive integer.
        Defaults to `None`, which means all the clients contribute all their
        words. Note that this does not cap the count of each individual word
        each client can contribute. Set `multi_contirbution=False` to restrict
        the per-client count for each word.
      multi_contribution: Whether each client is allowed to contribute multiple
        instances of each string, or only a count of one for each unique word.
        Defaults to `True` meaning clients contribute the full count for each
        contributed string. Note that this doesn't limit the total number of
        strings each client can contribute. Set `max_words_per_user` to limit
        the total number of strings per client.
      batch_size: The number of elements in each batch of the dataset. Batching
        is an optimization for pulling multiple inputs at a time from the input
        `tf.data.Dataset`, amortizing the overhead cost of each read to the
        `batch_size`. Consider batching if you observe poor client execution
        performance or reading inputs is particularly expsensive. Defaults to
        `1`, means the input dataset is processed by `tf.data.Dataset.batch(1)`.
        Must be positive.
      string_max_bytes: The maximum length in bytes of a string in the IBLT.
        Strings longer than `string_max_bytes` will be truncated. Defaults to
        `10`. Must be positive.
      unique_counts: If True, the value for every element is the array [count,
        1].
    """
    self.inner_iblt_agg = inner_iblt_agg
    self.max_words_per_user = max_words_per_user
    self.multi_contribution = multi_contribution
    self.batch_size = batch_size
    self.string_max_bytes = string_max_bytes
    self.unique_counts = unique_counts

  def create(
      self, value_type: factory.ValueType
  ) -> aggregation_process.AggregationProcess:
    expected_type = computation_types.SequenceType(
        computation_types.TensorType(shape=[None], dtype=tf.string)
    )

    if value_type != expected_type:
      raise ValueError(
          'Expected value_type to be %s, got %s' % (expected_type, value_type)
      )

    @tensorflow_computation.tf_computation(value_type)
    @tf.function
    def preprocess(client_data):
      return get_clipped_elements_with_counts(
          client_data,
          self.max_words_per_user,
          self.multi_contribution,
          self.batch_size,
          self.string_max_bytes,
          self.unique_counts,
      )

    inner_process = self.inner_iblt_agg.create(preprocess.type_signature.result)

    @federated_computation.federated_computation(
        inner_process.initialize.type_signature.result,
        computation_types.at_clients(value_type),
    )
    def next_fn(state, client_data):
      preprocessed = intrinsics.federated_map(preprocess, client_data)
      return inner_process.next(state, preprocessed)

    return aggregation_process.AggregationProcess(
        inner_process.initialize, next_fn
    )
