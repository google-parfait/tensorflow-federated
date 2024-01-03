# Copyright 2023, The TensorFlow Federated Authors.
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

import collections
from typing import Any

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.learning.metrics import sampling_aggregation_factory
from tensorflow_federated.python.learning.metrics import types


def _create_metric_finalizers() -> types.MetricFinalizersType:
  return collections.OrderedDict(
      accuracy=tf.function(func=lambda x: x),
      loss=tf.function(func=lambda x: tf.math.divide_no_nan(x[0], x[1])),
      custom_sum=tf.function(
          func=lambda x: tf.add_n(map(tf.math.reduce_sum, x))
      ),
  )


def _create_functional_metric_finalizers() -> (
    types.FunctionalMetricFinalizersType
):
  def functional_metric_finalizers(unfinalized_metrics):
    return collections.OrderedDict(
        accuracy=unfinalized_metrics['accuracy'],
        loss=tf.math.divide_no_nan(
            unfinalized_metrics['loss'][0], unfinalized_metrics['loss'][1]
        ),
        custom_sum=tf.add_n(
            map(tf.math.reduce_sum, unfinalized_metrics['custom_sum'])
        ),
    )

  return functional_metric_finalizers


def _create_random_unfinalized_metrics(seed: int):
  metrics = collections.OrderedDict(
      accuracy=tf.random.stateless_uniform(shape=(), seed=[seed, seed]),
      loss=tf.random.stateless_uniform(shape=(2,), seed=[seed, seed]),
      custom_sum=[
          tf.random.stateless_uniform(shape=(1,), seed=[seed, seed]),
          tf.random.stateless_uniform(shape=(2,), seed=[seed, seed]),
      ],
  )
  metrics_type = computation_types.StructWithPythonType(
      [
          ('accuracy', computation_types.TensorType(np.float32)),
          ('loss', computation_types.TensorType(np.float32, (2,))),
          (
              'custom_sum',
              [
                  computation_types.TensorType(np.float32, shape=(1,)),
                  computation_types.TensorType(np.float32, shape=(2,)),
              ],
          ),
      ],
      collections.OrderedDict,
  )
  return metrics, metrics_type


def _create_finalized_clients_metrics(
    metric_finalizers: types.MetricFinalizersType,
    list_clients_unfinalized_metrics: list[dict[str, Any]],
) -> dict[str, list[float]]:
  # The returned dictionary maps metric names to lists of finalized metric
  # values, where each metric value is from a client.
  total_clients_finalized_metrics = collections.OrderedDict(
      {metric_name: [] for metric_name in metric_finalizers}
  )
  for unfinalized_metrics in list_clients_unfinalized_metrics:
    for metric_name in metric_finalizers:
      current_client_finalized_metric = metric_finalizers[metric_name](
          unfinalized_metrics[metric_name]
      )
      total_clients_finalized_metrics[metric_name].append(
          current_client_finalized_metric.numpy()
      )
  return total_clients_finalized_metrics


def _create_functional_finalized_clients_metrics(
    metric_finalizers: types.FunctionalMetricFinalizersType,
    list_clients_unfinalized_metrics: list[dict[str, Any]],
) -> dict[str, list[float]]:
  # The returned dictionary maps metric names to lists of finalized metric
  # values, where each metric value is from a client.
  total_clients_finalized_metrics = collections.OrderedDict(
      {metric_name: [] for metric_name in list_clients_unfinalized_metrics[0]}
  )
  for unfinalized_metrics in list_clients_unfinalized_metrics:
    finalized_metrics = metric_finalizers(unfinalized_metrics)
    for metric_name in finalized_metrics:
      total_clients_finalized_metrics[metric_name].append(
          finalized_metrics[metric_name].numpy()
      )
  return total_clients_finalized_metrics


class SamplingAggregationFactoryTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('zero', 0, ValueError, 'must be positive'),
      ('negative', -1, ValueError, 'must be positive'),
      ('none', None, TypeError, 'sample_size'),
      ('string', '5', TypeError, 'sample_size'),
  )
  def test_build_factory_fails_with_invalid_sample_size(
      self, bad_sample_size, expected_error_type, expected_error_message
  ):
    with self.assertRaisesRegex(expected_error_type, expected_error_message):
      sampling_aggregation_factory.FinalizeThenSampleFactory(
          sample_size=bad_sample_size
      )

  @parameterized.named_parameters(
      ('float', 1.0),
      (
          'list',
          [tf.function(func=lambda x: x), tf.function(func=lambda x: x + 1)],
      ),
  )
  def test_create_process_fails_with_invalid_metric_finalizers(
      self, bad_metric_finalizers
  ):
    _, unused_metrics_type = _create_random_unfinalized_metrics(seed=0)
    with self.assertRaisesRegex(TypeError, 'metric_finalizers'):
      sampling_aggregation_factory.FinalizeThenSampleFactory(
          sample_size=10
      ).create(bad_metric_finalizers, unused_metrics_type)

  @parameterized.named_parameters(
      (
          'federated_type',
          computation_types.FederatedType(np.float32, placements.SERVER),
      ),
      ('function_type', computation_types.FunctionType(None, ())),
      ('sequence_type', computation_types.SequenceType(np.float32)),
  )
  def test_create_process_fails_with_invalid_unfinalized_metrics_type(
      self, bad_unfinalized_metrics_type
  ):
    unused_metric_finalizers = _create_metric_finalizers()
    with self.assertRaisesRegex(
        TypeError, 'Expected .*`tff.types.StructWithPythonType`'
    ):
      sampling_aggregation_factory.FinalizeThenSampleFactory(
          sample_size=10
      ).create(unused_metric_finalizers, bad_unfinalized_metrics_type)

  def test_create_process_fails_with_mismatch_finalizers_and_metrics_type(self):
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x)
    )
    _, unfinalized_metrics_type = _create_random_unfinalized_metrics(seed=0)
    with self.assertRaisesRegex(
        ValueError, 'The metric names in `metric_finalizers` do not match those'
    ):
      sampling_aggregation_factory.FinalizeThenSampleFactory(
          sample_size=10
      ).create(metric_finalizers, unfinalized_metrics_type)

  @parameterized.named_parameters(
      ('sample_size_larger_than_per_round_clients', 5, 2, False),
      ('sample_size_equal_per_round_clients', 3, 3, False),
      ('sample_size_smaller_than_per_round_clients', 2, 3, False),
      ('sample_size_larger_than_per_round_clients_functional', 5, 2, True),
      ('sample_size_equal_per_round_clients_functional', 3, 3, True),
      ('sample_size_smaller_than_per_round_clients_functional', 2, 3, True),
  )
  def test_finalize_then_sample_returns_expected_samples(
      self,
      sample_size: int,
      num_clients_per_round: int,
      functional_metric_finalizers: bool,
  ):
    if functional_metric_finalizers:
      metric_finalizers = _create_functional_metric_finalizers()
      create_finalized_clients_metrics_fn = (
          _create_functional_finalized_clients_metrics
      )
    else:
      metric_finalizers = _create_metric_finalizers()
      create_finalized_clients_metrics_fn = _create_finalized_clients_metrics
    total_rounds = 5
    _, local_unfinalized_metrics_type = _create_random_unfinalized_metrics(
        seed=0
    )
    sampling_process = sampling_aggregation_factory.FinalizeThenSampleFactory(
        sample_size
    ).create(metric_finalizers, local_unfinalized_metrics_type)
    state = sampling_process.initialize()
    total_rounds_client_values = []
    for round_i in range(total_rounds):
      current_round_client_values = [
          _create_random_unfinalized_metrics(seed=round_i + client_i)[0]
          for client_i in range(num_clients_per_round)
      ]
      total_rounds_client_values.extend(current_round_client_values)
      output = sampling_process.next(state, current_round_client_values)
      state = output.state
      current_round_samples, total_rounds_samples = output.result
      expected_num_current_round_samples = min(
          sample_size, num_clients_per_round
      )
      expected_num_total_rounds_samples = min(
          sample_size, num_clients_per_round * (round_i + 1)
      )
      current_round_finalized_metrics = create_finalized_clients_metrics_fn(
          metric_finalizers, current_round_client_values
      )
      total_rounds_finalized_metrics = create_finalized_clients_metrics_fn(
          metric_finalizers, total_rounds_client_values
      )
      for metric_name in current_round_finalized_metrics:
        self.assertLen(
            current_round_samples[metric_name],
            expected_num_current_round_samples,
        )
        self.assertContainsSubset(
            current_round_samples[metric_name],
            current_round_finalized_metrics[metric_name],
        )
        self.assertLen(
            total_rounds_samples[metric_name], expected_num_total_rounds_samples
        )
        self.assertContainsSubset(
            total_rounds_samples[metric_name],
            total_rounds_finalized_metrics[metric_name],
        )

  @parameterized.named_parameters(
      # In this test, we set the number of per-round clients = 5. Large/small
      # sample size is relative to this number. The `sample_size` should not
      # affect the measurements because they are computed *before* sampling.
      ('sample_size_larger_than_per_round_clients', 6),
      ('sample_size_equal_per_round_clients', 5),
      ('sample_size_smaller_than_per_round_clients', 1),
  )
  def test_finalize_then_sample_returns_correct_measurements(self, sample_size):
    metric_finalizers = lambda x: x
    unfinalized_metrics_type = computation_types.StructWithPythonType(
        [('loss', np.float32)], collections.OrderedDict
    )
    sampling_process = sampling_aggregation_factory.FinalizeThenSampleFactory(
        sample_size
    ).create(metric_finalizers, unfinalized_metrics_type)
    state = sampling_process.initialize()
    output = sampling_process.next(
        state,
        [
            collections.OrderedDict(loss=v)
            for v in [1.0, np.nan, np.inf, 2.0, 3.0]
        ],
    )
    # Two clients' values are non-finite. Note that `sample_size` does not
    # matter because we count per-round non-finite values *before* sampling.
    self.assertEqual(
        output.measurements,
        collections.OrderedDict(loss=np.array(2, dtype=np.int64)),
    )
    state = output.state
    output = sampling_process.next(
        state,
        [collections.OrderedDict(loss=v) for v in [1.0, 4.0, np.inf, 2.0, 3.0]],
    )
    # One client's values are non-finite. Note that `sample_size` does not
    # matter because we count per-round non-finite values *before* sampling.
    self.assertEqual(
        output.measurements,
        collections.OrderedDict(loss=np.array(1, dtype=np.int64)),
    )


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
