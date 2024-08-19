# Copyright 2022, The TensorFlow Federated Authors.
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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import estimation_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.core.test import static_assert
from tensorflow_federated.python.learning.metrics import sum_aggregation_factory

# Convenience aliases.
TensorType = computation_types.TensorType
MetricRange = sum_aggregation_factory._MetricRange


def _get_finalized_metrics_type(metric_finalizers, unfinalized_metrics):

  # TODO: b/319261270 - Avoid the need for inferring types here, if possible.
  def _tensor_type_from_tensor_like(x):
    x_as_tensor = tf.convert_to_tensor(x)
    return computation_types.tensorflow_to_type(
        (x_as_tensor.dtype, x_as_tensor.shape)
    )

  if callable(metric_finalizers):
    finalized_metrics = metric_finalizers(unfinalized_metrics)
  else:
    finalized_metrics = collections.OrderedDict(
        (metric, finalizer(unfinalized_metrics[metric]))
        for metric, finalizer in metric_finalizers.items()
    )
  finalizer_type = tf.nest.map_structure(
      _tensor_type_from_tensor_like, finalized_metrics
  )
  return computation_types.StructWithPythonType(
      finalizer_type, collections.OrderedDict
  )


@tf.function
def _tf_mean(x):
  return tf.math.divide_no_nan(x[0], x[1])


_DEFAULT_FLOAT_FACTORY_KEY = 'None/default_estimation_process/float32'
_DEFAULT_INT_FACTORY_KEY = '0/1048575/int32'

_DEFAULT_FIXED_FLOAT_RANGE = (
    float(sum_aggregation_factory.DEFAULT_FIXED_SECURE_LOWER_BOUND),
    float(sum_aggregation_factory.DEFAULT_FIXED_SECURE_UPPER_BOUND),
)
_DEFAULT_INT_RANGE = (
    int(sum_aggregation_factory.DEFAULT_FIXED_SECURE_LOWER_BOUND),
    int(sum_aggregation_factory.DEFAULT_FIXED_SECURE_UPPER_BOUND),
)


class SumThenFinalizeFactoryComputationTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      (
          'scalar_metric',
          collections.OrderedDict(num_examples=tf.function(func=lambda x: x)),
          collections.OrderedDict(num_examples=1.0),
          computation_types.StructWithPythonType(
              [('num_examples', np.float32)], collections.OrderedDict
          ),
      ),
      (
          'non_scalar_metric',
          collections.OrderedDict(loss=_tf_mean),
          collections.OrderedDict(loss=[2.0, 1.0]),
          computation_types.StructWithPythonType(
              [('loss', [np.float32, np.float32])], collections.OrderedDict
          ),
      ),
      (
          'callable',
          tf.function(
              lambda x: collections.OrderedDict(mean_loss=_tf_mean(x['loss']))
          ),
          collections.OrderedDict(loss=[1.0, 2.0]),
          computation_types.StructWithPythonType(
              [('loss', [np.float32, np.float32])], collections.OrderedDict
          ),
      ),
  )
  def test_type_properties(
      self,
      metric_finalizers,
      unfinalized_metrics,
      local_unfinalized_metrics_type,
  ):
    aggregate_factory = sum_aggregation_factory.SumThenFinalizeFactory(
        metric_finalizers
    )
    self.assertIsInstance(
        aggregate_factory, factory.UnweightedAggregationFactory
    )
    process = aggregate_factory.create(local_unfinalized_metrics_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    expected_state_type = computation_types.FederatedType(
        ((), local_unfinalized_metrics_type), placements.SERVER
    )
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type
    )
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

    finalized_metrics_type = _get_finalized_metrics_type(
        metric_finalizers, unfinalized_metrics
    )
    result_value_type = computation_types.FederatedType(
        (finalized_metrics_type, finalized_metrics_type), placements.SERVER
    )
    measurements_type = computation_types.FederatedType((), placements.SERVER)
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            unfinalized_metrics=computation_types.FederatedType(
                local_unfinalized_metrics_type, placements.CLIENTS
            ),
        ),
        result=measured_process.MeasuredProcessOutput(
            expected_state_type, result_value_type, measurements_type
        ),
    )
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type)
    )

  @parameterized.named_parameters(
      ('float', 1.0),
      (
          'list',
          [tf.function(func=lambda x: x), tf.function(func=lambda x: x + 1)],
      ),
  )
  def test_incorrect_finalizers_type_raises(self, bad_finalizers):
    with self.assertRaises(TypeError):
      sum_aggregation_factory.SumThenFinalizeFactory(bad_finalizers)

  @parameterized.named_parameters(
      (
          'federated_type',
          computation_types.FederatedType(np.float32, placements.SERVER),
      ),
      ('function_type', computation_types.FunctionType(None, ())),
      ('sequence_type', computation_types.SequenceType(np.float32)),
  )
  def test_incorrect_unfinalized_metrics_type_raises(
      self, bad_unfinalized_metrics_type
  ):
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x)
    )
    aggregate_factory = sum_aggregation_factory.SumThenFinalizeFactory(
        metric_finalizers
    )
    with self.assertRaisesRegex(
        TypeError, 'Expected .*`tff.types.StructWithPythonType`'
    ):
      aggregate_factory.create(bad_unfinalized_metrics_type)

  def test_finalizers_and_unfinalized_metrics_type_mismatch_raises(self):
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x)
    )
    aggregate_factory = sum_aggregation_factory.SumThenFinalizeFactory(
        metric_finalizers
    )
    local_unfinalized_metrics_type = computation_types.StructWithPythonType(
        collections.OrderedDict(
            x=computation_types.TensorType(shape=[None, 2], dtype=np.float32),
            y=computation_types.TensorType(shape=[None, 1], dtype=np.float32),
        ),
        container_type=collections.OrderedDict,
    )
    with self.assertRaisesRegex(
        ValueError, 'The metric names in `metric_finalizers` do not match those'
    ):
      aggregate_factory.create(local_unfinalized_metrics_type)

  def test_unfinalized_metrics_type_and_initial_values_mismatch_raises(self):
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x)
    )
    local_unfinalized_metrics_type = computation_types.StructWithPythonType(
        [('num_examples', np.float32)], collections.OrderedDict
    )
    initial_unfinalized_metrics = collections.OrderedDict(num_examples=[1.0])
    aggregate_factory = sum_aggregation_factory.SumThenFinalizeFactory(
        metric_finalizers,
        initial_unfinalized_metrics=initial_unfinalized_metrics,
    )
    with self.assertRaisesRegex(
        TypeError,
        'member constituents of the mapped value are of incompatible type',
    ):
      aggregate_factory.create(local_unfinalized_metrics_type)


class SumThenFinalizeFactoryExecutionTest(tf.test.TestCase):

  def test_sum_then_finalize_metrics(self):
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x),
        loss=tf.function(func=lambda x: tf.math.divide_no_nan(x[0], x[1])),
        custom_sum=tf.function(
            func=lambda x: tf.add_n(map(tf.math.reduce_sum, x))
        ),
    )
    local_unfinalized_metrics = collections.OrderedDict(
        num_examples=1.0,
        loss=[2.0, 1.0],
        custom_sum=[
            np.array(1.0, np.float32),
            np.array([1.0, 1.0], np.float32),
        ],
    )
    local_unfinalized_metrics_type = computation_types.StructWithPythonType(
        [
            ('num_examples', np.float32),
            ('loss', [np.float32, np.float32]),
            (
                'custom_sum',
                [np.float32, computation_types.TensorType(np.float32, [2])],
            ),
        ],
        collections.OrderedDict,
    )
    aggregate_factory = sum_aggregation_factory.SumThenFinalizeFactory(
        metric_finalizers
    )
    process = aggregate_factory.create(local_unfinalized_metrics_type)

    state = process.initialize()
    _, unfinalized_metrics_accumulators = state
    expected_unfinalized_metrics_accumulators = collections.OrderedDict(
        num_examples=0.0,
        loss=[0.0, 0.0],
        custom_sum=[tf.constant(0.0), tf.constant([0.0, 0.0])],
    )
    tf.nest.map_structure(
        self.assertAllEqual,
        unfinalized_metrics_accumulators,
        expected_unfinalized_metrics_accumulators,
    )

    client_data = [local_unfinalized_metrics, local_unfinalized_metrics]
    output = process.next(state, client_data)
    _, unfinalized_metrics_accumulators = output.state
    current_round_metrics, total_rounds_metrics = output.result
    expected_unfinalized_metrics_accumulators = collections.OrderedDict(
        num_examples=2.0,
        loss=[4.0, 2.0],
        custom_sum=[tf.constant(2.0), tf.constant([2.0, 2.0])],
    )
    tf.nest.map_structure(
        self.assertAllEqual,
        unfinalized_metrics_accumulators,
        expected_unfinalized_metrics_accumulators,
    )
    self.assertEqual(
        current_round_metrics,
        collections.OrderedDict(
            num_examples=2.0, loss=2.0, custom_sum=tf.constant(6.0)
        ),
    )
    self.assertEqual(
        total_rounds_metrics,
        collections.OrderedDict(
            num_examples=2.0, loss=2.0, custom_sum=tf.constant(6.0)
        ),
    )

  def test_sum_then_finalize_metrics_with_initial_values(self):
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x),
        loss=tf.function(func=lambda x: tf.math.divide_no_nan(x[0], x[1])),
    )
    local_unfinalized_metrics = collections.OrderedDict(
        num_examples=1.0, loss=[2.0, 1.0]
    )
    local_unfinalized_metrics_type = computation_types.StructWithPythonType(
        [
            ('num_examples', np.float32),
            ('loss', [np.float32, np.float32]),
        ],
        collections.OrderedDict,
    )
    initial_unfinalized_metrics = collections.OrderedDict(
        num_examples=2.0, loss=[3.0, 2.0]
    )
    aggregate_factory = sum_aggregation_factory.SumThenFinalizeFactory(
        metric_finalizers,
        initial_unfinalized_metrics=initial_unfinalized_metrics,
    )
    process = aggregate_factory.create(local_unfinalized_metrics_type)

    state = process.initialize()
    _, unfinalized_metrics_accumulators = state
    self.assertEqual(
        unfinalized_metrics_accumulators, initial_unfinalized_metrics
    )

    client_data = [local_unfinalized_metrics, local_unfinalized_metrics]
    output = process.next(state, client_data)
    _, unfinalized_metrics_accumulators = output.state
    current_round_metrics, total_rounds_metrics = output.result
    self.assertEqual(
        unfinalized_metrics_accumulators,
        collections.OrderedDict(num_examples=4.0, loss=[7.0, 4.0]),
    )
    self.assertEqual(
        current_round_metrics,
        collections.OrderedDict(num_examples=2.0, loss=2.0),
    )
    self.assertEqual(
        total_rounds_metrics,
        collections.OrderedDict(num_examples=4.0, loss=1.75),
    )

  def test_secure_sum_then_finalize_metrics(self):
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x),
        loss=tf.function(func=lambda x: tf.math.divide_no_nan(x[0], x[1])),
        custom_sum=tf.function(
            func=lambda x: tf.add_n(map(tf.math.reduce_sum, x))
        ),
    )
    local_unfinalized_metrics = collections.OrderedDict(
        num_examples=1,
        loss=[2.0, 1.0],
        custom_sum=[
            np.array(101.0, np.float32),
            np.array([1.0, 1.0], np.float32),
        ],
    )
    local_unfinalized_metrics_type = computation_types.StructWithPythonType(
        [
            ('num_examples', np.int32),
            ('loss', [np.float32, np.float32]),
            (
                'custom_sum',
                [np.float32, computation_types.TensorType(np.float32, [2])],
            ),
        ],
        collections.OrderedDict,
    )
    secure_sum_factory = sum_aggregation_factory.SecureSumFactory()

    aggregate_factory = sum_aggregation_factory.SumThenFinalizeFactory(
        metric_finalizers, inner_summation_factory=secure_sum_factory
    )
    process = aggregate_factory.create(local_unfinalized_metrics_type)
    state = process.initialize()
    init_inner_summation_process_state, unfinalized_metrics_accumulators = state
    expected_init_unfinalized_metrics_accumulators = collections.OrderedDict(
        num_examples=0.0,
        loss=[0.0, 0.0],
        custom_sum=[tf.constant(0.0), tf.constant([0.0, 0.0])],
    )
    secure_sum_process = secure_sum_factory.create(
        local_unfinalized_metrics_type
    )
    tf.nest.map_structure(
        self.assertAllEqual,
        unfinalized_metrics_accumulators,
        expected_init_unfinalized_metrics_accumulators,
    )
    tf.nest.map_structure(
        self.assertAllEqual,
        init_inner_summation_process_state,
        secure_sum_process.initialize(),
    )

    client_data = [local_unfinalized_metrics, local_unfinalized_metrics]
    output = process.next(state, client_data)
    static_assert.assert_not_contains_unsecure_aggregation(process.next)

    _, unfinalized_metrics_accumulators = output.state
    # Inital clippling bounds for float values are [-100.0, 100.0], metric
    # values fall outside the ranges will be clipped to within the range (e.g.,
    # 101.0 in `custom_sum` will be clipped to 100.0).
    expected_unfinalized_metrics_accumulators = collections.OrderedDict(
        num_examples=2.0,
        loss=[4.0, 2.0],
        custom_sum=[tf.constant(200.0), tf.constant([2.0, 2.0])],
    )
    tf.nest.map_structure(
        self.assertAllEqual,
        unfinalized_metrics_accumulators,
        expected_unfinalized_metrics_accumulators,
    )

    current_round_metrics, total_rounds_metrics = output.result
    self.assertEqual(
        current_round_metrics,
        collections.OrderedDict(
            num_examples=2.0, loss=2.0, custom_sum=tf.constant(204.0)
        ),
    )
    self.assertEqual(
        total_rounds_metrics,
        collections.OrderedDict(
            num_examples=2.0, loss=2.0, custom_sum=tf.constant(204.0)
        ),
    )

    self.assertEqual(
        output.measurements,
        collections.OrderedDict([
            (
                _DEFAULT_FLOAT_FACTORY_KEY,
                collections.OrderedDict(
                    secure_upper_clipped_count=2,
                    secure_lower_clipped_count=0,
                    secure_upper_threshold=100.0,
                    secure_lower_threshold=-100.0,
                ),
            ),
            (
                _DEFAULT_INT_FACTORY_KEY,
                collections.OrderedDict(
                    secure_upper_clipped_count=0,
                    secure_lower_clipped_count=0,
                    secure_upper_threshold=1048575,
                    secure_lower_threshold=0,
                ),
            ),
        ]),
    )

if __name__ == '__main__':
  execution_contexts.set_sync_test_cpp_execution_context()
  tf.test.main()
