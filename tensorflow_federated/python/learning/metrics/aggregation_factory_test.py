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
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning.metrics import aggregation_factory


def _get_finalized_metrics_type(metric_finalizers, unfinalized_metrics):
  if callable(metric_finalizers):
    finalized_metrics = metric_finalizers(unfinalized_metrics)
  else:
    finalized_metrics = collections.OrderedDict(
        (metric, finalizer(unfinalized_metrics[metric]))
        for metric, finalizer in metric_finalizers.items())
  return type_conversions.type_from_tensors(finalized_metrics)


@tf.function
def _tf_mean(x):
  return tf.math.divide_no_nan(x[0], x[1])


class SumThenFinalizeFactoryComputationTest(tf.test.TestCase,
                                            parameterized.TestCase):

  @parameterized.named_parameters(
      ('scalar_metric',
       collections.OrderedDict(num_examples=tf.function(func=lambda x: x)),
       collections.OrderedDict(num_examples=1.0)),
      ('non_scalar_metric', collections.OrderedDict(loss=_tf_mean),
       collections.OrderedDict(loss=[2.0, 1.0])),
      ('callable',
       tf.function(
           lambda x: collections.OrderedDict(mean_loss=_tf_mean(x['loss']))),
       collections.OrderedDict(loss=[1.0, 2.0])))
  def test_type_properties(self, metric_finalizers, unfinalized_metrics):
    aggregate_factory = aggregation_factory.SumThenFinalizeFactory()
    self.assertIsInstance(aggregate_factory,
                          factory.UnweightedAggregationFactory)
    local_unfinalized_metrics_type = type_conversions.type_from_tensors(
        unfinalized_metrics)
    process = aggregate_factory.create(metric_finalizers,
                                       local_unfinalized_metrics_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    expected_state_type = computation_types.FederatedType(
        ((), local_unfinalized_metrics_type), placements.SERVER)
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    finalized_metrics_type = _get_finalized_metrics_type(
        metric_finalizers, unfinalized_metrics)
    result_value_type = computation_types.FederatedType(
        (finalized_metrics_type, finalized_metrics_type), placements.SERVER)
    measurements_type = computation_types.FederatedType((), placements.SERVER)
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            unfinalized_metrics=computation_types.FederatedType(
                local_unfinalized_metrics_type, placements.CLIENTS)),
        result=measured_process.MeasuredProcessOutput(expected_state_type,
                                                      result_value_type,
                                                      measurements_type))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))

  @parameterized.named_parameters(
      ('float', 1.0),
      ('list',
       [tf.function(func=lambda x: x),
        tf.function(func=lambda x: x + 1)]))
  def test_incorrect_finalizers_type_raises(self, bad_finalizers):
    local_unfinalized_metrics = collections.OrderedDict(num_examples=1.0)
    local_unfinalized_metrics_type = type_conversions.type_from_tensors(
        local_unfinalized_metrics)
    aggregate_factory = aggregation_factory.SumThenFinalizeFactory()
    with self.assertRaises(TypeError):
      aggregate_factory.create(bad_finalizers, local_unfinalized_metrics_type)

  @parameterized.named_parameters(
      ('federated_type',
       computation_types.FederatedType(tf.float32, placements.SERVER)),
      ('function_type', computation_types.FunctionType(None, ())),
      ('sequence_type', computation_types.SequenceType(tf.float32)))
  def test_incorrect_unfinalized_metrics_type_raises(
      self, bad_unfinalized_metrics_type):
    aggregate_factory = aggregation_factory.SumThenFinalizeFactory()
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x))
    with self.assertRaisesRegex(TypeError,
                                'Expected .*`tff.types.StructWithPythonType`'):
      aggregate_factory.create(metric_finalizers, bad_unfinalized_metrics_type)

  def test_finalizers_and_unfinalized_metrics_type_mismatch_raises(self):
    aggregate_factory = aggregation_factory.SumThenFinalizeFactory()
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x))
    local_unfinalized_metrics_type = computation_types.StructWithPythonType(
        collections.OrderedDict(
            x=tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
            y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32)),
        container_type=collections.OrderedDict)
    with self.assertRaisesRegex(
        ValueError,
        'The metric names in `metric_finalizers` do not match those'):
      aggregate_factory.create(metric_finalizers,
                               local_unfinalized_metrics_type)

  def test_unfinalized_metrics_type_and_initial_values_mismatch_raises(self):
    aggregate_factory = aggregation_factory.SumThenFinalizeFactory()
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x))
    local_unfinalized_metrics_type = type_conversions.type_from_tensors(
        collections.OrderedDict(num_examples=1.0))
    initial_unfinalized_metrics = collections.OrderedDict(num_examples=[1.0])
    with self.assertRaisesRegex(TypeError, 'initial unfinalized metrics type'):
      aggregate_factory.create(metric_finalizers,
                               local_unfinalized_metrics_type,
                               initial_unfinalized_metrics)


class SumThenFinalizeFactoryExecutionTest(tf.test.TestCase):

  def test_sum_then_finalize_metrics(self):
    aggregate_factory = aggregation_factory.SumThenFinalizeFactory()
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x),
        loss=tf.function(func=lambda x: tf.math.divide_no_nan(x[0], x[1])),
        custom_sum=tf.function(
            func=lambda x: tf.add_n(map(tf.math.reduce_sum, x))))
    local_unfinalized_metrics = collections.OrderedDict(
        num_examples=1.0,
        loss=[2.0, 1.0],
        custom_sum=[tf.constant(1.0), tf.constant([1.0, 1.0])])
    local_unfinalized_metrics_type = type_conversions.type_from_tensors(
        local_unfinalized_metrics)
    process = aggregate_factory.create(metric_finalizers,
                                       local_unfinalized_metrics_type)

    state = process.initialize()
    _, unfinalized_metrics_accumulators = state
    expected_unfinalized_metrics_accumulators = collections.OrderedDict(
        num_examples=0.0,
        loss=[0.0, 0.0],
        custom_sum=[tf.constant(0.0), tf.constant([0.0, 0.0])])
    tf.nest.map_structure(self.assertAllEqual, unfinalized_metrics_accumulators,
                          expected_unfinalized_metrics_accumulators)

    client_data = [local_unfinalized_metrics, local_unfinalized_metrics]
    output = process.next(state, client_data)
    _, unfinalized_metrics_accumulators = output.state
    current_round_metrics, total_rounds_metrics = output.result
    expected_unfinalized_metrics_accumulators = collections.OrderedDict(
        num_examples=2.0,
        loss=[4.0, 2.0],
        custom_sum=[tf.constant(2.0), tf.constant([2.0, 2.0])])
    tf.nest.map_structure(self.assertAllEqual, unfinalized_metrics_accumulators,
                          expected_unfinalized_metrics_accumulators)
    self.assertEqual(
        current_round_metrics,
        collections.OrderedDict(
            num_examples=2.0, loss=2.0, custom_sum=tf.constant(6.0)))
    self.assertEqual(
        total_rounds_metrics,
        collections.OrderedDict(
            num_examples=2.0, loss=2.0, custom_sum=tf.constant(6.0)))

  def test_sum_then_finalize_metrics_with_initial_values(self):
    aggregate_factory = aggregation_factory.SumThenFinalizeFactory()
    metric_finalizers = collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x),
        loss=tf.function(func=lambda x: tf.math.divide_no_nan(x[0], x[1])))
    local_unfinalized_metrics = collections.OrderedDict(
        num_examples=1.0, loss=[2.0, 1.0])
    local_unfinalized_metrics_type = type_conversions.type_from_tensors(
        local_unfinalized_metrics)
    initial_unfinalized_metrics = collections.OrderedDict(
        num_examples=2.0, loss=[3.0, 2.0])
    process = aggregate_factory.create(metric_finalizers,
                                       local_unfinalized_metrics_type,
                                       initial_unfinalized_metrics)

    state = process.initialize()
    _, unfinalized_metrics_accumulators = state
    self.assertEqual(unfinalized_metrics_accumulators,
                     initial_unfinalized_metrics)

    client_data = [local_unfinalized_metrics, local_unfinalized_metrics]
    output = process.next(state, client_data)
    _, unfinalized_metrics_accumulators = output.state
    current_round_metrics, total_rounds_metrics = output.result
    self.assertEqual(unfinalized_metrics_accumulators,
                     collections.OrderedDict(num_examples=4.0, loss=[7.0, 4.0]))
    self.assertEqual(current_round_metrics,
                     collections.OrderedDict(num_examples=2.0, loss=2.0))
    self.assertEqual(total_rounds_metrics,
                     collections.OrderedDict(num_examples=4.0, loss=1.75))


if __name__ == '__main__':
  execution_contexts.set_local_python_execution_context()
  tf.test.main()
