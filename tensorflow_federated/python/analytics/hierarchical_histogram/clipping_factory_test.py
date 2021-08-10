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
"""Tests for clipping_factory."""

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.analytics.hierarchical_histogram import clipping_factory
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


class ClippingSumFactoryComputationTest(test_case.TestCase,
                                        parameterized.TestCase):

  @parameterized.named_parameters(
      ('test_sub_sampling', 'sub-sampling'),
      ('test_distinct', 'distinct'),
  )
  def test_clip(self, clip_mechanism):
    clip_factory = clipping_factory.HistogramClippingSumFactory(
        clip_mechanism, 1)
    self.assertIsInstance(clip_factory, factory.UnweightedAggregationFactory)
    value_type = computation_types.to_type((tf.int32, (2,)))
    process = clip_factory.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    param_value_type = computation_types.FederatedType(value_type,
                                                       placements.CLIENTS)
    result_value_type = computation_types.FederatedType(value_type,
                                                        placements.SERVER)
    expected_state_type = computation_types.at_server(())
    expected_measurements_type = computation_types.at_server(())

    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=expected_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type, value=param_value_type),
        result=measured_process.MeasuredProcessOutput(
            expected_state_type, result_value_type, expected_measurements_type))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))


class ClippingSumFactoryExecutionTest(test_case.TestCase,
                                      parameterized.TestCase):

  @parameterized.named_parameters(
      ('invalid_clip_mechanism', 'invalid', 1, (tf.int32, (2,))),
      ('invalid_max_records_per_user', 'sub-sampling', 0, (tf.int32, (2,))))
  def test_raises_value_error(self, clip_mechanism, max_records_per_user,
                              value_type):
    value_type = computation_types.to_type(value_type)
    with self.assertRaises(ValueError):
      clip_factory = clipping_factory.HistogramClippingSumFactory(
          clip_mechanism=clip_mechanism,
          max_records_per_user=max_records_per_user)
      clip_factory.create(value_type)

  @parameterized.named_parameters(
      ('struct_value_type', ((tf.int32, (2,)), tf.int32)),
      ('float_value_type', (tf.float32, (2,))),
  )
  def test_raises_type_error(self, value_type):
    value_type = computation_types.to_type(value_type)
    with self.assertRaises(TypeError):
      clip_factory = clipping_factory.HistogramClippingSumFactory()
      clip_factory.create(value_type)

  @parameterized.named_parameters(
      ('test_1_1', 1, 1),
      ('test_2_5', 2, 5),
      ('test_3_1', 3, 1),
      ('test_5_10', 5, 10),
  )
  def test_sub_sample_clip(self, value_shape, sample_num):

    histogram = np.arange(value_shape, dtype=int).tolist()

    clipped_histogram = clipping_factory._sub_sample_clip(
        tf.constant(histogram, dtype=tf.int32), sample_num)

    expected_l1_norm = min(np.linalg.norm(histogram, ord=1), sample_num)

    self.assertAllClose(tf.math.reduce_sum(clipped_histogram), expected_l1_norm)

  @parameterized.named_parameters(
      ('test_1_1', 1, 1),
      ('test_2_5', 2, 5),
      ('test_3_1', 3, 1),
      ('test_5_10', 5, 10),
  )
  def test_distinct_clip(self, value_shape, sample_num):

    histogram = np.arange(value_shape, dtype=int).tolist()

    clipped_histogram = clipping_factory._distinct_clip(
        tf.constant(histogram, dtype=tf.int32), sample_num)

    expected_l1_norm = min(np.linalg.norm(histogram, ord=0), sample_num)

    self.assertAllInSet(clipped_histogram, [0, 1])
    self.assertAllClose(tf.math.reduce_sum(clipped_histogram), expected_l1_norm)

  @parameterized.named_parameters(
      ('test_1_1_1', 1, 1, 1),
      ('test_2_5_5', 2, 5, 5),
      ('test_3_1_5', 3, 1, 5),
      ('test_5_10_5', 5, 10, 5),
  )
  def test_sub_sample_clip_factory(self, value_shape, max_records_per_user,
                                   num_clients):

    client_records = []
    for _ in range(num_clients):
      client_records.append(np.arange(value_shape, dtype=int).tolist())

    clip_factory = clipping_factory.HistogramClippingSumFactory(
        clip_mechanism='sub-sampling',
        max_records_per_user=max_records_per_user)
    outer_value_type = computation_types.to_type((tf.int32, (value_shape,)))
    process = clip_factory.create(outer_value_type)

    state = process.initialize()
    clipped_record = process.next(state, client_records).result

    expected_l1_norm = np.sum([
        min(np.linalg.norm(x, ord=1), max_records_per_user)
        for x in client_records
    ])

    self.assertAllClose(tf.math.reduce_sum(clipped_record), expected_l1_norm)

  @parameterized.named_parameters(
      ('test_1_1_1', 1, 1, 1),
      ('test_2_5_5', 2, 5, 5),
      ('test_3_1_5', 3, 1, 5),
      ('test_5_10_5', 5, 10, 5),
  )
  def test_distinct_clip_factory(self, value_shape, max_records_per_user,
                                 num_clients):

    client_records = []
    for _ in range(num_clients):
      client_records.append(np.arange(value_shape, dtype=int).tolist())

    clip_factory = clipping_factory.HistogramClippingSumFactory(
        clip_mechanism='distinct', max_records_per_user=max_records_per_user)

    value_type = computation_types.to_type((tf.int32, (value_shape,)))
    process = clip_factory.create(value_type)

    state = process.initialize()
    clipped_record = process.next(state, client_records).result

    expected_l1_norm = np.sum([
        min(np.linalg.norm(x, ord=0), max_records_per_user)
        for x in client_records
    ])

    self.assertAllClose(tf.math.reduce_sum(clipped_record), expected_l1_norm)


if __name__ == '__main__':
  execution_contexts.set_test_execution_context()
  test_case.main()
