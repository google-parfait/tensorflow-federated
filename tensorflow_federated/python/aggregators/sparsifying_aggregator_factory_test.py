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
"""Tests for sparsify_aggregator_factory."""

import collections

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import sparsifying_aggregator_factory
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_test_utils
from tensorflow_federated.python.core.templates import measured_process


class SparsifyingAggregatorFactoryTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'single_scalar',
          computation_types.TensorType(dtype=tf.float32, shape=[]),
      ),
      (
          'structed_values',
          collections.OrderedDict(
              a=computation_types.TensorType(dtype=tf.float32, shape=[]),
              b=computation_types.TensorType(
                  dtype=tf.float32, shape=[100, 100]
              ),
          ),
      ),
  )
  def test_construct_aggregation_process(self, client_value_type):
    client_value_type = computation_types.to_type(client_value_type)
    factory = sparsifying_aggregator_factory.SparsifyingSumFactory()
    mean_process = factory.create(value_type=client_value_type)
    empty_tuple_at_server = computation_types.at_server(())
    type_test_utils.assert_types_identical(
        mean_process.initialize.type_signature,
        computation_types.FunctionType(
            parameter=None, result=empty_tuple_at_server
        ),
    )
    type_test_utils.assert_types_equivalent(
        mean_process.next.type_signature,
        computation_types.FunctionType(
            parameter=computation_types.StructType([
                ('state', empty_tuple_at_server),
                (
                    'client_values',
                    computation_types.at_clients(client_value_type),
                ),
            ]),
            result=measured_process.MeasuredProcessOutput(
                state=empty_tuple_at_server,
                result=computation_types.at_server(client_value_type),
                measurements=computation_types.at_server(
                    collections.OrderedDict(
                        client_coordinate_counts=type_conversions.structure_from_tensor_type_tree(
                            lambda _: tf.int64, client_value_type
                        ),
                        aggregate_coordinate_counts=type_conversions.structure_from_tensor_type_tree(
                            lambda _: tf.int64, client_value_type
                        ),
                    )
                ),
            ),
        ),
    )


class SparsifyingAggregatorFactoryExecutionTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    execution_contexts.set_sync_local_cpp_execution_context()

  def test_scalar_sum_no_zeros(self):
    client_value_type = computation_types.TensorType(dtype=tf.float32, shape=[])
    factory = sparsifying_aggregator_factory.SparsifyingSumFactory()
    mean_process = factory.create(value_type=client_value_type)
    empty_state = ()
    client_values = [1.0, 2.0, 3.0]
    output = mean_process.next(empty_state, client_values)
    self.assertEmpty(output.state)
    self.assertAllClose(output.result, sum(client_values))
    self.assertAllClose(
        output.measurements,
        collections.OrderedDict(
            client_coordinate_counts=0, aggregate_coordinate_counts=0
        ),
    )

  def test_single_sparse_sum(self):
    tensor_shape = [20, 20]
    client_value_type = computation_types.TensorType(
        dtype=tf.float32, shape=tensor_shape
    )
    factory = sparsifying_aggregator_factory.SparsifyingSumFactory()
    mean_process = factory.create(value_type=client_value_type)
    empty_state = ()
    # Create three clients, two with overlaping sparse data, and one that is
    # non-overlapping. Later assert three client update coordinates, and only
    # two aggregate update coordinates (due to overlap).
    num_clients = 3
    client_values = [
        np.zeros(shape=tensor_shape, dtype=np.float32)
        for _ in range(num_clients)
    ]
    client_values[0][1, 2] = 1
    for index in [1, 2]:
      client_values[index][5, 6] = 1
    output = mean_process.next(empty_state, client_values)
    self.assertEmpty(output.state)
    self.assertAllClose(output.result, np.sum(client_values, axis=0))
    self.assertAllClose(
        output.measurements,
        collections.OrderedDict(
            client_coordinate_counts=3, aggregate_coordinate_counts=2
        ),
    )

  def test_single_sparse_sum_high_threshold(self):
    tensor_shape = [20, 20]
    client_value_type = computation_types.TensorType(
        dtype=tf.float32, shape=tensor_shape
    )
    # Increase the element threshold such that nothing is sparsified.
    factory = sparsifying_aggregator_factory.SparsifyingSumFactory(
        element_threshold=500
    )
    mean_process = factory.create(value_type=client_value_type)
    empty_state = ()
    # Create three clients, two with overlaping sparse data, and one that is
    # non-overlapping. Later assert three client update coordinates, and only
    # two aggregate update coordinates (due to overlap).
    num_clients = 3
    client_values = [
        np.zeros(shape=tensor_shape, dtype=np.float32)
        for _ in range(num_clients)
    ]
    client_values[0][1, 2] = 1
    for index in [1, 2]:
      client_values[index][5, 6] = 1
    output = mean_process.next(empty_state, client_values)
    self.assertEmpty(output.state)
    self.assertAllClose(output.result, np.sum(client_values, axis=0))
    self.assertAllClose(
        output.measurements,
        collections.OrderedDict(
            # No sparse values because threshold is too high.
            client_coordinate_counts=0,
            aggregate_coordinate_counts=0,
        ),
    )

  def test_mixed_sparse_dense_structure_scalar(self):
    tensor_shape = [20, 20]
    client_value_type = computation_types.to_type(
        collections.OrderedDict(
            a=[
                computation_types.TensorType(dtype=tf.float32, shape=[]),
                computation_types.TensorType(dtype=tf.float32, shape=[]),
            ],
            b=computation_types.TensorType(
                dtype=tf.float32, shape=tensor_shape
            ),
        )
    )
    factory = sparsifying_aggregator_factory.SparsifyingSumFactory()
    mean_process = factory.create(value_type=client_value_type)
    empty_state = ()
    # Create three clients, two with overlaping sparse data, and one that is
    # separate.
    num_clients = 3
    client_values = [
        collections.OrderedDict(
            a=[float(client_id), float(client_id)],
            b=np.zeros(shape=tensor_shape, dtype=np.float32),
        )
        for client_id in range(num_clients)
    ]
    client_values[0]['b'][1, 2] = 1
    for index in [1, 2]:
      client_values[index]['b'][5, 6] = 1
    output = mean_process.next(empty_state, client_values)
    self.assertEmpty(output.state)
    expected_sum = collections.OrderedDict(
        a=[sum(range(num_clients)), sum(range(num_clients))],
        b=np.zeros(shape=tensor_shape, dtype=np.float32),
    )
    expected_sum['b'][1, 2] = 1
    expected_sum['b'][5, 6] = 2
    self.assertAllClose(output.result, expected_sum)
    self.assertAllClose(
        output.measurements,
        collections.OrderedDict(
            client_coordinate_counts=collections.OrderedDict(a=[0, 0], b=3),
            aggregate_coordinate_counts=collections.OrderedDict(a=[0, 0], b=2),
        ),
    )


if __name__ == '__main__':
  absltest.main()
