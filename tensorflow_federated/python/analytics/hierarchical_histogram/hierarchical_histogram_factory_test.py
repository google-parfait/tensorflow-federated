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
"""Tests for hierarchical_histogram_factory."""

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.analytics.hierarchical_histogram import build_tree_from_leaf
from tensorflow_federated.python.analytics.hierarchical_histogram import hierarchical_histogram_factory as hihi_factory
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


class TreeAggregationFactoryComputationTest(test_case.TestCase,
                                            parameterized.TestCase):

  @parameterized.named_parameters(
      ('test_1_2_sub_sampling', 1, 2, 'sub-sampling'),
      ('test_5_3_sub_sampling', 5, 3, 'sub-sampling'),
      ('test_3_2_distinct', 3, 2, 'distinct'),
      ('test_2_3_distinct', 2, 3, 'distinct'),
  )
  def test_no_noise_tree_aggregation(self, value_shape, arity, clip_mechanism):

    agg_factory = hihi_factory.create_hierarchical_histogram_aggregation_factory(
        num_bins=value_shape,
        arity=arity,
        clip_mechanism=clip_mechanism,
        dp_mechanism='no-noise',
    )
    self.assertIsInstance(agg_factory, factory.UnweightedAggregationFactory)
    value_type = computation_types.to_type((tf.int32, (value_shape,)))
    process = agg_factory.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    query = tfp.privacy.dp_query.tree_aggregation_query.TreeRangeSumQuery(
        arity=arity,
        inner_query=tfp.privacy.dp_query.no_privacy_query.NoPrivacySumQuery())
    query_state = query.initial_global_state()
    query_state_type = type_conversions.type_from_tensors(query_state)
    query_metrics_type = type_conversions.type_from_tensors(
        query.derive_metrics(query_state))

    server_state_type = computation_types.at_server((query_state_type, ()))
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    expected_measurements_type = computation_types.at_server(
        collections.OrderedDict(dp_query_metrics=query_metrics_type, dp=()))

    tree_depth = hihi_factory._tree_depth(value_shape, arity)
    flat_tree_shape = (arity**tree_depth - 1) // (arity - 1)
    result_value_type = computation_types.to_type(
        collections.OrderedDict([
            ('flat_values',
             computation_types.to_type((tf.int32, (flat_tree_shape,)))),
            ('nested_row_splits', [(tf.int64, (tree_depth + 1,))])
        ]))
    value_type = computation_types.to_type((tf.int32, (value_shape,)))
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.at_clients(value_type)),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.at_server(result_value_type),
            measurements=expected_measurements_type))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))

  @parameterized.named_parameters(
      ('test_1_2_sub_sampling', 1, 2, 'sub-sampling'),
      ('test_5_3_sub_sampling', 5, 3, 'sub-sampling'),
      ('test_3_2_distinct', 3, 2, 'distinct'),
      ('test_2_3_distinct', 2, 3, 'distinct'),
  )
  def test_central_gaussian_tree_aggregation(self, value_shape, arity,
                                             clip_mechanism):

    agg_factory = hihi_factory.create_hierarchical_histogram_aggregation_factory(
        num_bins=value_shape,
        arity=arity,
        clip_mechanism=clip_mechanism,
        dp_mechanism='central-gaussian',
    )
    self.assertIsInstance(agg_factory, factory.UnweightedAggregationFactory)
    value_type = computation_types.to_type((tf.int32, (value_shape,)))
    process = agg_factory.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    query = tfp.privacy.dp_query.tree_aggregation_query.TreeRangeSumQuery.build_central_gaussian_query(
        l2_norm_clip=1.0, stddev=0.0)
    query_state = query.initial_global_state()
    query_state_type = type_conversions.type_from_tensors(query_state)
    query_metrics_type = type_conversions.type_from_tensors(
        query.derive_metrics(query_state))

    server_state_type = computation_types.at_server((query_state_type, ()))
    expected_initialize_type = computation_types.FunctionType(
        parameter=None, result=server_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    expected_measurements_type = computation_types.at_server(
        collections.OrderedDict(dp_query_metrics=query_metrics_type, dp=()))
    tree_depth = hihi_factory._tree_depth(value_shape, arity)
    flat_tree_shape = (arity**tree_depth - 1) // (arity - 1)
    result_value_type = computation_types.to_type(
        collections.OrderedDict([
            ('flat_values',
             computation_types.to_type((tf.float32, (flat_tree_shape,)))),
            ('nested_row_splits', [(tf.int64, (tree_depth + 1,))])
        ]))
    value_type = computation_types.to_type((tf.int32, (value_shape,)))
    expected_next_type = computation_types.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=computation_types.at_clients(value_type)),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=computation_types.at_server(result_value_type),
            measurements=expected_measurements_type))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))


class TreeAggregationFactoryExecutionTest(test_case.TestCase,
                                          parameterized.TestCase):

  @parameterized.named_parameters(
      ('non_positive_value_shape', 0, 2, 'sub-sampling', 1, 'central-gaussian',
       1.),
      ('invalid_arity', 1, 1, 'sub-sampling', 1, 'central-gaussian', 1.),
      ('invalid_clip_mechanism', 1, 2, 'invalid', 1, 'central-gaussian', 1.),
      ('non_positive_max_records_per_user', 1, 2, 'sub-sampling', 0,
       'central-gaussian', 1.),
      ('invalid_dp_mechanism', 1, 2, 'sub-sampling', 1, 'invalid', 1.),
      ('negative_noise_multiplier', 1, 2, 'sub-sampling', 1, 'central-gaussian',
       -1.),
  )
  def test_raises_error(self, value_shape, arity, clip_mechanism,
                        max_records_per_user, dp_mechanism, noise_multiplier):
    with self.assertRaises(ValueError):
      hihi_factory.create_hierarchical_histogram_aggregation_factory(
          value_shape, arity, clip_mechanism, max_records_per_user,
          dp_mechanism, noise_multiplier)

  @parameterized.named_parameters(
      ('test_1_1_2_sub_sampling', 1, 1, 2, 'sub-sampling'),
      ('test_2_3_3_sub_sampling', 2, 3, 3, 'sub-sampling'),
      ('test_3_5_2_distinct', 3, 5, 2, 'distinct'),
      ('test_5_3_3_distinct', 5, 3, 3, 'distinct'),
  )
  def test_no_noise_tree_aggregation_wo_clip(self, value_shape, num_clients,
                                             arity, clip_mechanism):
    client_records = []
    for _ in range(num_clients):
      client_records.append(np.arange(value_shape, dtype=int).tolist())

    agg_factory = hihi_factory.create_hierarchical_histogram_aggregation_factory(
        num_bins=value_shape,
        arity=arity,
        clip_mechanism=clip_mechanism,
        max_records_per_user=5,
        dp_mechanism='no-noise')
    value_type = computation_types.to_type((tf.int32, (value_shape,)))
    process = agg_factory.create(value_type)

    state = process.initialize()

    output = process.next(state, client_records).result

    if clip_mechanism == 'sub-sampling':
      reference_aggregated_record = build_tree_from_leaf.create_hierarchical_histogram(
          np.sum(client_records, axis=0).tolist(), arity)
    else:
      reference_aggregated_record = build_tree_from_leaf.create_hierarchical_histogram(
          np.sum(np.minimum(client_records, 1), axis=0).tolist(), arity)

    self.assertAllClose(output, reference_aggregated_record)

  @parameterized.named_parameters(
      ('test_1_1_2_sub_sampling', 1, 1, 2, 'sub-sampling', 1),
      ('test_2_3_3_sub_sampling', 2, 3, 3, 'sub-sampling', 2),
      ('test_3_5_2_distinct', 3, 5, 2, 'distinct', 3),
      ('test_5_3_3_distinct', 5, 3, 3, 'distinct', 2),
  )
  def test_no_noise_tree_aggregation_w_clip(self, value_shape, num_clients,
                                            arity, clip_mechanism,
                                            max_records_per_user):
    client_records = []
    for _ in range(num_clients):
      client_records.append(np.arange(value_shape, dtype=int).tolist())

    agg_factory = hihi_factory.create_hierarchical_histogram_aggregation_factory(
        num_bins=value_shape,
        arity=arity,
        clip_mechanism=clip_mechanism,
        max_records_per_user=max_records_per_user,
        dp_mechanism='no-noise')
    value_type = computation_types.to_type((tf.int32, (value_shape,)))
    process = agg_factory.create(value_type)

    state = process.initialize()
    output = process.next(state, client_records).result

    if clip_mechanism == 'sub-sampling':
      expected_l1_norm = np.sum([
          min(np.linalg.norm(x, ord=1), max_records_per_user)
          for x in client_records
      ])
    elif clip_mechanism == 'distinct':
      expected_l1_norm = np.sum([
          min(np.linalg.norm(x, ord=0), max_records_per_user)
          for x in client_records
      ])

    for layer in range(hihi_factory._tree_depth(value_shape, arity)):
      self.assertAllClose(tf.math.reduce_sum(output[layer]), expected_l1_norm)

  @parameterized.named_parameters(
      ('test_1_1_2_sub_sampling', 1, 1, 2, 'sub-sampling', 0.1),
      ('test_2_3_3_sub_sampling', 2, 3, 3, 'sub-sampling', 1.0),
      ('test_3_5_2_distinct', 3, 5, 2, 'distinct', 5.0),
      ('test_5_3_3_distinct', 5, 3, 3, 'distinct', 10.0),
  )
  def test_central_gaussian_tree_aggregation_wo_clip(self, value_shape,
                                                     num_clients, arity,
                                                     clip_mechanism,
                                                     noise_multiplier):
    client_records = []
    for _ in range(num_clients):
      client_records.append(np.arange(value_shape, dtype=int).tolist())

    agg_factory = hihi_factory.create_hierarchical_histogram_aggregation_factory(
        num_bins=value_shape,
        arity=arity,
        clip_mechanism=clip_mechanism,
        max_records_per_user=5,
        dp_mechanism='central-gaussian',
        noise_multiplier=noise_multiplier)
    value_type = computation_types.to_type((tf.int32, (value_shape,)))
    process = agg_factory.create(value_type)

    state = process.initialize()
    output = process.next(state, client_records).result

    if clip_mechanism == 'sub-sampling':
      reference_aggregated_record = build_tree_from_leaf.create_hierarchical_histogram(
          np.sum(client_records, axis=0).astype(float).tolist(), arity)
    else:
      reference_aggregated_record = build_tree_from_leaf.create_hierarchical_histogram(
          np.sum(np.minimum(client_records, 1), axis=0).astype(float).tolist(),
          arity)

    # 300 is a rough estimation of six-sigma considering the effect of the L2
    # norm bound and the privacy composition.
    self.assertAllClose(
        output, reference_aggregated_record, atol=300. * noise_multiplier)

  @parameterized.named_parameters(
      ('test_1_1_2_sub_sampling', 1, 1, 2, 'sub-sampling', 1, 0.1),
      ('test_2_3_3_sub_sampling', 2, 3, 3, 'sub-sampling', 2, 1.0),
      ('test_3_5_2_distinct', 3, 5, 2, 'distinct', 3, 5.0),
      ('test_5_3_3_distinct', 5, 3, 3, 'distinct', 2, 10.0),
  )
  def test_central_gaussian_tree_aggregation_w_clip(self, value_shape,
                                                    num_clients, arity,
                                                    clip_mechanism,
                                                    max_records_per_user,
                                                    noise_multiplier):
    client_records = []
    for _ in range(num_clients):
      client_records.append(np.arange(value_shape, dtype=int).tolist())

    agg_factory = hihi_factory.create_hierarchical_histogram_aggregation_factory(
        num_bins=value_shape,
        arity=arity,
        clip_mechanism=clip_mechanism,
        max_records_per_user=max_records_per_user,
        dp_mechanism='central-gaussian',
        noise_multiplier=noise_multiplier)
    value_type = computation_types.to_type((tf.int32, (value_shape,)))
    process = agg_factory.create(value_type)

    state = process.initialize()
    output = process.next(state, client_records).result

    if clip_mechanism == 'sub-sampling':
      expected_l1_norm = np.sum([
          min(np.linalg.norm(x, ord=1), max_records_per_user)
          for x in client_records
      ])
    elif clip_mechanism == 'distinct':
      expected_l1_norm = np.sum([
          min(np.linalg.norm(x, ord=0), max_records_per_user)
          for x in client_records
      ])

    # 300 is a rough estimation of six-sigma considering the effect of the L2
    # norm bound and the privacy composition.
    for layer in range(hihi_factory._tree_depth(value_shape, arity)):
      self.assertAllClose(
          tf.math.reduce_sum(output[layer]),
          expected_l1_norm,
          atol=300. * np.sqrt(arity**layer) * noise_multiplier)


if __name__ == '__main__':
  execution_contexts.set_test_execution_context()
  test_case.main()
