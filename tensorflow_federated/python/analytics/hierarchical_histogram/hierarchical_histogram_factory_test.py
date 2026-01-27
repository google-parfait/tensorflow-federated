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

import collections

from absl.testing import parameterized
import federated_language
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.aggregators import differential_privacy
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.analytics.hierarchical_histogram import build_tree_from_leaf
from tensorflow_federated.python.analytics.hierarchical_histogram import hierarchical_histogram_factory as hihi_factory
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


class TreeAggregationFactoryComputationTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('test_1_2_sub_sampling', 1, 2, 'sub-sampling', True),
      ('test_5_3_sub_sampling', 5, 3, 'sub-sampling', False),
      ('test_3_2_distinct', 3, 2, 'distinct', True),
      ('test_2_3_distinct', 2, 3, 'distinct', False),
  )
  def test_no_noise_tree_aggregation(
      self, value_shape, arity, clip_mechanism, enable_secure_sum
  ):
    agg_factory = (
        hihi_factory.create_hierarchical_histogram_aggregation_factory(
            num_bins=value_shape,
            arity=arity,
            clip_mechanism=clip_mechanism,
            dp_mechanism='no-noise',
            enable_secure_sum=enable_secure_sum,
        )
    )
    self.assertIsInstance(agg_factory, factory.UnweightedAggregationFactory)
    value_type = federated_language.to_type((np.int32, (value_shape,)))
    process = agg_factory.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    query_state_type = federated_language.StructType([
        ('arity', federated_language.TensorType(np.int32)),
        ('inner_query_state', federated_language.StructType([])),
    ])
    query_metrics_type = federated_language.StructType([])

    dp_event_type = federated_language.StructType([
        ('module_name', federated_language.TensorType(np.str_)),
        ('class_name', federated_language.TensorType(np.str_)),
    ])
    server_state_type = federated_language.FederatedType(
        differential_privacy.DPAggregatorState(
            query_state_type, (), dp_event_type, np.bool_
        ),
        federated_language.SERVER,
    )
    expected_initialize_type = federated_language.FunctionType(
        parameter=None, result=server_state_type
    )
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

    if enable_secure_sum:
      expected_measurements_dp = collections.OrderedDict(
          secure_upper_clipped_count=np.int32,
          secure_lower_clipped_count=np.int32,
          secure_upper_threshold=np.int32,
          secure_lower_threshold=np.int32,
      )
    else:
      expected_measurements_dp = ()
    expected_measurements_type = federated_language.FederatedType(
        collections.OrderedDict(
            dp_query_metrics=query_metrics_type, dp=expected_measurements_dp
        ),
        federated_language.SERVER,
    )

    tree_depth = hihi_factory._tree_depth(value_shape, arity)
    flat_tree_shape = (arity**tree_depth - 1) // (arity - 1)
    result_value_type = federated_language.to_type(
        collections.OrderedDict([
            (
                'flat_values',
                federated_language.to_type((np.int32, (flat_tree_shape,))),
            ),
            ('nested_row_splits', [(np.int64, (tree_depth + 1,))]),
        ])
    )
    value_type = federated_language.to_type((np.int32, (value_shape,)))
    expected_next_type = federated_language.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=federated_language.FederatedType(
                value_type, federated_language.CLIENTS
            ),
        ),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=federated_language.FederatedType(
                result_value_type, federated_language.SERVER
            ),
            measurements=expected_measurements_type,
        ),
    )
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type)
    )

  @parameterized.named_parameters(
      ('test_1_2_sub_sampling', 1, 2, 'sub-sampling', True),
      ('test_5_3_sub_sampling', 5, 3, 'sub-sampling', False),
      ('test_3_2_distinct', 3, 2, 'distinct', True),
      ('test_2_3_distinct', 2, 3, 'distinct', False),
  )
  def test_central_gaussian_tree_aggregation(
      self, value_shape, arity, clip_mechanism, enable_secure_sum
  ):
    agg_factory = (
        hihi_factory.create_hierarchical_histogram_aggregation_factory(
            num_bins=value_shape,
            arity=arity,
            clip_mechanism=clip_mechanism,
            dp_mechanism='central-gaussian',
            enable_secure_sum=enable_secure_sum,
        )
    )
    self.assertIsInstance(agg_factory, factory.UnweightedAggregationFactory)
    value_type = federated_language.to_type((np.int32, (value_shape,)))
    process = agg_factory.create(value_type)
    self.assertIsInstance(process, aggregation_process.AggregationProcess)

    query_state_type = federated_language.StructType([
        ('arity', federated_language.TensorType(np.int32)),
        (
            'inner_query_state',
            federated_language.StructType([
                ('l2_norm_clip', federated_language.TensorType(np.float32)),
                ('stddev', federated_language.TensorType(np.float32)),
            ]),
        ),
    ])
    query_metrics_type = federated_language.StructType([])

    # template_type is not derived from value_type in this test because the
    # outer factory converts the ints to floats before they reach the query.
    dp_event_type = federated_language.StructType([
        ('module_name', federated_language.TensorType(np.str_)),
        ('class_name', federated_language.TensorType(np.str_)),
    ])
    server_state_type = federated_language.FederatedType(
        differential_privacy.DPAggregatorState(
            query_state_type, (), dp_event_type, np.bool_
        ),
        federated_language.SERVER,
    )
    expected_initialize_type = federated_language.FunctionType(
        parameter=None, result=server_state_type
    )
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

    if enable_secure_sum:
      expected_measurements_dp = collections.OrderedDict(
          secure_upper_clipped_count=np.int32,
          secure_lower_clipped_count=np.int32,
          secure_upper_threshold=np.float32,
          secure_lower_threshold=np.float32,
      )
    else:
      expected_measurements_dp = ()
    expected_measurements_type = federated_language.FederatedType(
        collections.OrderedDict(
            dp_query_metrics=query_metrics_type, dp=expected_measurements_dp
        ),
        federated_language.SERVER,
    )
    tree_depth = hihi_factory._tree_depth(value_shape, arity)
    flat_tree_shape = (arity**tree_depth - 1) // (arity - 1)
    result_value_type = federated_language.to_type(
        collections.OrderedDict([
            (
                'flat_values',
                federated_language.to_type((np.float32, (flat_tree_shape,))),
            ),
            ('nested_row_splits', [(np.int64, (tree_depth + 1,))]),
        ])
    )
    value_type = federated_language.to_type((np.int32, (value_shape,)))
    expected_next_type = federated_language.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=federated_language.FederatedType(
                value_type, federated_language.CLIENTS
            ),
        ),
        result=measured_process.MeasuredProcessOutput(
            state=server_state_type,
            result=federated_language.FederatedType(
                result_value_type, federated_language.SERVER
            ),
            measurements=expected_measurements_type,
        ),
    )
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type)
    )


class TreeAggregationFactoryExecutionTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      (
          'non_positive_value_shape',
          0,
          2,
          'sub-sampling',
          1,
          'central-gaussian',
          1.0,
          False,
      ),
      (
          'invalid_arity',
          1,
          1,
          'sub-sampling',
          1,
          'central-gaussian',
          1.0,
          False,
      ),
      (
          'invalid_clip_mechanism',
          1,
          2,
          'invalid',
          1,
          'central-gaussian',
          1.0,
          False,
      ),
      (
          'non_positive_max_records_per_user',
          1,
          2,
          'sub-sampling',
          0,
          'central-gaussian',
          1.0,
          False,
      ),
      (
          'invalid_dp_mechanism',
          1,
          2,
          'sub-sampling',
          1,
          'invalid',
          1.0,
          False,
      ),
      (
          'negative_noise_multiplier',
          1,
          2,
          'sub-sampling',
          1,
          'central-gaussian',
          -1.0,
          False,
      ),
  )
  def test_raises_error(
      self,
      value_shape,
      arity,
      clip_mechanism,
      max_records_per_user,
      dp_mechanism,
      noise_multiplier,
      enable_secure_sum,
  ):
    with self.assertRaises(ValueError):
      hihi_factory.create_hierarchical_histogram_aggregation_factory(
          value_shape,
          arity,
          clip_mechanism,
          max_records_per_user,
          dp_mechanism,
          noise_multiplier,
          enable_secure_sum,
      )

  @parameterized.named_parameters(
      ('test_1_1_2_sub_sampling', 1, 1, 2, 'sub-sampling', True),
      # ('test_2_3_3_sub_sampling', 2, 3, 3, 'sub-sampling', False),
      # ('test_3_5_2_distinct', 3, 5, 2, 'distinct', True),
      # ('test_5_3_3_distinct', 5, 3, 3, 'distinct', False),
  )
  def test_no_noise_tree_aggregation_wo_clip(
      self, value_shape, num_clients, arity, clip_mechanism, enable_secure_sum
  ):
    client_records = []
    for _ in range(num_clients):
      client_records.append(np.arange(value_shape, dtype=int).tolist())

    agg_factory = (
        hihi_factory.create_hierarchical_histogram_aggregation_factory(
            num_bins=value_shape,
            arity=arity,
            clip_mechanism=clip_mechanism,
            max_records_per_user=5,
            dp_mechanism='no-noise',
            enable_secure_sum=enable_secure_sum,
        )
    )
    value_type = federated_language.to_type((np.int32, (value_shape,)))
    process = agg_factory.create(value_type)

    state = process.initialize()

    output = process.next(state, client_records).result
    output = tf.RaggedTensor.from_nested_row_splits(
        flat_values=output['flat_values'],
        nested_row_splits=output['nested_row_splits'],
    )

    if clip_mechanism == 'sub-sampling':
      reference_aggregated_record = (
          build_tree_from_leaf.create_hierarchical_histogram(
              np.sum(client_records, axis=0).tolist(), arity
          )
      )
    else:
      reference_aggregated_record = (
          build_tree_from_leaf.create_hierarchical_histogram(
              np.sum(np.minimum(client_records, 1), axis=0).tolist(), arity
          )
      )

    self.assertAllClose(output, reference_aggregated_record)

  @parameterized.named_parameters(
      ('test_1_1_2_sub_sampling', 1, 1, 2, 'sub-sampling', 1, True),
      ('test_2_3_3_sub_sampling', 2, 3, 3, 'sub-sampling', 2, False),
      ('test_3_5_2_distinct', 3, 5, 2, 'distinct', 3, True),
      ('test_5_3_3_distinct', 5, 3, 3, 'distinct', 2, False),
  )
  def test_no_noise_tree_aggregation_w_clip(
      self,
      value_shape,
      num_clients,
      arity,
      clip_mechanism,
      max_records_per_user,
      enable_secure_sum,
  ):
    client_records = []
    for _ in range(num_clients):
      client_records.append(np.arange(value_shape, dtype=int).tolist())

    agg_factory = (
        hihi_factory.create_hierarchical_histogram_aggregation_factory(
            num_bins=value_shape,
            arity=arity,
            clip_mechanism=clip_mechanism,
            max_records_per_user=max_records_per_user,
            dp_mechanism='no-noise',
            enable_secure_sum=enable_secure_sum,
        )
    )
    value_type = federated_language.to_type((np.int32, (value_shape,)))
    process = agg_factory.create(value_type)

    state = process.initialize()
    output = process.next(state, client_records).result
    output = tf.RaggedTensor.from_nested_row_splits(
        flat_values=output['flat_values'],
        nested_row_splits=output['nested_row_splits'],
    )

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
    else:
      self.fail(f'Unexpected `clip_mechanism` found: {clip_mechanism}.')

    for layer in range(hihi_factory._tree_depth(value_shape, arity)):
      self.assertAllClose(tf.math.reduce_sum(output[layer]), expected_l1_norm)

  @parameterized.named_parameters(
      ('test_1_1_2_sub_sampling', 1, 1, 2, 'sub-sampling', 0.1, True),
      ('test_2_3_3_sub_sampling', 2, 3, 3, 'sub-sampling', 1.0, False),
      ('test_3_5_2_distinct', 3, 5, 2, 'distinct', 5.0, True),
      ('test_5_3_3_distinct', 5, 3, 3, 'distinct', 10.0, False),
  )
  def test_central_gaussian_tree_aggregation_wo_clip(
      self,
      value_shape,
      num_clients,
      arity,
      clip_mechanism,
      noise_multiplier,
      enable_secure_sum,
  ):
    client_records = []
    for _ in range(num_clients):
      client_records.append(np.arange(value_shape, dtype=int).tolist())

    agg_factory = (
        hihi_factory.create_hierarchical_histogram_aggregation_factory(
            num_bins=value_shape,
            arity=arity,
            clip_mechanism=clip_mechanism,
            max_records_per_user=5,
            dp_mechanism='central-gaussian',
            noise_multiplier=noise_multiplier,
            enable_secure_sum=enable_secure_sum,
        )
    )
    value_type = federated_language.to_type((np.int32, (value_shape,)))
    process = agg_factory.create(value_type)

    state = process.initialize()
    output = process.next(state, client_records).result
    output = tf.RaggedTensor.from_nested_row_splits(
        flat_values=output['flat_values'],
        nested_row_splits=output['nested_row_splits'],
    )

    if clip_mechanism == 'sub-sampling':
      reference_aggregated_record = (
          build_tree_from_leaf.create_hierarchical_histogram(
              np.sum(client_records, axis=0).astype(float).tolist(), arity
          )
      )
    else:
      reference_aggregated_record = (
          build_tree_from_leaf.create_hierarchical_histogram(
              np.sum(np.minimum(client_records, 1), axis=0)
              .astype(float)
              .tolist(),
              arity,
          )
      )

    # 300 is a rough estimation of six-sigma considering the effect of the L2
    # norm bound and the privacy composition.
    self.assertAllClose(
        output, reference_aggregated_record, atol=300.0 * noise_multiplier
    )

  @parameterized.named_parameters(
      ('test_1_1_2_sub_sampling', 1, 1, 2, 'sub-sampling', 1, 0.1, True),
      ('test_2_3_3_sub_sampling', 2, 3, 3, 'sub-sampling', 2, 1.0, False),
      ('test_3_5_2_distinct', 3, 5, 2, 'distinct', 3, 5.0, True),
      ('test_5_3_3_distinct', 5, 3, 3, 'distinct', 2, 10.0, False),
  )
  def test_central_gaussian_tree_aggregation_w_clip(
      self,
      value_shape,
      num_clients,
      arity,
      clip_mechanism,
      max_records_per_user,
      noise_multiplier,
      enable_secure_sum,
  ):
    client_records = []
    for _ in range(num_clients):
      client_records.append(np.arange(value_shape, dtype=int).tolist())

    agg_factory = (
        hihi_factory.create_hierarchical_histogram_aggregation_factory(
            num_bins=value_shape,
            arity=arity,
            clip_mechanism=clip_mechanism,
            max_records_per_user=max_records_per_user,
            dp_mechanism='central-gaussian',
            noise_multiplier=noise_multiplier,
            enable_secure_sum=enable_secure_sum,
        )
    )
    value_type = federated_language.to_type((np.int32, (value_shape,)))
    process = agg_factory.create(value_type)

    state = process.initialize()
    output = process.next(state, client_records).result
    output = tf.RaggedTensor.from_nested_row_splits(
        flat_values=output['flat_values'],
        nested_row_splits=output['nested_row_splits'],
    )

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
    else:
      self.fail(f'Unexpected `clip_mechanism` found: {clip_mechanism}.')

    # 300 is a rough estimation of six-sigma considering the effect of the L2
    # norm bound and the privacy composition.
    for layer in range(hihi_factory._tree_depth(value_shape, arity)):
      self.assertAllClose(
          tf.math.reduce_sum(output[layer]),
          expected_l1_norm,
          atol=300.0 * np.sqrt(arity**layer) * noise_multiplier,
      )


if __name__ == '__main__':
  execution_contexts.set_sync_test_cpp_execution_context()
  tf.test.main()
