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
from typing import Optional, Union

from absl import logging
from absl.testing import parameterized
import grpc

import tensorflow as tf
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import secure
from tensorflow_federated.python.aggregators import sum_factory
from tensorflow_federated.python.analytics.heavy_hitters.iblt import chunkers
from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_factory
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types

# Convenience Aliases
_CharacterEncoding = chunkers.CharacterEncoding

DATA = [
    (
        ['seattle', 'hello', 'world', 'bye'],
        [[1, 2, 3], [4, 5, 1], [1, 1, 1], [-5, 2, 9]],
    ),
    (['hi', 'seattle'], [[2, 3, 4], [-5, -5, -5]]),
    (
        ['good', 'morning', 'hi', 'bye'],
        [[3, 3, 8], [-1, -5, -6], [0, 0, 0], [3, 1, 8]],
    ),
]

VALUE_TYPE = computation_types.SequenceType(
    collections.OrderedDict(
        key=tf.string,
        value=computation_types.TensorType(shape=(3,), dtype=tf.int64),
    )
)

AGGREGATED_DATA = {
    'seattle': [-4, -3, -2],
    'hello': [4, 5, 1],
    'world': [1, 1, 1],
    'hi': [2, 3, 4],
    'good': [3, 3, 8],
    'morning': [-1, -5, -6],
    'bye': [-2, 3, 17],
}

DEFAULT_REPETITIONS = 3
WHIMSY_SEED = 42


def _generate_client_data(
    input_structure: list[tuple[list[Union[str, bytes]], list[list[int]]]]
) -> list[tf.data.Dataset]:
  client_data = []
  for input_strings, string_values in input_structure:
    client = collections.OrderedDict([
        (
            iblt_factory.DATASET_KEY,
            tf.constant(input_strings, dtype=tf.string),
        ),
        (
            iblt_factory.DATASET_VALUE,
            tf.constant(string_values, dtype=tf.int64),
        ),
    ])
    client_data.append(tf.data.Dataset.from_tensor_slices(client))
  return client_data


CLIENT_DATA = _generate_client_data(DATA)


class IbltFactoryTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    execution_contexts.set_sync_test_cpp_execution_context()

  def test_capacity_validation(self):
    with self.assertRaisesRegex(ValueError, 'capacity'):
      iblt_factory.IbltFactory(
          capacity=0, string_max_bytes=10, repetitions=3, seed=0
      )
    with self.assertRaisesRegex(ValueError, 'capacity'):
      iblt_factory.IbltFactory(
          capacity=-1, string_max_bytes=10, repetitions=3, seed=0
      )
    # Should not raise
    iblt_factory.IbltFactory(
        capacity=1, string_max_bytes=10, repetitions=3, seed=0
    )

  def test_string_max_bytes_validation(self):
    with self.assertRaisesRegex(ValueError, 'string_max_bytes'):
      iblt_factory.IbltFactory(
          string_max_bytes=0, capacity=10, repetitions=3, seed=0
      )
    with self.assertRaisesRegex(ValueError, 'string_max_bytes'):
      iblt_factory.IbltFactory(
          string_max_bytes=-1, capacity=10, repetitions=3, seed=0
      )
    # Should not raise
    iblt_factory.IbltFactory(
        string_max_bytes=1, capacity=10, repetitions=3, seed=0
    )

  def test_repetitions_validation(self):
    with self.assertRaisesRegex(ValueError, 'repetitions'):
      iblt_factory.IbltFactory(
          repetitions=0, capacity=10, string_max_bytes=10, seed=0
      )
    with self.assertRaisesRegex(ValueError, 'repetitions'):
      iblt_factory.IbltFactory(
          repetitions=2, capacity=10, string_max_bytes=10, seed=0
      )
    # Should not raise
    iblt_factory.IbltFactory(
        repetitions=3, capacity=10, string_max_bytes=10, seed=0
    )

  @parameterized.named_parameters(
      (
          'scalar',
          computation_types.SequenceType(
              computation_types.TensorType(shape=(), dtype=tf.int64)
          ),
      ),
      (
          'list',
          computation_types.SequenceType(
              computation_types.TensorType(shape=(3,), dtype=tf.int64)
          ),
      ),
      (
          'dict_wrong_key',
          computation_types.SequenceType(
              collections.OrderedDict([
                  ('foo', tf.int64),
                  (
                      iblt_factory.DATASET_VALUE,
                      computation_types.TensorType(shape=(1,), dtype=tf.int64),
                  ),
              ])
          ),
      ),
      (
          'dict_extra_key',
          computation_types.SequenceType(
              collections.OrderedDict([
                  ('bar', tf.int64),
                  (iblt_factory.DATASET_KEY, tf.int64),
                  (
                      iblt_factory.DATASET_VALUE,
                      computation_types.TensorType(shape=(1,), dtype=tf.int64),
                  ),
              ])
          ),
      ),
      (
          'dict_int64_int64',
          computation_types.SequenceType(
              collections.OrderedDict([
                  (iblt_factory.DATASET_KEY, tf.int64),
                  (
                      iblt_factory.DATASET_VALUE,
                      computation_types.TensorType(shape=(1,), dtype=tf.int64),
                  ),
              ])
          ),
      ),
      (
          'dict_string_int32',
          computation_types.SequenceType(
              collections.OrderedDict([
                  (iblt_factory.DATASET_KEY, tf.string),
                  (
                      iblt_factory.DATASET_VALUE,
                      computation_types.TensorType(shape=(1,), dtype=tf.int32),
                  ),
              ])
          ),
      ),
  )
  def test_value_type_validation(self, value_type):
    iblt_agg_factory = iblt_factory.IbltFactory(
        capacity=10, string_max_bytes=5, repetitions=3, seed=0
    )
    with self.assertRaises(ValueError):
      iblt_agg_factory.create(value_type)

  def test_string_max_bytes_error(self):
    client = collections.OrderedDict([
        (
            iblt_factory.DATASET_KEY,
            tf.constant(['thisisalongword'], dtype=tf.string),
        ),
        (iblt_factory.DATASET_VALUE, tf.constant([[1]], dtype=tf.int64)),
    ])
    value_type = computation_types.SequenceType(
        collections.OrderedDict(
            key=tf.string,
            value=computation_types.TensorType(shape=(1,), dtype=tf.int64),
        )
    )
    client_data = [tf.data.Dataset.from_tensor_slices(client)]
    iblt_agg_factory = iblt_factory.IbltFactory(
        capacity=10, string_max_bytes=5, repetitions=3, seed=0
    )
    iblt_agg_process = iblt_agg_factory.create(value_type)
    with self.assertRaises(grpc.RpcError):
      iblt_agg_process.next(iblt_agg_process.initialize(), client_data)

  @parameterized.named_parameters(
      {
          'testcase_name': 'default_factories',
          'capacity': 10,
          'string_max_bytes': 10,
          'repetitions': DEFAULT_REPETITIONS,
          'seed': 0,
      },
      {
          'testcase_name': 'sketch_secure_factory',
          'sketch_agg_factory': secure.SecureSumFactory(2**32 - 1),
          'capacity': 20,
          'string_max_bytes': 20,
          'repetitions': DEFAULT_REPETITIONS,
          'seed': 1,
      },
      {
          'testcase_name': 'tensor_value_sum_factory',
          'value_tensor_agg_factory': sum_factory.SumFactory(),
          'capacity': 100,
          'string_max_bytes': 10,
          'repetitions': 5,
          'seed': 5,
      },
      {
          'testcase_name': 'secure_sum_factories',
          'sketch_agg_factory': secure.SecureSumFactory(2**32 - 1),
          'value_tensor_agg_factory': secure.SecureSumFactory(2**32 - 1),
          'capacity': 10,
          'string_max_bytes': 10,
          'repetitions': 4,
          'seed': 5,
      },
  )
  def test_iblt_aggregation_as_expected(
      self,
      *,
      capacity: int,
      string_max_bytes: int,
      repetitions: int,
      seed: int,
      sketch_agg_factory: Optional[factory.UnweightedAggregationFactory] = None,
      value_tensor_agg_factory: Optional[
          factory.UnweightedAggregationFactory
      ] = None,
  ):
    iblt_agg_factory = iblt_factory.IbltFactory(
        sketch_agg_factory=sketch_agg_factory,
        value_tensor_agg_factory=value_tensor_agg_factory,
        capacity=capacity,
        string_max_bytes=string_max_bytes,
        repetitions=repetitions,
        seed=seed,
    )
    iblt_agg_process = iblt_agg_factory.create(VALUE_TYPE)
    process_output = iblt_agg_process.next(
        iblt_agg_process.initialize(), CLIENT_DATA
    )
    output_strings = [
        s.decode('utf-8') for s in process_output.result.output_strings
    ]
    string_values = process_output.result.string_values
    result = dict(zip(output_strings, string_values))

    self.assertCountEqual(result, AGGREGATED_DATA)

    expected_measurements = collections.OrderedDict(
        [('num_not_decoded', 0), ('sketch', ()), ('value_tensor', ())]
    )
    self.assertCountEqual(process_output.measurements, expected_measurements)

  def test_binary_string_aggregation(self):
    non_unicode_string = b'\xFF' * 5
    # Each line represents a single clients' data;
    # Each binary key has two int64 values at the corresponding tuple position.
    client_data: list[tf.data.Dataset] = _generate_client_data([
        ([b'AAA', b'BBB', non_unicode_string], [[1, 2], [3, 4], [5, 6]]),
        ([b'CCC', non_unicode_string], [[7, 8], [9, 10]]),
        ([b'AAA'], [[11, 12]]),
    ])
    expected_aggregated = {
        b'AAA': [1 + 11, 2 + 12],
        b'BBB': [3, 4],
        b'CCC': [7, 8],
        non_unicode_string: [5 + 9, 6 + 10],
    }

    iblt_agg_factory = iblt_factory.IbltFactory(
        encoding=_CharacterEncoding.UNKNOWN,
        repetitions=DEFAULT_REPETITIONS,
        seed=WHIMSY_SEED,
        capacity=2,
        string_max_bytes=5,
    )
    iblt_agg_process = iblt_agg_factory.create(
        value_type=computation_types.SequenceType(
            collections.OrderedDict(
                key=tf.string,
                value=computation_types.TensorType(
                    shape=(2,),
                    dtype=tf.int64,
                ),
            )
        )
    )
    process_output = iblt_agg_process.next(
        iblt_agg_process.initialize(), client_data
    )
    logging.info('process_output: %s', process_output)

    self.assertEqual(process_output.measurements['num_not_decoded'], 0)
    result = dict(
        zip(
            process_output.result.output_strings,
            process_output.result.string_values,
        )
    )
    self.assertCountEqual(result, expected_aggregated)


if __name__ == '__main__':
  tf.test.main()
