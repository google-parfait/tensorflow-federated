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
import itertools

from typing import Dict, List, Optional
from absl.testing import parameterized

import tensorflow as tf

from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_tff
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types

DATA = [
    ['hello', 'hey', 'hi', 'hi', 'hi', '新年快乐'],
    ['hello', 'pumpkin', 'folks', 'I am on my way'],
    ['hello', 'world', '☺️😇', 'I am on my way'],
    ['hi how are', 'you :-)', 'I will be', 'there soon', 'I am on my way'],
    ['hey', 'hi', 'hi how are', 'you :-)', 'I will be', 'pumpkin'],
    ['hello', 'worm', '新年快乐', 'Seattle', 'Seattle'],
    ['hello', 'world', 'pumpkin', 'pumpkin'],
    ['way', 'I', 'I am on my way'],
    ['way', '☺️😇', 'worm', ':-)'],
    [':-)', 'there soon'],
]


def _iblt_test_data_sampler(data: List[List[str]],
                            batch_size: int = 1) -> List[tf.data.Dataset]:
  """Returns a deterministic batched sample.

  Args:
    data: a tff.simulation.datasets.ClientData object.
    batch_size: The number of elements in each batch of the dataset.

  Returns:
    list of tf.data.Datasets.
  """
  return [
      tf.data.Dataset.from_tensor_slices(client_data).batch(batch_size)
      for client_data in data
  ]


def _execute_computation(
    data: List[List[str]],
    *,
    batch_size: int = 1,
    capacity: int = 1000,
    max_string_length: int = 10,
    repetitions: int = 3,
    seed: int = 0,
    max_heavy_hitters: Optional[int] = None,
    max_words_per_user: Optional[int] = None,
    k_anonymity: int = 1,
    secure_sum_bitwidth: Optional[int] = None,
    multi_contribution: bool = True,
) -> Dict[str, tf.Tensor]:
  """Executes one round of IBLT computation over the given datasets.

  Args:
    data: A reference to all ClientData on device.
    batch_size: The number of elements in each batch of the dataset. Defaults to
      `1`, means the input dataset is processed by `tf.data.Dataset.batch(1)`.
    capacity: Capacity of the underlying IBLT. Defaults to `1000`.
    max_string_length: Maximum length (in bytes) of an item in the IBLT. Multi-
      byte characters in the string will be truncated on byte (not character)
      boundaries. Defaults to `10`.
    repetitions: The number of repetitions in IBLT data structure (must be >=
      3). Defaults to `3`.
    seed: An integer seed for hash functions. Defaults to `0`.
    max_heavy_hitters: The maximum number of items to return. If the decoded
      results have more than this number of items, will order decreasingly by
      the estimated counts and return the top max_heavy_hitters items. Default
      max_heavy_hitters == `None`, which means to return all the heavy hitters
      in the result.
    max_words_per_user: If set, bounds the number of contributions any user can
      make to the total counts in the iblt. If not `None`, must be a positive
      integer. Defaults to `None`.
    k_anonymity: Sets the number of users required for an element's count to be
      visible. Defaults to `1`.
    secure_sum_bitwidth: The bitwidth used for secure sum. The default value is
      `None`, which disables secure sum. If not `None`, must be in the range
      `[1,62]`. See `tff.federated_secure_sum_bitwidth`.
    multi_contribution: Whether each client is allowed to contribute multiple
      counts or only a count of one for each unique word. Defaults to `True`.

  Returns:
    A dictionary containing the heavy hitter results.
  """
  one_round_computation = iblt_tff.build_iblt_computation(
      capacity=capacity,
      max_string_length=max_string_length,
      repetitions=repetitions,
      seed=seed,
      max_heavy_hitters=max_heavy_hitters,
      max_words_per_user=max_words_per_user,
      k_anonymity=k_anonymity,
      secure_sum_bitwidth=secure_sum_bitwidth,
      batch_size=batch_size,
      multi_contribution=multi_contribution)
  datasets = _iblt_test_data_sampler(data, batch_size)

  output = one_round_computation(datasets)

  heavy_hitters = output.heavy_hitters
  heavy_hitters_counts = output.heavy_hitters_counts

  heavy_hitters = [word.decode('utf-8', 'ignore') for word in heavy_hitters]

  iteration_results = collections.defaultdict(int)
  total_num_heavy_hitters = len(heavy_hitters)
  for index in range(total_num_heavy_hitters):
    iteration_results[heavy_hitters[index]] += heavy_hitters_counts[index]

  return dict(iteration_results)


class IbltTffConstructionTest(test_case.TestCase):
  """Tests the `build_iblt_computation` construction API."""

  def test_default_construction(self):
    iblt_computation = iblt_tff.build_iblt_computation()
    self.assertIsInstance(iblt_computation, computation_base.Computation)
    self.assert_types_identical(
        iblt_computation.type_signature,
        computation_types.FunctionType(
            parameter=computation_types.at_clients(
                computation_types.SequenceType(
                    computation_types.TensorType(shape=[None],
                                                 dtype=tf.string))),
            result=computation_types.at_server(
                iblt_tff.ServerOutput(
                    clients=tf.int32,
                    heavy_hitters=computation_types.TensorType(
                        shape=[None], dtype=tf.string),
                    heavy_hitters_counts=computation_types.TensorType(
                        shape=[None], dtype=tf.int64),
                    num_not_decoded=tf.int64,
                ))))

  def test_max_string_length_validation(self):
    with self.assertRaisesRegex(ValueError, 'max_string_length'):
      iblt_tff.build_iblt_computation(max_string_length=0)
    with self.assertRaisesRegex(ValueError, 'max_string_length'):
      iblt_tff.build_iblt_computation(max_string_length=-1)
    iblt_tff.build_iblt_computation(max_string_length=1)

  def test_repetitions_validation(self):
    with self.assertRaisesRegex(ValueError, 'repetitions'):
      iblt_tff.build_iblt_computation(repetitions=0)
    with self.assertRaisesRegex(ValueError, 'repetitions'):
      iblt_tff.build_iblt_computation(repetitions=2)
    iblt_tff.build_iblt_computation(repetitions=3)

  def test_max_heavy_hitters_validation(self):
    with self.assertRaisesRegex(ValueError, 'max_heavy_hitters'):
      iblt_tff.build_iblt_computation(max_heavy_hitters=0)
    with self.assertRaisesRegex(ValueError, 'max_heavy_hitters'):
      iblt_tff.build_iblt_computation(max_heavy_hitters=-1)
    iblt_tff.build_iblt_computation(max_heavy_hitters=1)
    iblt_tff.build_iblt_computation(max_heavy_hitters=None)

  def test_max_words_per_user_validation(self):
    with self.assertRaisesRegex(ValueError, 'max_words_per_user'):
      iblt_tff.build_iblt_computation(max_words_per_user=0)
    with self.assertRaisesRegex(ValueError, 'max_words_per_user'):
      iblt_tff.build_iblt_computation(max_words_per_user=-1)
    iblt_tff.build_iblt_computation(max_words_per_user=1)
    iblt_tff.build_iblt_computation(max_words_per_user=None)

  def test_k_anonymity_validation(self):
    with self.assertRaisesRegex(ValueError, 'k_anonymity'):
      iblt_tff.build_iblt_computation(k_anonymity=0)
    with self.assertRaisesRegex(ValueError, 'k_anonymity'):
      iblt_tff.build_iblt_computation(k_anonymity=-1)
    iblt_tff.build_iblt_computation(k_anonymity=1)

  def test_secure_sum_bitwidth_validation(self):
    with self.assertRaisesRegex(ValueError, 'secure_sum_bitwidth'):
      iblt_tff.build_iblt_computation(secure_sum_bitwidth=-1)
    with self.assertRaisesRegex(ValueError, 'secure_sum_bitwidth'):
      iblt_tff.build_iblt_computation(secure_sum_bitwidth=0)
    with self.assertRaisesRegex(ValueError, 'secure_sum_bitwidth'):
      iblt_tff.build_iblt_computation(secure_sum_bitwidth=63)
    with self.assertRaisesRegex(ValueError, 'secure_sum_bitwidth'):
      iblt_tff.build_iblt_computation(secure_sum_bitwidth=64)
    iblt_tff.build_iblt_computation(secure_sum_bitwidth=None)
    iblt_tff.build_iblt_computation(secure_sum_bitwidth=1)
    iblt_tff.build_iblt_computation(secure_sum_bitwidth=62)

  def test_batch_size_validation(self):
    with self.assertRaisesRegex(ValueError, 'batch_size'):
      iblt_tff.build_iblt_computation(batch_size=0)
    with self.assertRaisesRegex(ValueError, 'batch_size'):
      iblt_tff.build_iblt_computation(batch_size=-1)
    iblt_tff.build_iblt_computation(batch_size=1)

  def test_multi_contribution_validation(self):
    iblt_tff.build_iblt_computation(multi_contribution=True)
    iblt_tff.build_iblt_computation(multi_contribution=False)


class SecAggIbltTffExecutionTest(test_case.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    execution_contexts.set_test_execution_context()

  @parameterized.named_parameters(
      ('lower_cap_seed_0_batch_1', 10, 20, 3, 0, 1),
      ('higher_cap_seed_1_batch_1', 20, 30, 6, 1, 1),
      ('lower_cap_seed_0_batch_5', 10, 20, 3, 0, 5),
      ('higher_cap_seed_1_batch_5', 20, 30, 6, 1, 5),
  )
  def test_computation(self, capacity: int, max_string_length: int,
                       repetitions: int, seed: int, batch_size: int):
    results = _execute_computation(
        DATA,
        capacity=capacity,
        max_string_length=max_string_length,
        repetitions=repetitions,
        seed=seed,
        max_words_per_user=10,
        batch_size=batch_size)

    all_strings = list(itertools.chain.from_iterable(DATA))
    ground_truth = dict(collections.Counter(all_strings))

    self.assertEqual(results, ground_truth)

  def test_computation_with_max_string_length(self):
    results = _execute_computation(
        DATA,
        capacity=100,
        max_string_length=5,
        max_words_per_user=10,
        max_heavy_hitters=4,
        batch_size=1)
    self.assertEqual(results, {'hello': 5, 'pumpk': 4, 'hi': 4, 'I am ': 4})

  def test_computation_with_max_string_length_multibyte(self):
    client_data = [['七転び八起き', '取らぬ狸の皮算用', '一石二鳥'] for _ in range(10)]
    results = _execute_computation(
        client_data,
        capacity=100,
        max_string_length=3,
        max_words_per_user=10,
        max_heavy_hitters=4,
        batch_size=1)
    self.assertEqual(results, {'一': 10, '七': 10, '取': 10})

  @parameterized.named_parameters(('batch_1', 1), ('batch_5', 5))
  def test_computation_with_max_heavy_hitters(self, batch_size):
    results = _execute_computation(
        DATA,
        capacity=100,
        max_string_length=30,
        max_words_per_user=10,
        max_heavy_hitters=4,
        batch_size=batch_size)
    self.assertEqual(results, {
        'hello': 5,
        'pumpkin': 4,
        'hi': 4,
        'I am on my way': 4
    })

  @parameterized.named_parameters(
      ('batch_size_1', 1),
      ('batch_size_5', 5),
  )
  def test_computation_with_k_anonymity(self, batch_size):
    results = _execute_computation(
        DATA,
        capacity=100,
        max_string_length=30,
        max_words_per_user=10,
        k_anonymity=3,
        batch_size=batch_size)
    self.assertEqual(results, {'hello': 5, 'I am on my way': 4, 'pumpkin': 4})

  @parameterized.named_parameters(
      ('k_3_max_string_len_5', 1, 3, 5, {
          'hello': 5,
          'I am ': 4,
          'pumpk': 4
      }),
      ('k_3_max_string_len_2', 2, 3, 2, {
          'he': 7,
          'hi': 6,
          'I ': 6,
          'wo': 4,
          'pu': 4
      }),
      ('k_4_max_string_len_2', 3, 4, 2, {
          'he': 7,
          'I ': 6,
          'wo': 4
      }),
      ('k_5_max_string_len_1', 5, 5, 1, {
          'h': 13,
          'w': 6,
          'I': 7
      }),
  )
  def test_computation_with_k_anonymity_and_max_string_length(
      self, batch_size, k_anonymity, max_string_length, expected_result):
    results = _execute_computation(
        DATA,
        capacity=100,
        max_string_length=max_string_length,
        max_words_per_user=10,
        k_anonymity=k_anonymity,
        batch_size=batch_size)
    self.assertEqual(results, expected_result)

  @parameterized.named_parameters(('batch_size_1', 1), ('batch_size_5', 5))
  def test_computation_with_secure_sum_bitwidth(self, batch_size):
    capacity = 100
    max_string_length = 30
    max_words_per_user = 10
    secure_sum_bitwidth = 32

    results = _execute_computation(
        DATA,
        capacity=capacity,
        max_string_length=max_string_length,
        max_words_per_user=max_words_per_user,
        batch_size=batch_size,
        secure_sum_bitwidth=secure_sum_bitwidth)

    all_strings = list(itertools.chain.from_iterable(DATA))
    ground_truth = dict(collections.Counter(all_strings))

    self.assertEqual(results, ground_truth)


class SecAggIbltUniqueCountsTffTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    execution_contexts.set_test_execution_context()

  @parameterized.named_parameters(('batch_size_1', 10, 20, 3, 0, 1),
                                  ('batch_size_5', 20, 30, 6, 1, 5))
  def test_computation(self, capacity, max_string_length, repetitions, seed,
                       batch_size):
    results = _execute_computation(
        DATA,
        capacity=capacity,
        max_string_length=max_string_length,
        repetitions=repetitions,
        seed=seed,
        max_words_per_user=10,
        batch_size=batch_size,
        multi_contribution=False)

    unique_data = [list(set(client_data)) for client_data in DATA]
    all_strings = list(itertools.chain.from_iterable(unique_data))
    ground_truth = dict(collections.Counter(all_strings))

    self.assertEqual(results, ground_truth)

  @parameterized.named_parameters(('max_length_6', 20, 6, 3, 0, 1, {
      'hello': 5,
      'hey': 2,
      'hi': 2,
      '新年': 2,
      'pumpki': 3,
      'folks': 1,
      'I am o': 4,
      'world': 2,
      '☺️': 2,
      'hi how': 2,
      'you :-': 2,
      'I will': 2,
      'there ': 2,
      'worm': 2,
      'Seattl': 1,
      'way': 2,
      'I': 1,
      ':-)': 2
  }), ('max_length_2', 100, 2, 6, 1, 5, {
      'he': 6,
      'hi': 3,
      '': 4,
      'pu': 3,
      'fo': 1,
      'I ': 5,
      'wo': 4,
      'yo': 2,
      'th': 2,
      'Se': 1,
      'wa': 2,
      'I': 1,
      ':-': 2
  }))
  def test_computation_with_max_string_length(self, capacity, max_string_length,
                                              repetitions, seed, batch_size,
                                              expected_results):
    results = _execute_computation(
        DATA,
        capacity=capacity,
        max_string_length=max_string_length,
        repetitions=repetitions,
        seed=seed,
        max_words_per_user=10,
        batch_size=batch_size,
        multi_contribution=False)

    self.assertEqual(results, expected_results)

  @parameterized.named_parameters(('batch_size_1', 1), ('batch_size_5', 5))
  def test_computation_with_max_heavy_hitters(self, batch_size):
    results = _execute_computation(
        DATA,
        capacity=100,
        max_string_length=30,
        max_words_per_user=10,
        max_heavy_hitters=3,
        batch_size=batch_size,
        multi_contribution=False)
    self.assertEqual(results, {'hello': 5, 'I am on my way': 4, 'pumpkin': 3})

  @parameterized.named_parameters(('batch_size_1', 1), ('batch_size_5', 5))
  def test_computation_with_k_anonymity(self, batch_size):
    results = _execute_computation(
        DATA,
        capacity=100,
        max_string_length=30,
        max_words_per_user=10,
        k_anonymity=3,
        batch_size=batch_size,
        multi_contribution=False)
    self.assertEqual(results, {'hello': 5, 'I am on my way': 4, 'pumpkin': 3})

  @parameterized.named_parameters(
      ('k_3_max_string_len_5', 1, 3, 5, {
          'hello': 5,
          'I am ': 4,
          'pumpk': 3
      }),
      ('k_3_max_string_len_2', 3, 3, 2, {
          'he': 6,
          'hi': 3,
          'I ': 5,
          'pu': 3,
          'wo': 4
      }),
      ('k_4_max_string_len_2', 2, 4, 2, {
          'he': 6,
          'I ': 5,
          'wo': 4
      }),
      ('k_5_max_string_len_1', 5, 5, 1, {
          'h': 7,
          'w': 5,
          'I': 5
      }),
  )
  def test_computation_with_k_anonymity_and_max_string_length(
      self, batch_size, k_anonymity, max_string_length, expected_result):
    results = _execute_computation(
        DATA,
        capacity=100,
        max_string_length=max_string_length,
        max_words_per_user=10,
        k_anonymity=k_anonymity,
        batch_size=batch_size,
        multi_contribution=False)
    self.assertEqual(results, expected_result)

  @parameterized.named_parameters(('batch_size_1', 1), ('batch_size_5', 5))
  def test_computation_with_secure_sum_bitwidth(self, batch_size):
    results = _execute_computation(
        DATA,
        capacity=100,
        max_string_length=30,
        max_words_per_user=10,
        batch_size=batch_size,
        secure_sum_bitwidth=32,
        multi_contribution=False)

    unique_data = [list(set(client_data)) for client_data in DATA]
    all_strings = list(itertools.chain.from_iterable(unique_data))
    ground_truth = dict(collections.Counter(all_strings))

    self.assertEqual(results, ground_truth)


if __name__ == '__main__':
  tf.test.main()
