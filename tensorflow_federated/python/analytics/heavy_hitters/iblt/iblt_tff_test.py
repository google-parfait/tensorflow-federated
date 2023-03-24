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
from collections.abc import Callable
import itertools
from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_tff
from tensorflow_federated.python.core.backends.test import execution_contexts
from tensorflow_federated.python.core.impl.computation import computation_base
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_test_utils

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


def _iblt_test_data_sampler(
    data: list[list[str]], batch_size: int = 1
) -> list[tf.data.Dataset]:
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


class SamplePostProcessor:
  """A class for simulating postprocessing of decoded IBLT strings."""

  def __init__(self):
    self.suffix = 'abcdefg'

  def postprocess(self, encoded_input):
    suffixes = tf.fill(tf.shape(encoded_input), self.suffix)
    return encoded_input + suffixes


def _execute_computation(
    data: list[list[str]],
    *,
    batch_size: int = 1,
    capacity: int = 1000,
    string_max_bytes: int = 10,
    repetitions: int = 3,
    seed: int = 0,
    max_heavy_hitters: Optional[int] = None,
    max_words_per_user: Optional[int] = None,
    k_anonymity: int = 1,
    secure_sum_bitwidth: Optional[int] = None,
    multi_contribution: bool = True,
    string_postprocessor: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
) -> tuple[dict[str, tf.Tensor], tf.Tensor, tf.Tensor]:
  """Executes one round of IBLT computation over the given datasets.

  Args:
    data: A reference to all ClientData on device.
    batch_size: The number of elements in each batch of the dataset. Defaults to
      `1`, means the input dataset is processed by `tf.data.Dataset.batch(1)`.
    capacity: Capacity of the underlying IBLT. Defaults to `1000`.
    string_max_bytes: Maximum length (in bytes) of an item in the IBLT. Multi-
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
    string_postprocessor: A callable function that is run after strings are
      decoded from the IBLT in order to postprocess them. It should accept a
      single string tensor and output a single string tensor of the same shape.
      If `None`, no postprocessing is done.

  Returns:
    A tuple, with elements:
      1. A dictionary containing the heavy hitter results
      2. The count of undecoded strings
      3. The round timestamp
  """
  one_round_computation = iblt_tff.build_iblt_computation(
      capacity=capacity,
      string_max_bytes=string_max_bytes,
      repetitions=repetitions,
      seed=seed,
      max_heavy_hitters=max_heavy_hitters,
      max_words_per_user=max_words_per_user,
      k_anonymity=k_anonymity,
      secure_sum_bitwidth=secure_sum_bitwidth,
      batch_size=batch_size,
      multi_contribution=multi_contribution,
      string_postprocessor=string_postprocessor,
  )
  datasets = _iblt_test_data_sampler(data, batch_size)

  output = one_round_computation(datasets)

  heavy_hitters = output.heavy_hitters
  heavy_hitters_counts = output.heavy_hitters_counts
  heavy_hitters_unique_counts = output.heavy_hitters_unique_counts

  heavy_hitters = [word.decode('utf-8', 'ignore') for word in heavy_hitters]

  iteration_results = dict(
      zip(heavy_hitters, zip(heavy_hitters_unique_counts, heavy_hitters_counts))
  )

  return dict(iteration_results), output.num_not_decoded, output.round_timestamp


class IbltTffConstructionTest(absltest.TestCase):

  def test_default_construction(self):
    iblt_computation = iblt_tff.build_iblt_computation()
    self.assertIsInstance(iblt_computation, computation_base.Computation)
    type_test_utils.assert_types_identical(
        iblt_computation.type_signature,
        computation_types.FunctionType(
            parameter=computation_types.at_clients(
                computation_types.SequenceType(
                    computation_types.TensorType(shape=[None], dtype=tf.string)
                )
            ),
            result=computation_types.at_server(
                iblt_tff.ServerOutput(
                    clients=tf.int32,
                    heavy_hitters=computation_types.TensorType(
                        shape=[None], dtype=tf.string
                    ),
                    heavy_hitters_unique_counts=computation_types.TensorType(
                        shape=[None], dtype=tf.int64
                    ),
                    heavy_hitters_counts=computation_types.TensorType(
                        shape=[None], dtype=tf.int64
                    ),
                    num_not_decoded=tf.int64,
                    round_timestamp=tf.int64,
                )
            ),
        ),
    )

  def test_string_max_bytes_validation(self):
    with self.assertRaisesRegex(ValueError, 'string_max_bytes'):
      iblt_tff.build_iblt_computation(string_max_bytes=0)
    with self.assertRaisesRegex(ValueError, 'string_max_bytes'):
      iblt_tff.build_iblt_computation(string_max_bytes=-1)
    iblt_tff.build_iblt_computation(string_max_bytes=1)

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


class SecAggIbltTffExecutionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    execution_contexts.set_sync_test_cpp_execution_context()

  @parameterized.named_parameters(
      ('lower_cap_seed_0_batch_1', 10, 20, 3, 0, 1, None, False),
      ('higher_cap_seed_1_batch_1', 20, 30, 6, 1, 1, 32, False),
      ('lower_cap_seed_0_batch_5', 10, 20, 3, 0, 5, 50, False),
      ('higher_cap_seed_1_batch_5', 20, 30, 6, 1, 5, None, False),
      ('lower_cap_seed_0_batch_1_postprocess', 10, 20, 3, 0, 1, None, True),
      ('higher_cap_seed_1_batch_1_postprocess', 20, 30, 6, 1, 1, 32, True),
      ('lower_cap_seed_0_batch_5_postprocess', 10, 20, 3, 0, 5, 50, True),
      ('higher_cap_seed_1_batch_5_postprocess', 20, 30, 6, 1, 5, None, True),
  )
  def test_computation(
      self,
      capacity,
      string_max_bytes,
      repetitions,
      seed,
      batch_size,
      secure_sum_bitwidth,
      postprocess,
  ):
    (results, num_not_decoded, _) = _execute_computation(
        DATA,
        capacity=capacity,
        string_max_bytes=string_max_bytes,
        repetitions=repetitions,
        seed=seed,
        batch_size=batch_size,
        secure_sum_bitwidth=secure_sum_bitwidth,
        string_postprocessor=None
        if not postprocess
        else SamplePostProcessor().postprocess,
    )

    self.assertEqual(num_not_decoded, 0)

    all_strings = list(itertools.chain.from_iterable(DATA))

    # Extract the number of times each string appears and the number of clients
    # that contribute the string.
    ground_truth_raw_counts = dict(collections.Counter(all_strings))
    ground_truth_unique_counts = {
        s: sum(s in lst for lst in DATA) for s in ground_truth_raw_counts
    }

    suffix = 'abcdefg' if postprocess else ''
    ground_truth = {
        s + suffix: (ground_truth_unique_counts[s], ground_truth_raw_counts[s])
        for s in ground_truth_raw_counts
    }

    self.assertDictEqual(ground_truth, results)

  def test_computation_with_string_max_bytes(self):
    results, _, _ = _execute_computation(
        DATA,
        capacity=100,
        string_max_bytes=5,
        max_words_per_user=10,
        max_heavy_hitters=4,
        batch_size=1,
    )
    self.assertEqual(
        results,
        {'hello': (5, 5), 'pumpk': (3, 4), 'hi': (2, 4), 'I am ': (4, 4)},
    )

  def test_computation_with_string_max_bytes_multibyte(self):
    client_data = [['七転び八起き', '取らぬ狸の皮算用', '一石二鳥'] for _ in range(10)]
    results, _, _ = _execute_computation(
        client_data,
        capacity=100,
        string_max_bytes=3,
        max_words_per_user=10,
        max_heavy_hitters=4,
        batch_size=1,
    )
    self.assertEqual(results, {'一': (10, 10), '七': (10, 10), '取': (10, 10)})

  @parameterized.named_parameters(('batch_1', 1), ('batch_5', 5))
  def test_computation_with_max_heavy_hitters(self, batch_size):
    results, _, _ = _execute_computation(
        DATA,
        capacity=100,
        string_max_bytes=30,
        max_words_per_user=10,
        max_heavy_hitters=4,
        batch_size=batch_size,
    )
    self.assertEqual(
        results,
        {
            'hello': (5, 5),
            'pumpkin': (3, 4),
            'hi': (2, 4),
            'I am on my way': (4, 4),
        },
    )

  @parameterized.named_parameters(
      ('batch_size_1', 1),
      ('batch_size_5', 5),
  )
  def test_computation_with_k_anonymity(self, batch_size):
    results, _, _ = _execute_computation(
        DATA,
        capacity=100,
        string_max_bytes=30,
        max_words_per_user=10,
        k_anonymity=3,
        batch_size=batch_size,
    )
    self.assertEqual(
        results, {'hello': (5, 5), 'I am on my way': (4, 4), 'pumpkin': (3, 4)}
    )

  @parameterized.named_parameters(
      (
          'k_3_max_string_len_5',
          1,
          3,
          5,
          {'hello': (5, 5), 'I am ': (4, 4), 'pumpk': (3, 4)},
      ),
      (
          'k_3_max_string_len_2',
          2,
          3,
          2,
          {
              'he': (6, 7),
              'hi': (3, 6),
              'I ': (5, 6),
              'wo': (4, 4),
              'pu': (3, 4),
          },
      ),
      (
          'k_4_max_string_len_2',
          3,
          4,
          2,
          {'he': (6, 7), 'I ': (5, 6), 'wo': (4, 4)},
      ),
      (
          'k_5_max_string_len_1',
          5,
          5,
          1,
          {'h': (7, 13), 'w': (5, 6), 'I': (5, 7)},
      ),
  )
  def test_computation_with_k_anonymity_and_string_max_bytes(
      self, batch_size, k_anonymity, string_max_bytes, expected_result
  ):
    results, _, _ = _execute_computation(
        DATA,
        capacity=100,
        string_max_bytes=string_max_bytes,
        max_words_per_user=10,
        k_anonymity=k_anonymity,
        batch_size=batch_size,
    )
    self.assertEqual(results, expected_result)


class SecAggIbltUniqueCountsTffTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    execution_contexts.set_sync_test_cpp_execution_context()

  @parameterized.named_parameters(
      ('lower_cap_seed_0_batch_1', 10, 20, 3, 0, 1, None, False),
      ('higher_cap_seed_1_batch_1', 20, 30, 6, 1, 1, 32, False),
      ('lower_cap_seed_0_batch_5', 10, 20, 3, 0, 5, 50, False),
      ('higher_cap_seed_1_batch_5', 20, 30, 6, 1, 5, None, False),
      ('lower_cap_seed_0_batch_1_postprocess', 10, 20, 3, 0, 1, None, True),
      ('higher_cap_seed_1_batch_1_postprocess', 20, 30, 6, 1, 1, 32, True),
      ('lower_cap_seed_0_batch_5_postprocess', 10, 20, 3, 0, 5, 50, True),
      ('higher_cap_seed_1_batch_5_postprocess', 20, 30, 6, 1, 5, None, True),
  )
  def test_computation(
      self,
      capacity,
      string_max_bytes,
      repetitions,
      seed,
      batch_size,
      secure_sum_bitwidth,
      postprocess,
  ):
    (results, num_not_decoded, _) = _execute_computation(
        DATA,
        capacity=capacity,
        string_max_bytes=string_max_bytes,
        repetitions=repetitions,
        seed=seed,
        batch_size=batch_size,
        secure_sum_bitwidth=secure_sum_bitwidth,
        multi_contribution=False,
        string_postprocessor=None
        if not postprocess
        else SamplePostProcessor().postprocess,
    )

    self.assertEqual(num_not_decoded, 0)

    all_strings = list(itertools.chain.from_iterable(DATA))

    # Extract the number of times each string appears and the number of clients
    # that contribute the string.
    ground_truth_raw_counts = dict(collections.Counter(all_strings))
    ground_truth_unique_counts = {
        s: sum(s in lst for lst in DATA) for s in ground_truth_raw_counts
    }

    suffix = 'abcdefg' if postprocess else ''
    ground_truth = {
        s
        + suffix: (ground_truth_unique_counts[s], ground_truth_unique_counts[s])
        for s in ground_truth_unique_counts
    }

    self.assertDictEqual(ground_truth, results)

  @parameterized.named_parameters(
      (
          'max_length_6',
          20,
          6,
          3,
          0,
          1,
          {
              'hello': (5, 5),
              'hey': (2, 2),
              'hi': (2, 2),
              '新年': (2, 2),
              'pumpki': (3, 3),
              'folks': (1, 1),
              'I am o': (4, 4),
              'world': (2, 2),
              '☺️': (2, 2),
              'hi how': (2, 2),
              'you :-': (2, 2),
              'I will': (2, 2),
              'there ': (2, 2),
              'worm': (2, 2),
              'Seattl': (1, 1),
              'way': (2, 2),
              'I': (1, 1),
              ':-)': (2, 2),
          },
      ),
      (
          'max_length_2',
          100,
          2,
          6,
          1,
          5,
          {
              'he': (6, 6),
              'hi': (3, 3),
              # Both '新年快乐' and '☺️😇' become empty strings with 2 counts.
              # However, because of the way we generate dictionaries in
              # `_execute_computation`, only one empty string with count 2
              # is showing.
              '': (2, 2),
              'pu': (3, 3),
              'fo': (1, 1),
              'I ': (5, 5),
              'wo': (4, 4),
              'yo': (2, 2),
              'th': (2, 2),
              'Se': (1, 1),
              'wa': (2, 2),
              'I': (1, 1),
              ':-': (2, 2),
          },
      ),
  )
  def test_computation_with_string_max_bytes(
      self,
      capacity,
      string_max_bytes,
      repetitions,
      seed,
      batch_size,
      expected_results,
  ):
    results, _, _ = _execute_computation(
        DATA,
        capacity=capacity,
        string_max_bytes=string_max_bytes,
        repetitions=repetitions,
        seed=seed,
        max_words_per_user=10,
        batch_size=batch_size,
        multi_contribution=False,
    )

    self.assertEqual(results, expected_results)

  @parameterized.named_parameters(('batch_size_1', 1), ('batch_size_5', 5))
  def test_computation_with_max_heavy_hitters(self, batch_size):
    results, _, _ = _execute_computation(
        DATA,
        capacity=100,
        string_max_bytes=30,
        max_words_per_user=10,
        max_heavy_hitters=3,
        batch_size=batch_size,
        multi_contribution=False,
    )
    self.assertEqual(
        results, {'hello': (5, 5), 'I am on my way': (4, 4), 'pumpkin': (3, 3)}
    )

  @parameterized.named_parameters(('batch_size_1', 1), ('batch_size_5', 5))
  def test_computation_with_k_anonymity(self, batch_size):
    results, _, _ = _execute_computation(
        DATA,
        capacity=100,
        string_max_bytes=30,
        max_words_per_user=10,
        k_anonymity=3,
        batch_size=batch_size,
        multi_contribution=False,
    )
    self.assertEqual(
        results, {'hello': (5, 5), 'I am on my way': (4, 4), 'pumpkin': (3, 3)}
    )

  @parameterized.named_parameters(
      (
          'k_3_max_string_len_5',
          1,
          3,
          5,
          {'hello': (5, 5), 'I am ': (4, 4), 'pumpk': (3, 3)},
      ),
      (
          'k_3_max_string_len_2',
          3,
          3,
          2,
          {
              'he': (6, 6),
              'hi': (3, 3),
              'I ': (5, 5),
              'pu': (3, 3),
              'wo': (4, 4),
          },
      ),
      (
          'k_4_max_string_len_2',
          2,
          4,
          2,
          {'he': (6, 6), 'I ': (5, 5), 'wo': (4, 4)},
      ),
      (
          'k_5_max_string_len_1',
          5,
          5,
          1,
          {'h': (7, 7), 'w': (5, 5), 'I': (5, 5)},
      ),
  )
  def test_computation_with_k_anonymity_and_string_max_bytes(
      self, batch_size, k_anonymity, string_max_bytes, expected_result
  ):
    results, _, _ = _execute_computation(
        DATA,
        capacity=100,
        string_max_bytes=string_max_bytes,
        max_words_per_user=10,
        k_anonymity=k_anonymity,
        batch_size=batch_size,
        multi_contribution=False,
    )
    self.assertEqual(results, expected_result)


if __name__ == '__main__':
  absltest.main()
