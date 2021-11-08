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

import functools
from typing import Dict, Optional, Union

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_lib


def _graph_and_eager_test(test_fn):

  @functools.wraps(test_fn)
  def wrapped_test_fn(*args, **kwargs):
    # In the eager context
    test_fn(*args, **kwargs)
    # In a graph context
    with tf.Graph().as_default():
      test_fn(*args, **kwargs)

  return wrapped_test_fn


class IbltTest(tf.test.TestCase, parameterized.TestCase):

  def _get_decoded_results_by_get_freq_estimates_tf(
      self,
      iblt_table: tf.Tensor,
      capacity: int,
      string_max_length: int,
      *,
      seed: int = 0,
      repetitions: int = iblt_lib.DEFAULT_REPETITIONS,
      hash_family: Optional[str] = None,
      hash_family_params: Optional[Dict[str, Union[int, float]]] = None,
      dtype=tf.int64,
      field_size: int = iblt_lib.DEFAULT_FIELD_SIZE,
  ) -> Dict[str, int]:
    iblt_decoder = iblt_lib.IbltDecoder(
        iblt=iblt_table,
        capacity=capacity,
        string_max_length=string_max_length,
        seed=seed,
        repetitions=repetitions,
        hash_family=hash_family,
        hash_family_params=hash_family_params,
        dtype=dtype,
        field_size=field_size,
    )

    decoding_graph = iblt_decoder.get_freq_estimates_tf()
    out_strings, out_counts, num_not_decoded = self.evaluate(decoding_graph)
    counter = dict(
        zip(
            [
                # Set 'ignore' in `.decode()` to ignore decoding error because
                # the strings are trimmed when they are encoded, and the
                # trimming might cut in the middle of a multi-byte utf-8
                # character.
                string.decode('utf-8', 'ignore')
                for string in out_strings.tolist()
            ],
            out_counts.tolist()))
    if num_not_decoded:
      counter[None] = num_not_decoded
    return counter

  def _get_decoded_results(
      self,
      iblt_table: tf.Tensor,
      capacity: int,
      string_max_length: int,
      *,
      seed: int = 0,
      repetitions: int = iblt_lib.DEFAULT_REPETITIONS,
      hash_family: Optional[str] = None,
      hash_family_params: Optional[Dict[str, Union[int, float]]] = None,
      dtype=tf.int64,
      field_size: int = iblt_lib.DEFAULT_FIELD_SIZE,
  ) -> Dict[str, int]:
    decoding_graph = iblt_lib.decode_iblt_tf(
        iblt=iblt_table,
        capacity=capacity,
        string_max_length=string_max_length,
        seed=seed,
        repetitions=repetitions,
        hash_family=hash_family,
        hash_family_params=hash_family_params,
        dtype=dtype,
        field_size=field_size)
    out_strings, out_counts, num_not_decoded = self.evaluate(decoding_graph)
    counter = dict(
        zip(
            [
                # Set 'ignore' in `.decode()` to ignore decoding error because
                # the strings are trimmed when they are encoded, and the
                # trimming might cut in the middle of a multi-byte utf-8
                # character.
                string.decode('utf-8', 'ignore')
                for string in out_strings.tolist()
            ],
            out_counts.tolist()))
    if num_not_decoded:
      counter[None] = num_not_decoded

    counter_by_get_freq_estimates_tf = self._get_decoded_results_by_get_freq_estimates_tf(
        iblt_table=iblt_table,
        capacity=capacity,
        string_max_length=string_max_length,
        seed=seed,
        repetitions=repetitions,
        hash_family=hash_family,
        hash_family_params=hash_family_params,
        dtype=dtype,
        field_size=field_size)
    self.assertAllClose(counter, counter_by_get_freq_estimates_tf)

    return counter

  @_graph_and_eager_test
  def test_decode_string_from_chunks(self):
    capacity = 10
    string_max_length = 7
    repetitions = 3
    seed = 0
    iblt_encoder = iblt_lib.IbltEncoder(
        capacity, string_max_length, seed=seed, dtype=tf.int64)
    input_strings_list = [
        '2019', 'seattle', 'mtv', 'heavy', 'hitters', '新年快乐',
        '☺️😇'
    ]
    input_strings = tf.constant(input_strings_list, dtype=tf.string)
    chunks, _ = iblt_encoder.chunker.encode_tensorflow(input_strings)
    iblt_table = np.zeros(
        [repetitions, iblt_encoder.table_size, iblt_encoder.num_chunks + 2])
    iblt_decoder = iblt_lib.IbltDecoder(
        iblt=iblt_table,
        capacity=capacity,
        string_max_length=string_max_length,
        repetitions=repetitions,
        seed=seed)
    for i in range(chunks.shape[0]):
      for j in range(len(bytes(input_strings_list[i], 'utf-8'))):
        if not tf.executing_eagerly():
          decoded_string = self.evaluate(
              iblt_decoder.decode_string_from_chunks(chunks[i]))
        else:
          decoded_string = iblt_decoder.decode_string_from_chunks(
              chunks[i]).numpy()
        if j < len(decoded_string):
          self.assertEqual(
              bytes(input_strings_list[i], 'utf-8')[j], decoded_string[j])

  @_graph_and_eager_test
  def test_iblt_encode_and_decode(self):
    capacity = 10
    string_max_length = 12
    repetitions = 3
    seed = 0

    iblt_encoder = iblt_lib.IbltEncoder(
        capacity, string_max_length, seed=seed, dtype=tf.int64)
    input_strings_list = [
        '2019', 'seattle', 'heavy', 'hitters', 'क', '☺️', 'has space',
        'has, comma', '新年快乐', '☺️😇'
    ]
    input_strings = tf.constant(input_strings_list, dtype=tf.string)
    iblt_table = iblt_encoder.compute_iblt(input_strings)
    strings_with_frequency = self._get_decoded_results(
        iblt_table=iblt_table,
        capacity=capacity,
        string_max_length=string_max_length,
        repetitions=repetitions,
        seed=seed)
    self.assertCountEqual(input_strings_list, strings_with_frequency.keys())

  @_graph_and_eager_test
  def test_iblt_with_coupled_hash_edges(self):
    capacity = 10
    string_max_length = 12
    repetitions = 3
    seed = 0
    hash_family = 'coupled'
    hash_family_params = {'rescale_factor': 4}

    iblt_encoder = iblt_lib.IbltEncoder(
        capacity,
        string_max_length,
        hash_family=hash_family,
        hash_family_params=hash_family_params,
        seed=seed,
        dtype=tf.int64)
    input_strings_list = [
        '2019', 'seattle', 'heavy', 'hitters', 'क', '☺️', 'has space',
        'has, comma', '新年快乐', '☺️😇'
    ]
    input_strings = tf.constant(input_strings_list, dtype=tf.string)
    iblt_table = iblt_encoder.compute_iblt(input_strings)
    strings_with_frequency = self._get_decoded_results(
        iblt_table=iblt_table,
        capacity=capacity,
        string_max_length=string_max_length,
        repetitions=repetitions,
        hash_family=hash_family,
        hash_family_params=hash_family_params,
        seed=seed)
    self.assertCountEqual(input_strings_list, strings_with_frequency.keys())

  @parameterized.named_parameters(
      {
          'testcase_name': 'large_field_size',
          'field_size': 2**62 - 57
      },
      {
          'testcase_name': 'int32',
          'dtype': tf.int32,
      },
      {
          'testcase_name': 'zero_capacity',
          'capacity': 0,
      },
      {
          'testcase_name': 'small_capacity',
          'capacity': 1,
      },
  )
  @_graph_and_eager_test
  def test_iblt_tensorflow(self,
                           capacity=10,
                           dtype=tf.int64,
                           field_size=iblt_lib.DEFAULT_FIELD_SIZE):
    string_max_length = 12
    repetitions = 3
    seed = 0

    iblt_encoder = iblt_lib.IbltEncoder(
        capacity,
        string_max_length,
        seed=seed,
        field_size=field_size,
        dtype=dtype)
    input_strings_list = [
        '2019', 'seattle', 'heavy', 'hitters', 'क', '☺️', 'has space',
        'has, comma', '新年快乐', '☺️😇'
    ]
    input_strings = tf.constant(input_strings_list, dtype=tf.string)
    iblt_table = iblt_encoder.compute_iblt(input_strings)
    strings_with_frequency = self._get_decoded_results(
        iblt_table=iblt_table,
        capacity=capacity,
        string_max_length=string_max_length,
        repetitions=repetitions,
        field_size=field_size,
        dtype=dtype,
        seed=seed)
    self.assertCountEqual(input_strings_list, strings_with_frequency.keys())

  @_graph_and_eager_test
  def test_iblt_trim_strings_above_max_length(self):
    capacity = 10
    string_max_length = 4
    repetitions = 3
    seed = 0
    iblt_encoder = iblt_lib.IbltEncoder(
        capacity,
        string_max_length,
        seed=seed,
        drop_strings_above_max_length=False,
        dtype=tf.int64)
    input_strings_list = [
        '2019', 'seattle', 'heavy', 'hitters', 'क', '☺️', 'has space',
        'has, comma', '新年快乐', '😇☺️'
    ]
    input_strings = tf.constant(input_strings_list, dtype=tf.string)
    iblt_table = iblt_encoder.compute_iblt(input_strings)
    strings_with_frequency = self._get_decoded_results(
        iblt_table=iblt_table,
        capacity=capacity,
        string_max_length=string_max_length,
        repetitions=repetitions,
        seed=seed)

    # The IBLT automatically chooses a larger string_max_length if some space
    # is being wasted in the field encoding. For example if field is 2**31 - 1
    # and as such we can encode 3 bytes per int, then if we choose
    # string_max_length = 4, it will automatically update it to 2*3 = 6.
    expected_decoded_strings = [
        '2019', 'seattl', 'heavy', 'hitter', 'क', '☺️', 'has sp',
        'has, c', '新年', '😇'
    ]
    self.assertCountEqual(expected_decoded_strings,
                          strings_with_frequency.keys())

  @_graph_and_eager_test
  def test_iblt_drop_strings_above_max_length(self):
    capacity = 10
    string_max_length = 3
    repetitions = 3
    seed = 0
    iblt_encoder = iblt_lib.IbltEncoder(
        capacity,
        string_max_length,
        seed=seed,
        drop_strings_above_max_length=True,
        dtype=tf.int64)
    input_strings_list = [
        '201', 'seattle', 'heavy', 'hitters', 'क', '☺️', 'has space',
        'has, comma', '新年快乐', '☺️😇'
    ]
    input_strings = tf.constant(input_strings_list, dtype=tf.string)
    iblt_table = iblt_encoder.compute_iblt(input_strings)
    strings_with_frequency = self._get_decoded_results(
        iblt_table=iblt_table,
        capacity=capacity,
        string_max_length=string_max_length,
        repetitions=repetitions,
        seed=seed)
    # ☺️, '新年快乐', '☺️😇' are filtered out as it takes more than 3 bytes to
    # encode.
    self.assertCountEqual(['201', 'क'], strings_with_frequency.keys())

  @_graph_and_eager_test
  def test_iblt_with_counts(self):
    capacity = 10
    string_max_length = 12
    repetitions = 3
    seed = 0
    iblt_encoder = iblt_lib.IbltEncoder(
        capacity,
        string_max_length,
        seed=seed,
        drop_strings_above_max_length=False,
        dtype=tf.int64)
    input_map = {
        '201': 10,
        'seattle': -1,
        'heavy': iblt_lib.DEFAULT_FIELD_SIZE,
        'hitters': 2**16,
        'क': iblt_lib.DEFAULT_FIELD_SIZE - 1,
        '☺️': 0,
        '新年快乐': 100
    }
    input_strings_list, input_counts_list = zip(*input_map.items())
    input_strings = tf.constant(input_strings_list, dtype=tf.string)
    input_counts = tf.constant(input_counts_list, dtype=tf.int64)
    iblt_table = iblt_encoder.compute_iblt(
        input_strings, input_counts=input_counts)
    strings_with_frequency = self._get_decoded_results(
        iblt_table=iblt_table,
        capacity=capacity,
        string_max_length=string_max_length,
        repetitions=repetitions,
        seed=seed)
    input_map_mod_field_size = {
        string: count % iblt_lib.DEFAULT_FIELD_SIZE
        for string, count in input_map.items()
    }
    input_map_mod_field_size_non_zero = {
        string: count
        for string, count in input_map_mod_field_size.items()
        if count != 0
    }
    self.assertCountEqual(input_map_mod_field_size_non_zero.items(),
                          strings_with_frequency.items())

  @parameterized.named_parameters(
      {
          'testcase_name': 'incorrect_string_list_rank',
          'input_strings_list': [['rank', '2', 'tensor']],
          'input_values_list': None,
          'exception_raised': ValueError,
      }, {
          'testcase_name': 'incorrect_value_list_rank',
          'input_strings_list': ['some', 'strings'],
          'input_values_list': [[1, 1]],
          'exception_raised': ValueError,
      }, {
          'testcase_name': 'unmatched_shapes',
          'input_strings_list': ['rank', '2', 'tensor'],
          'input_values_list': [1, 1],
          'exception_raised': tf.errors.InvalidArgumentError,
      }, {
          'testcase_name': 'incorrect_value_type',
          'input_strings_list': ['some', 'strings'],
          'input_values_list': [1, 1],
          'input_values_dtype': tf.int32,
          'exception_raised': TypeError,
      })
  @_graph_and_eager_test
  def test_iblt_input_checks(self,
                             input_strings_list,
                             input_values_list,
                             exception_raised,
                             input_values_dtype=tf.int64):
    capacity = 10
    string_max_length = 12
    seed = 0
    iblt_encoder = iblt_lib.IbltEncoder(
        capacity,
        string_max_length,
        seed=seed,
        drop_strings_above_max_length=False,
        dtype=tf.int64)
    input_strings = tf.constant(input_strings_list, dtype=tf.string)
    if input_values_list is not None:
      input_values = tf.constant(input_values_list, dtype=input_values_dtype)
    else:
      input_values = input_values_list
    with self.assertRaises(exception_raised):
      iblt_encoder.compute_iblt(input_strings, input_values)


if __name__ == '__main__':
  tf.test.main()
