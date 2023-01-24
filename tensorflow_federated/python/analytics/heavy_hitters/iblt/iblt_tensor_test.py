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
"""Tests for iblt_tensor.py."""
from collections.abc import Sequence
import functools
from typing import Any, Optional, Union
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_lib
from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_tensor


def _graph_and_eager_test(test_fn):
  @functools.wraps(test_fn)
  def wrapped_test_fn(*args, **kwargs):
    # In the eager context
    test_fn(*args, **kwargs)
    # In a graph context
    with tf.Graph().as_default():
      test_fn(*args, **kwargs)

  return wrapped_test_fn


class IbltTensorTest(tf.test.TestCase, parameterized.TestCase):

  def _get_decoded_results_by_get_freq_estimates_tf(
      self,
      iblt: tf.Tensor,
      iblt_values: tf.Tensor,
      capacity: int,
      string_max_bytes: int,
      value_shape: Sequence[int],
      *,
      seed: int = 0,
      repetitions: int = iblt_lib.DEFAULT_REPETITIONS,
      hash_family: Optional[str] = None,
      hash_family_params: Optional[dict[str, Union[int, float]]] = None,
      field_size: int = iblt_lib.DEFAULT_FIELD_SIZE,
  ) -> tuple[dict[Optional[str], int], dict[Optional[str], Sequence[Any]]]:
    iblt_decoder = iblt_tensor.IbltTensorDecoder(
        iblt=iblt,
        iblt_values=iblt_values,
        capacity=capacity,
        string_max_bytes=string_max_bytes,
        value_shape=value_shape,
        seed=seed,
        repetitions=repetitions,
        hash_family=hash_family,
        hash_family_params=hash_family_params,
        field_size=field_size,
    )

    if tf.executing_eagerly():
      # `get_freq_estimates` only works in eager mode.
      return iblt_decoder.get_freq_estimates()
    else:
      out_strings, out_counts, out_tensor_values, num_not_decoded = (
          self.evaluate(iblt_decoder.get_freq_estimates_tf())
      )

      out_strings = [
          string.decode('utf-8', 'ignore') for string in out_strings.tolist()
      ]
      string_counts = dict(zip(out_strings, out_counts.tolist()))
      string_tensor_values = dict(zip(out_strings, out_tensor_values.tolist()))

      if num_not_decoded:
        string_counts[None] = num_not_decoded
      return string_counts, string_tensor_values

  def _get_decoded_results(
      self,
      iblt: tf.Tensor,
      iblt_values: tf.Tensor,
      capacity: int,
      string_max_bytes: int,
      value_shape: Sequence[int],
      *,
      seed: int = 0,
      repetitions: int = iblt_lib.DEFAULT_REPETITIONS,
      hash_family: Optional[str] = None,
      hash_family_params: Optional[dict[str, Union[int, float]]] = None,
      field_size: int = iblt_lib.DEFAULT_FIELD_SIZE,
  ) -> tuple[dict[Optional[str], int], dict[Optional[str], Sequence[Any]]]:
    decoding_graph = iblt_tensor.decode_iblt_tensor_tf(
        iblt=iblt,
        iblt_values=iblt_values,
        capacity=capacity,
        string_max_bytes=string_max_bytes,
        value_shape=value_shape,
        seed=seed,
        repetitions=repetitions,
        hash_family=hash_family,
        hash_family_params=hash_family_params,
        field_size=field_size,
    )
    out_strings, out_counts, out_tensor_values, num_not_decoded = self.evaluate(
        decoding_graph
    )
    out_strings = [
        string.decode('utf-8', 'ignore') for string in out_strings.tolist()
    ]
    string_counts = dict(zip(out_strings, out_counts.tolist()))
    string_tensor_values = dict(zip(out_strings, out_tensor_values.tolist()))

    if num_not_decoded:
      string_counts[None] = num_not_decoded

    (counts_by_get_freq_estimates_tf, tensor_value_by_get_freq_estimates_tf) = (
        self._get_decoded_results_by_get_freq_estimates_tf(
            iblt=iblt,
            iblt_values=iblt_values,
            capacity=capacity,
            string_max_bytes=string_max_bytes,
            value_shape=value_shape,
            seed=seed,
            repetitions=repetitions,
            hash_family=hash_family,
            hash_family_params=hash_family_params,
            field_size=field_size,
        )
    )
    self.assertAllClose(string_counts, counts_by_get_freq_estimates_tf)
    self.assertAllClose(
        string_tensor_values, tensor_value_by_get_freq_estimates_tf
    )
    return string_counts, string_tensor_values

  @parameterized.named_parameters(
      {
          'testcase_name': 'multilingual',
          'value_shape': (3,),
          'input_strings_list': [
              '201',
              'seattle',
              'heavy',
              'hitters',
              'क',
              '☺️',
          ],
          'input_values_list': [
              [3, 2, 1],
              [7, 3, 9],
              [tf.int64.max, 0, 0],
              [tf.int64.max, tf.int64.max, 0],
              [0, tf.int64.max, 0],
              [1, 2, 19],
          ],
          'output_strings_list': [
              '201',
              'seattle',
              'heavy',
              'hitters',
              'क',
              '☺️',
          ],
          'output_values_list': [
              [3, 2, 1],
              [7, 3, 9],
              [tf.int64.max, 0, 0],
              [tf.int64.max, tf.int64.max, 0],
              [0, tf.int64.max, 0],
              [1, 2, 19],
          ],
          'output_count_list': [1, 1, 1, 1, 1, 1],
      },
      {
          'testcase_name': 'trim_strings_above_max_length',
          'value_shape': (1,),
          'string_max_bytes': 4,
          # IBLT automatically chooses a larger `string_max_bytes` if some
          # space is being wasted in the field encoding. For example if field is
          # 2**31 - 1 and as such we can encode 3 bytes per int, then if we
          # choose `string_max_bytes = 4`, it will automatically update it to
          # 2*3 = 6.
          'input_strings_list': [
              '2022',
              'seattle',
              'heavy',
              'hitters',
              'क',
              '☺️',
              'has space',
              'has, comma',
              '新年快乐',
              '😇☺️',
          ],
          'input_values_list': [
              [0],
              [1],
              [2],
              [3],
              [4],
              [5],
              [6],
              [7],
              [8],
              [9],
          ],
          'output_strings_list': [
              '2022',
              'seattl',
              'heavy',
              'hitter',
              'क',
              '☺️',
              'has sp',
              'has, c',
              '新年',
              '😇',
          ],
          'output_values_list': [
              [0],
              [1],
              [2],
              [3],
              [4],
              [5],
              [6],
              [7],
              [8],
              [9],
          ],
          'output_count_list': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      },
      {
          'testcase_name': 'drop_strings_above_max_length',
          'value_shape': (1,),
          'string_max_bytes': 3,  # Maximum number of bytes.
          'drop_strings_above_max_length': True,
          'input_strings_list': [
              '201',
              'seattle',
              'heavy',
              'hitters',
              'क',
              '☺️',
              'has space',
              'has, comma',
              '新年快乐',
              '☺️😇',
          ],
          'input_values_list': [
              [0],
              [1],
              [2],
              [3],
              [4],
              [5],
              [6],
              [7],
              [8],
              [9],
          ],
          'output_strings_list': ['201', 'क'],
          'output_values_list': [[0], [4]],
          'output_count_list': [1, 1],
      },
      {
          'testcase_name': 'coupled_hash',
          'value_shape': (3,),
          'hash_family': 'coupled',
          'hash_family_params': {'rescale_factor': 4},
          'input_strings_list': ['201', 'seattle', '☺️😇', '新年快乐', 'क', '☺️'],
          'input_values_list': [
              [3, 2, 1],
              [7, 3, 9],
              [tf.int64.max, 0, 0],
              [tf.int64.max, tf.int64.max, 0],
              [0, tf.int64.max, 0],
              [1, 2, 19],
          ],
          'output_strings_list': ['201', 'seattle', '☺️😇', '新年快乐', 'क', '☺️'],
          'output_values_list': [
              [3, 2, 1],
              [7, 3, 9],
              [tf.int64.max, 0, 0],
              [tf.int64.max, tf.int64.max, 0],
              [0, tf.int64.max, 0],
              [1, 2, 19],
          ],
          'output_count_list': [1, 1, 1, 1, 1, 1],
      },
      {
          'testcase_name': 'duplicate_keys',
          'value_shape': (3,),
          'input_strings_list': ['201', 'seattle', '201'],
          'input_values_list': [[3, 2, 1], [7, 3, 9], [1, 1, 0]],
          'output_strings_list': ['201', 'seattle'],
          'output_values_list': [[4, 3, 1], [7, 3, 9]],
          'output_count_list': [2, 1],
      },
      {
          'testcase_name': 'rank_2_value_shape',
          'value_shape': (3, 2),
          'input_strings_list': ['201', 'seattle', '201'],
          'input_values_list': [
              [[3, 5], [2, 2], [1, 7]],
              [[7, 7], [3, 3], [9, 9]],
              [[1, 1], [1, 1], [0, 0]],
          ],
          'output_strings_list': ['201', 'seattle'],
          'output_values_list': [
              [[4, 6], [3, 3], [1, 7]],
              [[7, 7], [3, 3], [9, 9]],
          ],
          'output_count_list': [2, 1],
      },
      {
          'testcase_name': 'rank_3_value_shape',
          'value_shape': (3, 2, 5),
          'input_strings_list': ['201', 'seattle', '201'],
          'input_values_list': [
              [
                  [[9, 3, 4, 5, 5], [4, 2, 9, 8, 4]],
                  [[5, 5, 5, 4, 6], [4, 3, 7, 3, 2]],
                  [[3, 8, 1, 1, 1], [8, 4, 6, 8, 9]],
              ],
              [
                  [[7, 4, 7, 8, 5], [3, 9, 2, 2, 5]],
                  [[9, 7, 6, 8, 2], [7, 7, 8, 5, 2]],
                  [[9, 2, 9, 5, 5], [5, 5, 1, 9, 5]],
              ],
              [
                  [[4, 3, 7, 4, 5], [8, 1, 6, 9, 4]],
                  [[1, 2, 6, 2, 4], [6, 2, 8, 3, 6]],
                  [[9, 2, 7, 4, 6], [1, 3, 9, 6, 7]],
              ],
          ],
          'output_strings_list': ['201', 'seattle'],
          'output_values_list': [
              [
                  [[13, 6, 11, 9, 10], [12, 3, 15, 17, 8]],
                  [[6, 7, 11, 6, 10], [10, 5, 15, 6, 8]],
                  [[12, 10, 8, 5, 7], [9, 7, 15, 14, 16]],
              ],
              [
                  [[7, 4, 7, 8, 5], [3, 9, 2, 2, 5]],
                  [[9, 7, 6, 8, 2], [7, 7, 8, 5, 2]],
                  [[9, 2, 9, 5, 5], [5, 5, 1, 9, 5]],
              ],
          ],
          'output_count_list': [2, 1],
      },
      {
          'testcase_name': 'empty',
          'value_shape': (),
          'input_strings_list': ['201', 'seattle', '201'],
          'input_values_list': [],
          'output_strings_list': ['201', 'seattle'],
          'output_values_list': [],
          'output_count_list': [2, 1],
      },
  )
  @_graph_and_eager_test
  def test_iblt_tensor_encode_and_decode(
      self,
      value_shape,
      input_strings_list,
      input_values_list,
      output_strings_list,
      output_values_list,
      output_count_list,
      string_max_bytes=12,
      hash_family=None,
      hash_family_params=None,
      drop_strings_above_max_length=False,
  ):
    capacity = 10
    repetitions = 3
    seed = 0
    iblt_encoder = iblt_tensor.IbltTensorEncoder(
        value_shape,
        capacity=capacity,
        string_max_bytes=string_max_bytes,
        repetitions=repetitions,
        hash_family=hash_family,
        hash_family_params=hash_family_params,
        seed=seed,
        drop_strings_above_max_length=drop_strings_above_max_length,
    )

    input_strings = tf.constant(input_strings_list, dtype=tf.string)
    input_values = tf.constant(input_values_list, dtype=tf.int64)
    iblt_table, iblt_values = iblt_encoder.compute_iblt(
        input_strings, input_values=input_values
    )
    string_counts, string_tensor_values = self._get_decoded_results(
        iblt=iblt_table,
        iblt_values=iblt_values,
        value_shape=value_shape,
        capacity=capacity,
        string_max_bytes=string_max_bytes,
        repetitions=repetitions,
        hash_family=hash_family,
        hash_family_params=hash_family_params,
        seed=seed,
    )

    self.assertCountEqual(
        dict(zip(output_strings_list, output_count_list)).items(),
        string_counts.items(),
    )
    self.assertCountEqual(
        dict(zip(output_strings_list, output_values_list)).items(),
        string_tensor_values.items(),
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'incorrect_string_list_rank',
          'value_shape': (1,),
          'input_strings_list': [['rank', '2', 'tensor']],
          'input_values_list': [0, 0],
          'exception_raised': ValueError,
      },
      {
          'testcase_name': 'incorrect_value_list_rank',
          'value_shape': (1,),
          'input_strings_list': ['some', 'strings'],
          'input_values_list': [1, 1],
          'exception_raised': ValueError,
      },
      {
          'testcase_name': 'incorrect_value_list_shape_1',
          'value_shape': (2,),
          'input_strings_list': ['some', 'strings'],
          'input_values_list': [[1], [1]],
          'exception_raised': ValueError,
      },
      {
          'testcase_name': 'incorrect_value_list_shape_2',
          'value_shape': (2,),
          'input_strings_list': ['some', 'strings'],
          'input_values_list': [[[1, 3]], [[1, 1]]],
          'exception_raised': ValueError,
      },
      {
          'testcase_name': 'incorrect_value_list_len',
          'value_shape': (2,),
          'input_strings_list': ['some', 'strings'],
          'input_values_list': [[1, 1]],
          'exception_raised': ValueError,
      },
      {
          'testcase_name': 'incorrect_rank_2_value_list_shape',
          'value_shape': (2, 3),
          'input_strings_list': ['some', 'strings'],
          'input_values_list': [[[1, 4], [2, 4]], [[2, 3], [4, 2]]],
          'exception_raised': ValueError,
      },
  )
  @_graph_and_eager_test
  def test_iblt_input_checks(
      self, value_shape, input_strings_list, input_values_list, exception_raised
  ):
    capacity = 10
    string_max_bytes = 12
    seed = 0
    iblt_encoder = iblt_tensor.IbltTensorEncoder(
        value_shape=value_shape,
        capacity=capacity,
        string_max_bytes=string_max_bytes,
        seed=seed,
        drop_strings_above_max_length=False,
    )
    input_strings = tf.constant(input_strings_list, dtype=tf.string)
    input_values = tf.constant(input_values_list, dtype=tf.int64)

    with self.assertRaises(exception_raised):
      iblt_encoder.compute_iblt(input_strings, input_values)


if __name__ == '__main__':
  tf.test.main()
