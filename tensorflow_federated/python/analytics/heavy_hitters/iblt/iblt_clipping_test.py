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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_clipping
from tensorflow_federated.python.analytics.heavy_hitters.iblt import iblt_factory
from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.impl.types import computation_types


class IbltClippingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'basic',
          ['a', 'a', 'a', 'b', 'b', 'c'],
          10,
          True,
          1,
          20,
          False,
          {'a': 3, 'b': 2, 'c': 1},
      ),
      (
          'unique',
          ['a', 'a', 'a', 'b', 'b', 'c'],
          10,
          True,
          1,
          20,
          True,
          {'a': [3, 1], 'b': [2, 1], 'c': [1, 1]},
      ),
      (
          'max_words',
          ['a', 'ab', 'a', 'cc', 'a', 'b', 'c', 'd'],
          4,
          True,
          1,
          20,
          False,
          {'a': 2, 'ab': 1, 'cc': 1},
      ),
      (
          'single_contrib',
          ['a', 'ab', 'a', 'cc', 'a', 'b', 'c', 'd'],
          4,
          False,
          1,
          20,
          False,
          {'a': 1, 'ab': 1, 'cc': 1, 'b': 1},
      ),
      (
          'max_bytes',
          ['aaaaaaa', 'a', 'b', 'c', 'dddddd', 'a'],
          10,
          True,
          1,
          1,
          False,
          {'a': 3, 'b': 1, 'c': 1, 'd': 1},
      ),
  )
  def test_get_clipped_elements_with_counts(
      self,
      input_data,
      max_words,
      multi_contribution,
      batch_size,
      string_max_bytes,
      unique_counts,
      expected,
  ):
    ds = tf.data.Dataset.from_tensor_slices(input_data).batch(
        batch_size=batch_size
    )

    result = iblt_clipping.get_clipped_elements_with_counts(
        ds,
        max_words,
        multi_contribution,
        batch_size,
        string_max_bytes,
        unique_counts,
    )
    result_dict = {}
    dtype = list if unique_counts else int
    for x in result.as_numpy_iterator():
      result_dict[x['key'].decode('utf-8', 'ignore')] = dtype(x['value'])
    self.assertDictEqual(result_dict, expected)

  def test_incorrect_value_type(self):
    iblt_fac = iblt_factory.IbltFactory(
        capacity=100, string_max_bytes=10, repetitions=3, seed=0
    )
    clip_fac = iblt_clipping.ClippingIbltFactory(iblt_fac)
    wrong_type = computation_types.SequenceType(
        computation_types.TensorType(shape=[None], dtype=np.int32)
    )
    with self.assertRaises(ValueError):
      clip_fac.create(wrong_type)

  def test_clipping_factory(self):
    iblt_fac = iblt_factory.IbltFactory(
        capacity=100, string_max_bytes=10, repetitions=3
    )
    clip_fac = iblt_clipping.ClippingIbltFactory(
        iblt_fac, max_words_per_user=4, unique_counts=True
    )
    value_type = computation_types.SequenceType(
        computation_types.TensorType(shape=[None], dtype=np.str_)
    )
    agg_process = clip_fac.create(value_type)
    data = [
        [['a', 'a', 'a', 'a']],
        [['a', 'a', 'b', 'b']],
        [['a', 'b', 'c', 'd']],
    ]
    state = agg_process.initialize()
    process_output = agg_process.next(state, data)
    heavy_hitters = process_output.result.output_strings
    heavy_hitters_counts = process_output.result.string_values[:, 0]
    ans = dict(zip(heavy_hitters, heavy_hitters_counts))
    self.assertEqual(ans, {b'a': 7, b'b': 3, b'c': 1, b'd': 1})


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
