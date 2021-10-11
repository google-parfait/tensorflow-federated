# Copyright 2021, Google LLC.
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

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.program import structure_utils


class FlattenTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('none', None, {'': None}),
      ('int', 1, {'': 1}),
      ('list', [1, 2, 3], {'0': 1, '1': 2, '2': 3}),
      ('dict', {'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'b': 2, 'c': 3}),
      ('nested_int',
       [1, [2, 2], {'a': 3}],
       {'0': 1, '1/0': 2, '1/1': 2, '2/a': 3}),
      ('tensor', tf.ones([1]), {'': tf.ones([1])}),
      ('nested_tensor',
       [tf.ones([1]), [tf.ones([1]), tf.ones([1])]],
       {'0': tf.ones([1]), '1/0': tf.ones([1]), '1/1': tf.ones([1])}),
  )
  # pyformat: enable
  def test_returns_result(self, structure, expected_result):
    actual_result = structure_utils.flatten(structure)

    self.assertEqual(actual_result, expected_result)


if __name__ == '__main__':
  absltest.main()
