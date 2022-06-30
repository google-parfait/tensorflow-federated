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

import unittest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.program import program_test_utils
from tensorflow_federated.python.program import value_reference


class MaterializeValueTest(parameterized.TestCase,
                           unittest.IsolatedAsyncioTestCase, tf.test.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      # materialized values
      ('none', None, None),
      ('bool', True, True),
      ('int', 1, 1),
      ('str', 'a', 'a'),
      ('tensor_int', tf.constant(1), tf.constant(1)),
      ('tensor_str', tf.constant('a'), tf.constant('a')),
      ('tensor_array', tf.ones([3]), tf.ones([3])),
      ('numpy_int', np.int32(1), np.int32(1)),
      ('numpy_array', np.ones([3]), np.ones([3])),

      # value references
      ('materializable_value_reference_tensor',
       program_test_utils.TestMaterializableValueReference(1), 1),
      ('materializable_value_reference_sequence',
       program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       tf.data.Dataset.from_tensor_slices([1, 2, 3])),

      # structures
      ('list',
       [True, program_test_utils.TestMaterializableValueReference(1), 'a'],
       [True, 1, 'a']),
      ('list_empty', [], []),
      ('list_nested',
       [[True, program_test_utils.TestMaterializableValueReference(1)], ['a']],
       [[True, 1], ['a']]),
      ('dict',
       {'a': True,
        'b': program_test_utils.TestMaterializableValueReference(1),
        'c': 'a'},
       {'a': True, 'b': 1, 'c': 'a'}),
      ('dict_empty', {}, {}),
      ('dict_nested',
       {'x': {'a': True,
              'b': program_test_utils.TestMaterializableValueReference(1)},
        'y': {'c': 'a'}},
       {'x': {'a': True, 'b': 1}, 'y': {'c': 'a'}}),
      ('attr',
       program_test_utils.TestAttrObject2(
           True, program_test_utils.TestMaterializableValueReference(1)),
       program_test_utils.TestAttrObject2(True, 1)),
      ('attr_nested',
       program_test_utils.TestAttrObject2(
           program_test_utils.TestAttrObject2(
               True, program_test_utils.TestMaterializableValueReference(1)),
           program_test_utils.TestAttrObject1('a')),
       program_test_utils.TestAttrObject2(
           program_test_utils.TestAttrObject2(True, 1),
           program_test_utils.TestAttrObject1('a'))),
  )
  # pyformat: enable
  async def test_returns_value(self, value, expected_value):
    actual_value = await value_reference.materialize_value(value)

    program_test_utils.assert_types_equal(actual_value, expected_value)
    if (isinstance(actual_value, tf.data.Dataset) and
        isinstance(expected_value, tf.data.Dataset)):
      actual_value = list(actual_value)
      expected_value = list(expected_value)
    self.assertAllEqual(actual_value, expected_value)


if __name__ == '__main__':
  absltest.main()
