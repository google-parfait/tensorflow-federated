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


class MaterializeValueTest(
    parameterized.TestCase, unittest.IsolatedAsyncioTestCase, tf.test.TestCase
):

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

      # materializable value references
      ('materializable_value_reference_tensor',
       program_test_utils.TestMaterializableValueReference(1),
       1),
      ('materializable_value_reference_sequence',
       program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       tf.data.Dataset.from_tensor_slices([1, 2, 3])),

      # structures
      ('list',
       [True, 1, 'a', program_test_utils.TestMaterializableValueReference(2)],
       [True, 1, 'a', 2]),
      ('list_empty', [], []),
      ('list_nested',
       [[True, 1, 'a', program_test_utils.TestMaterializableValueReference(2)],
        [3]],
       [[True, 1, 'a', 2], [3]]),
      ('dict',
       {
           'a': True,
           'b': 1,
           'c': 'a',
           'd': program_test_utils.TestMaterializableValueReference(2),
       },
       {'a': True, 'b': 1, 'c': 'a', 'd': 2}),
      ('dict_empty', {}, {}),
      ('dict_nested',
       {
           'x': {
               'a': True,
               'b': 1,
               'c': 'a',
               'd': program_test_utils.TestMaterializableValueReference(2),
           },
           'y': {
               'a': 3,
           },
       },
       {'x': {'a': True, 'b': 1, 'c': 'a', 'd': 2}, 'y': {'a': 3}}),
      ('named_tuple',
       program_test_utils.TestNamedTuple1(
           a=True,
           b=1,
           c='a',
           d=program_test_utils.TestMaterializableValueReference(2)),
       program_test_utils.TestNamedTuple1(a=True, b=1, c='a', d=2)),
      ('named_tuple_nested',
       program_test_utils.TestNamedTuple3(
           x=program_test_utils.TestNamedTuple1(
               a=True,
               b=1,
               c='a',
               d=program_test_utils.TestMaterializableValueReference(2)),
           y=program_test_utils.TestNamedTuple2(3)),
       program_test_utils.TestNamedTuple3(
           x=program_test_utils.TestNamedTuple1(a=True, b=1, c='a', d=2),
           y=program_test_utils.TestNamedTuple2(3))),
      ('attrs',
       program_test_utils.TestAttrs1(
           a=True,
           b=1,
           c='a',
           d=program_test_utils.TestMaterializableValueReference(2)),
       program_test_utils.TestAttrs1(a=True, b=1, c='a', d=2)),
      ('attrs_nested',
       program_test_utils.TestAttrs3(
           x=program_test_utils.TestAttrs1(
               a=True,
               b=1,
               c='a',
               d=program_test_utils.TestMaterializableValueReference(2)),
           y=program_test_utils.TestAttrs2(3)),
       program_test_utils.TestAttrs3(
           x=program_test_utils.TestAttrs1(a=True, b=1, c='a', d=2),
           y=program_test_utils.TestAttrs2(3))),
  )
  # pyformat: enable
  async def test_returns_value(self, value, expected_value):
    actual_value = await value_reference.materialize_value(value)

    program_test_utils.assert_types_equal(actual_value, expected_value)
    if isinstance(actual_value, tf.data.Dataset) and isinstance(
        expected_value, tf.data.Dataset
    ):
      actual_value = list(actual_value)
      expected_value = list(expected_value)
    self.assertAllEqual(actual_value, expected_value)


if __name__ == '__main__':
  absltest.main()
