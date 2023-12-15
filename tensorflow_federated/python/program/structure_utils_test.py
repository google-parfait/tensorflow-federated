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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tree

from tensorflow_federated.python.program import program_test_utils
from tensorflow_federated.python.program import structure_utils


class FilterStructureTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      # materialized values
      ('none', None, None),
      ('bool', True, None),
      ('int', 1, None),
      ('str', 'a', None),
      ('tensor_int', tf.constant(1), None),
      ('tensor_str', tf.constant('a'), None),
      ('tensor_array', tf.constant([1] * 3), None),
      ('numpy_int', np.int32(1), None),
      ('numpy_array', np.array([1] * 3, np.int32), None),

      # materializable value references
      ('materializable_value_reference_tensor',
       program_test_utils.TestMaterializableValueReference(1),
       None),
      ('materializable_value_reference_sequence',
       program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       None),

      # serializable values
      ('serializable_value',
       program_test_utils.TestSerializable(1, 2),
       None),

      # other values
      ('attrs', program_test_utils.TestAttrs(1, 2), None),

      # structures
      ('list',
       [
           True,
           1,
           'a',
           program_test_utils.TestMaterializableValueReference(2),
           program_test_utils.TestSerializable(3, 4),
       ],
       [None, None, None, None, None]),
      ('list_empty', [], []),
      ('list_nested',
       [
           [
               True,
               1,
               'a',
               program_test_utils.TestMaterializableValueReference(2),
               program_test_utils.TestSerializable(3, 4),
           ],
           [5],
       ],
       [[None, None, None, None, None], [None]]),
      ('dict',
       {
           'a': True,
           'b': 1,
           'c': 'a',
           'd': program_test_utils.TestMaterializableValueReference(2),
           'e': program_test_utils.TestSerializable(3, 4),
       },
       {'a': None, 'b': None, 'c': None, 'd': None, 'e': None}),
      ('dict_empty', {}, {}),
      ('dict_nested',
       {
           'x': {
               'a': True,
               'b': 1,
               'c': 'a',
               'd': program_test_utils.TestMaterializableValueReference(2),
               'e': program_test_utils.TestSerializable(3, 4),
           },
           'y': {'a': 5},
       },
       {
           'x': {'a': None, 'b': None, 'c': None, 'd': None, 'e': None},
           'y': {'a': None},
       }),
      ('named_tuple',
       program_test_utils.TestNamedTuple1(
           a=True,
           b=1,
           c='a',
           d=program_test_utils.TestMaterializableValueReference(2),
           e=program_test_utils.TestSerializable(3, 4),
       ),
       program_test_utils.TestNamedTuple1(
           a=None,
           b=None,
           c=None,
           d=None,
           e=None,
       )),
      ('named_tuple_nested',
       program_test_utils.TestNamedTuple3(
           x=program_test_utils.TestNamedTuple1(
               a=True,
               b=1,
               c='a',
               d=program_test_utils.TestMaterializableValueReference(2),
               e=program_test_utils.TestSerializable(3, 4),
           ),
           y=program_test_utils.TestNamedTuple2(a=5),
       ),
       program_test_utils.TestNamedTuple3(
           x=program_test_utils.TestNamedTuple1(
               a=None,
               b=None,
               c=None,
               d=None,
               e=None,
           ),
           y=program_test_utils.TestNamedTuple2(a=None),
       )),
  )
  # pyformat: enable
  def test_returns_result(self, structure, expected_result):
    actual_result = structure_utils._filter_structure(structure)
    self.assertEqual(actual_result, expected_result)


class FlattenWithNameTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      # materialized values
      ('none', None, [('', None)]),
      ('bool', True, [('', True)]),
      ('int', 1, [('', 1)]),
      ('str', 'a', [('', 'a')]),
      ('tensor_int', tf.constant(1), [('', tf.constant(1))]),
      ('tensor_str', tf.constant('a'), [('', tf.constant('a'))]),
      ('tensor_array', tf.constant([1] * 3), [('', tf.constant([1] * 3))]),
      ('numpy_int', np.int32(1), [('', np.int32(1))]),
      ('numpy_array',
       np.array([1] * 3, np.int32),
       [('', np.array([1] * 3, np.int32))]),

      # materializable value references
      ('materializable_value_reference_tensor',
       program_test_utils.TestMaterializableValueReference(1),
       [('', program_test_utils.TestMaterializableValueReference(1))]),
      ('materializable_value_reference_sequence',
       program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])),
       [('', program_test_utils.TestMaterializableValueReference(
           tf.data.Dataset.from_tensor_slices([1, 2, 3])))]),

      # serializable values
      ('serializable_value',
       program_test_utils.TestSerializable(1, 2),
       [('', program_test_utils.TestSerializable(1, 2))]),

      # other values
      ('attrs',
       program_test_utils.TestAttrs(1, 2),
       [('', program_test_utils.TestAttrs(1, 2))]),

      # structures
      ('list',
       [
           True,
           1,
           'a',
           program_test_utils.TestMaterializableValueReference(2),
           program_test_utils.TestSerializable(3, 4),
       ],
       [
           ('0', True),
           ('1', 1),
           ('2', 'a'),
           ('3', program_test_utils.TestMaterializableValueReference(2)),
           ('4', program_test_utils.TestSerializable(3, 4)),
       ]),
      ('list_empty', [], []),
      ('list_nested',
       [
           [
               True,
               1,
               'a',
               program_test_utils.TestMaterializableValueReference(2),
               program_test_utils.TestSerializable(3, 4),
           ],
           [5],
       ],
       [
           ('0/0', True),
           ('0/1', 1),
           ('0/2', 'a'),
           ('0/3', program_test_utils.TestMaterializableValueReference(2)),
           ('0/4', program_test_utils.TestSerializable(3, 4)),
           ('1/0', 5),
       ]),
      ('dict',
       {
           'a': True,
           'b': 1,
           'c': 'a',
           'd': program_test_utils.TestMaterializableValueReference(2),
           'e': program_test_utils.TestSerializable(3, 4),
       },
       [
           ('a', True),
           ('b', 1),
           ('c', 'a'),
           ('d', program_test_utils.TestMaterializableValueReference(2)),
           ('e', program_test_utils.TestSerializable(3, 4)),
       ]),
      ('dict_empty', {}, []),
      ('dict_nested',
       {
           'x': {
               'a': True,
               'b': 1,
               'c': 'a',
               'd': program_test_utils.TestMaterializableValueReference(2),
               'e': program_test_utils.TestSerializable(3, 4),
           },
           'y': {'a': 5},
       },
       [
           ('x/a', True),
           ('x/b', 1),
           ('x/c', 'a'),
           ('x/d', program_test_utils.TestMaterializableValueReference(2)),
           ('x/e', program_test_utils.TestSerializable(3, 4)),
           ('y/a', 5),
       ]),
      ('named_tuple',
       program_test_utils.TestNamedTuple1(
           a=True,
           b=1,
           c='a',
           d=program_test_utils.TestMaterializableValueReference(2),
           e=program_test_utils.TestSerializable(3, 4),
       ),
       [
           ('a', True),
           ('b', 1),
           ('c', 'a'),
           ('d', program_test_utils.TestMaterializableValueReference(2)),
           ('e', program_test_utils.TestSerializable(3, 4)),
       ]),
      ('named_tuple_nested',
       program_test_utils.TestNamedTuple3(
           x=program_test_utils.TestNamedTuple1(
               a=True,
               b=1,
               c='a',
               d=program_test_utils.TestMaterializableValueReference(2),
               e=program_test_utils.TestSerializable(3, 4),
           ),
           y=program_test_utils.TestNamedTuple2(a=5),
       ),
       [
           ('x/a', True),
           ('x/b', 1),
           ('x/c', 'a'),
           ('x/d', program_test_utils.TestMaterializableValueReference(2)),
           ('x/e', program_test_utils.TestSerializable(3, 4)),
           ('y/a', 5),
       ]),
  )
  # pyformat: enable
  def test_returns_result(self, structure, expected_result):
    actual_result = structure_utils.flatten_with_name(structure)

    for actual_item, expected_item in zip(actual_result, expected_result):
      actual_path, actual_value = actual_item
      expected_path, expected_value = expected_item
      self.assertEqual(actual_path, expected_path)
      tree.assert_same_structure(actual_value, expected_value)
      actual_value = program_test_utils.to_python(actual_value)
      expected_value = program_test_utils.to_python(expected_value)
      self.assertEqual(actual_value, expected_value)


class MapStructureTest(absltest.TestCase):

  def test_returns_result(self):
    fn = lambda x, y: (x, y)
    structure1 = [1, 2, 3]
    structure2 = [4, 5, 6]

    result = structure_utils.map_structure(fn, structure1, structure2)
    self.assertEqual(result, [(1, 4), (2, 5), (3, 6)])

  def test_raises_value_error_with_no_structures(self):
    fn = lambda x, y: (x, y)

    with self.assertRaises(ValueError):
      structure_utils.map_structure(fn)

  def test_raises_value_error_with_different_structures(self):
    fn = lambda x, y: (x, y)
    structure1 = []
    structure2 = [1, 2, 3]

    with self.assertRaises(ValueError):
      structure_utils.map_structure(fn, structure1, structure2)

  def test_does_not_raises_type_error_with_different_types(self):
    fn = lambda x, y: (x, y)
    structure1 = [1, 2, 3]
    structure2 = (4, 5, 6)

    structure_utils.map_structure(fn, structure1, structure2, check_types=False)

  def test_raises_type_error_with_different_types(self):
    fn = lambda x, y: (x, y)
    structure1 = [1, 2, 3]
    structure2 = (4, 5, 6)

    with self.assertRaises(TypeError):
      structure_utils.map_structure(fn, structure1, structure2)


if __name__ == '__main__':
  absltest.main()
