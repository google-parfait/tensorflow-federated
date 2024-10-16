# Copyright 2018, The TensorFlow Federated Authors.
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
from collections.abc import Mapping
from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import attrs
import numpy as np

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import typed_object


class _TestTypedObject(typed_object.TypedObject):

  def __init__(self, type_signature: computation_types.Type):
    self._type_signature = type_signature

  @property
  def type_signature(self) -> computation_types.Type:
    return self._type_signature


class _TestNamedTuple(NamedTuple):
  a: int
  b: int
  c: int


class InferTypeTest(parameterized.TestCase):

  def test_with_none(self):
    self.assertIsNone(type_conversions.infer_type(None))

  def test_with_typed_object(self):
    obj = _TestTypedObject(computation_types.TensorType(np.bool_))

    whimsy_type = type_conversions.infer_type(obj)
    self.assertEqual(whimsy_type.compact_representation(), 'bool')

  def test_with_scalar_int_tensor(self):
    self.assertEqual(str(type_conversions.infer_type(np.int32(1))), 'int32')
    self.assertEqual(str(type_conversions.infer_type(np.int64(1))), 'int64')
    self.assertEqual(str(type_conversions.infer_type(np.int64(-1))), 'int64')

  def test_with_scalar_bool_tensor(self):
    self.assertEqual(str(type_conversions.infer_type(np.bool_(False))), 'bool')

  def test_with_int_array_tensor(self):
    self.assertEqual(
        str(type_conversions.infer_type(np.array([10, 20], dtype=np.int32))),
        'int32[2]',
    )
    self.assertEqual(
        str(
            type_conversions.infer_type(
                np.array([0, 2**40, -(2**60), 0], dtype=np.int64)
            )
        ),
        'int64[4]',
    )

  def test_with_int(self):
    self.assertEqual(str(type_conversions.infer_type(10)), 'int32')

  def test_with_float(self):
    self.assertEqual(str(type_conversions.infer_type(0.5)), 'float32')

  def test_with_bool(self):
    self.assertEqual(str(type_conversions.infer_type(True)), 'bool')

  def test_with_string(self):
    self.assertEqual(str(type_conversions.infer_type('abc')), 'str')

  def test_with_np_int32(self):
    self.assertEqual(str(type_conversions.infer_type(np.int32(10))), 'int32')

  def test_with_np_int64(self):
    self.assertEqual(str(type_conversions.infer_type(np.int64(10))), 'int64')

  def test_with_np_float32(self):
    self.assertEqual(
        str(type_conversions.infer_type(np.float32(10))), 'float32'
    )

  def test_with_np_float64(self):
    self.assertEqual(
        str(type_conversions.infer_type(np.float64(10))), 'float64'
    )

  def test_with_np_bool(self):
    self.assertEqual(str(type_conversions.infer_type(np.bool_(True))), 'bool')

  def test_with_unicode_string(self):
    self.assertEqual(str(type_conversions.infer_type('abc')), 'str')

  def test_with_numpy_int_array(self):
    self.assertEqual(
        str(type_conversions.infer_type(np.array([10, 20]))), 'int64[2]'
    )

  def test_with_numpy_nested_int_array(self):
    self.assertEqual(
        str(type_conversions.infer_type(np.array([[10], [20]]))), 'int64[2,1]'
    )

  def test_with_numpy_float64_scalar(self):
    self.assertEqual(str(type_conversions.infer_type(np.float64(1))), 'float64')

  def test_with_int_list(self):
    t = type_conversions.infer_type([1, 2, 3])
    self.assertEqual(str(t), '<int32,int32,int32>')
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, list)

  def test_with_nested_float_list(self):
    t = type_conversions.infer_type([[0.1], [0.2], [0.3]])
    self.assertEqual(str(t), '<<float32>,<float32>,<float32>>')
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, list)

  def test_with_structure(self):
    t = type_conversions.infer_type(
        structure.Struct([
            ('a', 10),
            (None, False),
        ])
    )
    self.assertEqual(str(t), '<a=int32,bool>')
    self.assertIsInstance(t, computation_types.StructType)
    self.assertNotIsInstance(t, computation_types.StructWithPythonType)

  def test_with_nested_structure(self):
    t = type_conversions.infer_type(
        structure.Struct([
            ('a', 10),
            (
                None,
                structure.Struct([
                    (None, True),
                    (None, 0.5),
                ]),
            ),
        ])
    )
    self.assertEqual(str(t), '<a=int32,<bool,float32>>')
    self.assertIsInstance(t, computation_types.StructType)
    self.assertNotIsInstance(t, computation_types.StructWithPythonType)

  def test_with_namedtuple(self):
    test_named_tuple = collections.namedtuple('TestNamedTuple', 'y x')
    t = type_conversions.infer_type(test_named_tuple(1, True))
    self.assertEqual(str(t), '<y=int32,x=bool>')
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, test_named_tuple)

  def test_with_dict(self):
    v1 = {
        'a': 1,
        'b': 2.0,
    }
    inferred_type = type_conversions.infer_type(v1)
    self.assertEqual(str(inferred_type), '<a=int32,b=float32>')
    self.assertIsInstance(inferred_type, computation_types.StructWithPythonType)
    self.assertIs(inferred_type.python_container, dict)

  def test_with_ordered_dict(self):
    t = type_conversions.infer_type(
        collections.OrderedDict([('b', 2.0), ('a', 1)])
    )
    self.assertEqual(str(t), '<b=float32,a=int32>')
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, collections.OrderedDict)

  def test_with_nested_attrs_class(self):

    @attrs.define
    class TestAttrClass:
      a: int
      b: Mapping[str, object]

    t = type_conversions.infer_type(
        TestAttrClass(a=0, b=collections.OrderedDict(x=True, y=0.0))
    )
    self.assertEqual(str(t), '<a=int32,b=<x=bool,y=float32>>')
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, TestAttrClass)
    self.assertIs(t.b.python_container, collections.OrderedDict)

  def test_with_empty_tuple(self):
    t = type_conversions.infer_type(())
    self.assertEqual(t, computation_types.StructWithPythonType([], tuple))


class ToStructureWithTypeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('value', 1, computation_types.TensorType(np.int32), 1),
      (
          'list_to_list',
          [1, 2, 3],
          computation_types.StructType([np.int32] * 3),
          [1, 2, 3],
      ),
      (
          'list_to_dict',
          [1, 2, 3],
          computation_types.StructType([
              ('a', np.int32),
              ('b', np.int32),
              ('c', np.int32),
          ]),
          {'a': 1, 'b': 2, 'c': 3},
      ),
      (
          'list_to_named_tuple',
          [1, 2, 3],
          computation_types.StructWithPythonType(
              [
                  ('a', np.int32),
                  ('b', np.int32),
                  ('c', np.int32),
              ],
              container_type=_TestNamedTuple,
          ),
          _TestNamedTuple(1, 2, 3),
      ),
      (
          'dict_to_list',
          {'a': 1, 'b': 2, 'c': 3},
          computation_types.StructType([np.int32] * 3),
          [1, 2, 3],
      ),
      (
          'dict_to_dict',
          {'a': 1, 'b': 2, 'c': 3},
          computation_types.StructType([
              ('a', np.int32),
              ('b', np.int32),
              ('c', np.int32),
          ]),
          {'a': 1, 'b': 2, 'c': 3},
      ),
      (
          'dict_to_named_tuple',
          {'a': 1, 'b': 2, 'c': 3},
          computation_types.StructWithPythonType(
              [
                  ('a', np.int32),
                  ('b', np.int32),
                  ('c', np.int32),
              ],
              container_type=_TestNamedTuple,
          ),
          _TestNamedTuple(1, 2, 3),
      ),
      (
          'named_tuple_to_list',
          _TestNamedTuple(1, 2, 3),
          computation_types.StructType([np.int32] * 3),
          [1, 2, 3],
      ),
      (
          'named_tuple_to_dict',
          _TestNamedTuple(1, 2, 3),
          computation_types.StructType([
              ('a', np.int32),
              ('b', np.int32),
              ('c', np.int32),
          ]),
          {'a': 1, 'b': 2, 'c': 3},
      ),
      (
          'named_tuple_to_named_tuple',
          _TestNamedTuple(1, 2, 3),
          computation_types.StructWithPythonType(
              [
                  ('a', np.int32),
                  ('b', np.int32),
                  ('c', np.int32),
              ],
              container_type=_TestNamedTuple,
          ),
          _TestNamedTuple(1, 2, 3),
      ),
      (
          'federated_value',
          1,
          computation_types.FederatedType(np.int32, placements.CLIENTS),
          1,
      ),
      (
          'federated_value_in_structure',
          [1, 2, 3],
          computation_types.StructType([
              computation_types.FederatedType(np.int32, placements.CLIENTS),
              computation_types.FederatedType(np.int32, placements.CLIENTS),
              computation_types.FederatedType(np.int32, placements.CLIENTS),
          ]),
          [1, 2, 3],
      ),
      (
          'federated_structure',
          [1, 2, 3],
          computation_types.FederatedType([np.int32] * 3, placements.CLIENTS),
          [1, 2, 3],
      ),
      (
          'federated_structure_nested',
          [[1, 2], [3]],
          computation_types.FederatedType(
              [[np.int32] * 2, [np.int32]], placements.CLIENTS
          ),
          [[1, 2], [3]],
      ),
  )
  def test_returns_result(self, obj, type_spec, expected):
    actual = type_conversions.to_structure_with_type(obj, type_spec)
    self.assertEqual(actual, expected)

  @parameterized.named_parameters(
      (
          'wrong_type_spec',
          [1, 2, 3],
          computation_types.TensorType(np.int32),
      ),
      (
          'wrong_type_spec_nested',
          [[1, 2], [3]],
          computation_types.StructType([np.int32] * 3),
      ),
      (
          'partially_named',
          [1, 2, 3],
          computation_types.StructType([
              ('a', np.int32),
              ('b', np.int32),
              (None, np.int32),
          ]),
      ),
  )
  def test_raises_value_error(self, obj, type_spec):
    with self.assertRaises(ValueError):
      type_conversions.to_structure_with_type(obj, type_spec)


class TypeToPyContainerTest(absltest.TestCase):

  def test_tuple_passthrough(self):
    value = (1, 2.0)
    result = type_conversions.type_to_py_container(
        (1, 2.0),
        computation_types.StructWithPythonType(
            [np.int32, np.float32], container_type=list
        ),
    )
    self.assertEqual(result, value)

  def test_represents_unnamed_fields_as_tuple(self):
    input_value = structure.Struct([(None, 1), (None, 2.0)])
    input_type = computation_types.StructType([np.int32, np.float32])
    self.assertEqual(
        type_conversions.type_to_py_container(input_value, input_type), (1, 2.0)
    )

  def test_represents_named_fields_as_odict(self):
    input_value = structure.Struct([('a', 1), ('b', 2.0)])
    input_type = computation_types.StructType(
        [('a', np.int32), ('b', np.float32)]
    )
    self.assertEqual(
        type_conversions.type_to_py_container(input_value, input_type),
        collections.OrderedDict(a=1, b=2.0),
    )

  def test_raises_on_mixed_named_unnamed(self):
    input_value = structure.Struct([('a', 1), (None, 2.0)])
    input_type = computation_types.StructType(
        [('a', np.int32), (None, np.float32)]
    )
    with self.assertRaises(ValueError):
      type_conversions.type_to_py_container(input_value, input_type)

  def test_anon_tuple_without_names_to_container_without_names(self):
    anon_tuple = structure.Struct([(None, 1), (None, 2.0)])
    types = [np.int32, np.float32]
    self.assertSequenceEqual(
        type_conversions.type_to_py_container(
            anon_tuple, computation_types.StructWithPythonType(types, list)
        ),
        [1, 2.0],
    )
    self.assertSequenceEqual(
        type_conversions.type_to_py_container(
            anon_tuple, computation_types.StructWithPythonType(types, tuple)
        ),
        (1, 2.0),
    )

  def test_succeeds_with_federated_namedtupletype(self):
    anon_tuple = structure.Struct([(None, 1), (None, 2.0)])
    types = [np.int32, np.float32]
    self.assertSequenceEqual(
        type_conversions.type_to_py_container(
            anon_tuple,
            computation_types.FederatedType(
                computation_types.StructWithPythonType(types, list),
                placements.SERVER,
            ),
        ),
        [1, 2.0],
    )
    self.assertSequenceEqual(
        type_conversions.type_to_py_container(
            anon_tuple,
            computation_types.FederatedType(
                computation_types.StructWithPythonType(types, tuple),
                placements.SERVER,
            ),
        ),
        (1, 2.0),
    )

  def test_client_placed_tuple(self):
    value = [
        structure.Struct([(None, 1), (None, 2)]),
        structure.Struct([(None, 3), (None, 4)]),
    ]
    type_spec = computation_types.FederatedType(
        computation_types.StructWithPythonType(
            [(None, np.int32), (None, np.int32)], tuple
        ),
        placements.CLIENTS,
    )
    self.assertEqual(
        [(1, 2), (3, 4)],
        type_conversions.type_to_py_container(value, type_spec),
    )

  def test_anon_tuple_with_names_to_container_without_names_fails(self):
    anon_tuple = structure.Struct([(None, 1), ('a', 2.0)])
    types = [np.int32, np.float32]
    with self.assertRaisesRegex(
        ValueError, 'Cannot convert value with field name'
    ):
      type_conversions.type_to_py_container(
          anon_tuple, computation_types.StructWithPythonType(types, tuple)
      )
    anon_tuple = structure.Struct([('a', 1), ('b', 2.0)])
    with self.assertRaisesRegex(
        ValueError, 'Cannot convert value with field name'
    ):
      type_conversions.type_to_py_container(
          anon_tuple, computation_types.StructWithPythonType(types, list)
      )

  def test_anon_tuple_with_names_to_container_with_names(self):
    anon_tuple = structure.Struct([('a', 1), ('b', 2.0)])
    types = [('a', np.int32), ('b', np.float32)]
    self.assertDictEqual(
        type_conversions.type_to_py_container(
            anon_tuple, computation_types.StructWithPythonType(types, dict)
        ),
        {'a': 1, 'b': 2.0},
    )
    self.assertSequenceEqual(
        type_conversions.type_to_py_container(
            anon_tuple,
            computation_types.StructWithPythonType(
                types, collections.OrderedDict
            ),
        ),
        collections.OrderedDict([('a', 1), ('b', 2.0)]),
    )
    test_named_tuple = collections.namedtuple('TestNamedTuple', ['a', 'b'])
    self.assertSequenceEqual(
        type_conversions.type_to_py_container(
            anon_tuple,
            computation_types.StructWithPythonType(types, test_named_tuple),
        ),
        test_named_tuple(a=1, b=2.0),
    )

    @attrs.define
    class TestFoo:
      a: int
      b: float

    self.assertEqual(
        type_conversions.type_to_py_container(
            anon_tuple, computation_types.StructWithPythonType(types, TestFoo)
        ),
        TestFoo(a=1, b=2.0),
    )

  def test_anon_tuple_without_names_promoted_to_container_with_names(self):
    anon_tuple = structure.Struct([(None, 1), (None, 2.0)])
    types = [('a', np.int32), ('b', np.float32)]
    dict_converted_value = type_conversions.type_to_py_container(
        anon_tuple, computation_types.StructWithPythonType(types, dict)
    )
    odict_converted_value = type_conversions.type_to_py_container(
        anon_tuple,
        computation_types.StructWithPythonType(types, collections.OrderedDict),
    )

    test_named_tuple = collections.namedtuple('TestNamedTuple', ['a', 'b'])
    named_tuple_converted_value = type_conversions.type_to_py_container(
        anon_tuple,
        computation_types.StructWithPythonType(types, test_named_tuple),
    )

    @attrs.define
    class TestFoo:
      a: object
      b: object

    attr_converted_value = type_conversions.type_to_py_container(
        anon_tuple, computation_types.StructWithPythonType(types, TestFoo)
    )

    self.assertIsInstance(dict_converted_value, dict)
    self.assertIsInstance(odict_converted_value, collections.OrderedDict)
    self.assertIsInstance(named_tuple_converted_value, test_named_tuple)
    self.assertIsInstance(attr_converted_value, TestFoo)

  def test_nested_py_containers(self):
    anon_tuple = structure.Struct([
        (None, 1),
        (None, 2.0),
        (
            None,
            structure.Struct(
                [('a', 3), ('b', structure.Struct([(None, 4), (None, 5)]))]
            ),
        ),
    ])

    dict_subtype = computation_types.StructWithPythonType(
        [
            ('a', np.int32),
            (
                'b',
                computation_types.StructWithPythonType(
                    [np.int32, np.int32], tuple
                ),
            ),
        ],
        dict,
    )
    type_spec = computation_types.StructType(
        [(None, np.int32), (None, np.float32), (None, dict_subtype)]
    )

    expected_nested_structure = (1, 2.0, collections.OrderedDict(a=3, b=(4, 5)))
    self.assertEqual(
        type_conversions.type_to_py_container(anon_tuple, type_spec),
        expected_nested_structure,
    )


class TypeToNonAllEqualTest(absltest.TestCase):

  def test_with_bool(self):
    for x in [True, False]:
      self.assertEqual(
          str(
              type_conversions.type_to_non_all_equal(
                  computation_types.FederatedType(
                      np.int32, placements.CLIENTS, all_equal=x
                  )
              )
          ),
          '{int32}@CLIENTS',
      )


if __name__ == '__main__':
  absltest.main()
