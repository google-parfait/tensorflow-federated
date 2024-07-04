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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import attrs
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_conversions
from tensorflow_federated.python.core.impl.types import type_test_utils
from tensorflow_federated.python.core.impl.types import typed_object


class _TestTypedObject(typed_object.TypedObject):

  def __init__(self, type_signature: computation_types.Type):
    self._type_signature = type_signature

  @property
  def type_signature(self) -> computation_types.Type:
    return self._type_signature


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


class TensorflowInferTypeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'tensor',
          tf.ones(shape=[2, 3], dtype=tf.int32),
          computation_types.TensorType(np.int32, shape=[2, 3]),
      ),
      (
          'tensor_nested',
          [tf.ones(shape=[2, 3], dtype=tf.int32)],
          computation_types.StructWithPythonType(
              [
                  computation_types.TensorType(np.int32, shape=[2, 3]),
              ],
              list,
          ),
      ),
      (
          'tensor_mixed',
          [tf.ones(shape=[2, 3], dtype=tf.int32), 1.0],
          computation_types.StructWithPythonType(
              [
                  computation_types.TensorType(np.int32, shape=[2, 3]),
                  computation_types.TensorType(np.float32),
              ],
              list,
          ),
      ),
      (
          'variable',
          tf.Variable(tf.ones(shape=[2, 3], dtype=tf.int32)),
          computation_types.TensorType(np.int32, shape=[2, 3]),
      ),
      (
          'variable_nested',
          [tf.Variable(tf.ones(shape=[2, 3], dtype=tf.int32))],
          computation_types.StructWithPythonType(
              [
                  computation_types.TensorType(np.int32, shape=[2, 3]),
              ],
              list,
          ),
      ),
      (
          'variable_mixed',
          [tf.Variable(tf.ones(shape=[2, 3], dtype=tf.int32)), 1.0],
          computation_types.StructWithPythonType(
              [
                  computation_types.TensorType(np.int32, shape=[2, 3]),
                  computation_types.TensorType(np.float32),
              ],
              list,
          ),
      ),
      (
          'dataset',
          tf.data.Dataset.from_tensors(tf.ones(shape=[2, 3], dtype=tf.int32)),
          computation_types.SequenceType(
              computation_types.TensorType(np.int32, shape=[2, 3])
          ),
      ),
      (
          'dataset_nested',
          [tf.data.Dataset.from_tensors(tf.ones(shape=[2, 3], dtype=tf.int32))],
          computation_types.StructWithPythonType(
              [
                  computation_types.SequenceType(
                      computation_types.TensorType(np.int32, shape=[2, 3])
                  ),
              ],
              list,
          ),
      ),
      (
          'dataset_mixed',
          [
              tf.data.Dataset.from_tensors(
                  tf.ones(shape=[2, 3], dtype=tf.int32)
              ),
              1.0,
          ],
          computation_types.StructWithPythonType(
              [
                  computation_types.SequenceType(
                      computation_types.TensorType(np.int32, shape=[2, 3])
                  ),
                  computation_types.TensorType(np.float32),
              ],
              list,
          ),
      ),
  )
  def test_returns_result_with_tensorflow_obj(self, obj, expected_result):
    actual_result = type_conversions.tensorflow_infer_type(obj)
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      ('none', None),
      (
          'typed_object',
          _TestTypedObject(
              computation_types.TensorType(np.int32, shape=[2, 3])
          ),
      ),
      ('int', 1),
      ('numpy', np.ones(shape=[2, 3], dtype=np.int32)),
      ('sequence_unnamed', [True, 1, 'a']),
      ('sequence_named', [('a', True), ('b', 1), ('c', 'a')]),
      ('mapping', {'a': True, 'b': 1, 'c': 'a'}),
  )
  def test_delegates_result_with_obj(self, obj):

    with mock.patch.object(
        type_conversions, 'infer_type', autospec=True, spec_set=True
    ) as mock_infer_type:
      type_conversions.tensorflow_infer_type(obj)
      mock_infer_type.assert_called_once_with(obj)


class TypeToTfDtypesAndShapesTest(absltest.TestCase):

  def test_with_int_scalar(self):
    type_signature = computation_types.TensorType(np.int32)
    dtypes, shapes = type_conversions._type_to_tf_dtypes_and_shapes(
        type_signature
    )
    self.assertEqual(dtypes, np.int32)
    self.assertEqual(shapes, ())

  def test_with_int_vector(self):
    type_signature = computation_types.TensorType(np.int32, [10])
    dtypes, shapes = type_conversions._type_to_tf_dtypes_and_shapes(
        type_signature
    )
    self.assertEqual(dtypes, np.int32)
    self.assertEqual(shapes, (10,))

  def test_with_tensor_triple(self):
    type_signature = computation_types.StructWithPythonType(
        [
            ('a', computation_types.TensorType(np.int32, [5])),
            ('b', computation_types.TensorType(np.bool_)),
            ('c', computation_types.TensorType(np.float32, [3])),
        ],
        collections.OrderedDict,
    )
    dtypes, shapes = type_conversions._type_to_tf_dtypes_and_shapes(
        type_signature
    )
    self.assertEqual(dtypes, {'a': np.int32, 'b': np.bool_, 'c': np.float32})
    self.assertEqual(shapes, {'a': [5], 'b': [], 'c': [3]})

  def test_with_two_level_tuple(self):
    type_signature = computation_types.StructWithPythonType(
        [
            ('a', np.bool_),
            (
                'b',
                computation_types.StructWithPythonType(
                    [
                        ('c', computation_types.TensorType(np.float32)),
                        ('d', computation_types.TensorType(np.int32, [20])),
                    ],
                    collections.OrderedDict,
                ),
            ),
            ('e', computation_types.StructType([])),
        ],
        collections.OrderedDict,
    )
    dtypes, shapes = type_conversions._type_to_tf_dtypes_and_shapes(
        type_signature
    )
    self.assertEqual(
        dtypes, {'a': np.bool_, 'b': {'c': np.float32, 'd': np.int32}, 'e': ()}
    )
    self.assertEqual(shapes, {'a': [], 'b': {'c': [], 'd': [20]}, 'e': ()})


class TypeToTfTensorSpecsTest(absltest.TestCase):

  def test_with_int_scalar(self):
    type_signature = computation_types.TensorType(np.int32)
    tensor_specs = type_conversions.type_to_tf_tensor_specs(type_signature)
    self.assertEqual(tensor_specs, tf.TensorSpec([], np.int32))

  def test_with_int_vector(self):
    type_signature = computation_types.TensorType(np.int32, [10])
    tensor_specs = type_conversions.type_to_tf_tensor_specs(type_signature)
    self.assertEqual(tensor_specs, tf.TensorSpec([10], np.int32))

  def test_with_tensor_triple(self):
    type_signature = computation_types.StructWithPythonType(
        [
            ('a', computation_types.TensorType(np.int32, [5])),
            ('b', computation_types.TensorType(np.bool_)),
            ('c', computation_types.TensorType(np.float32, [3])),
        ],
        collections.OrderedDict,
    )
    tensor_specs = type_conversions.type_to_tf_tensor_specs(type_signature)
    self.assertEqual(
        tensor_specs,
        {
            'a': tf.TensorSpec([5], np.int32),
            'b': tf.TensorSpec([], np.bool_),
            'c': tf.TensorSpec([3], np.float32),
        },
    )

  def test_with_two_level_tuple(self):
    type_signature = computation_types.StructWithPythonType(
        [
            ('a', np.bool_),
            (
                'b',
                computation_types.StructWithPythonType(
                    [
                        ('c', computation_types.TensorType(np.float32)),
                        ('d', computation_types.TensorType(np.int32, [20])),
                    ],
                    collections.OrderedDict,
                ),
            ),
            ('e', computation_types.StructType([])),
        ],
        collections.OrderedDict,
    )
    tensor_specs = type_conversions.type_to_tf_tensor_specs(type_signature)
    self.assertEqual(
        tensor_specs,
        {
            'a': tf.TensorSpec([], np.bool_),
            'b': {
                'c': tf.TensorSpec([], np.float32),
                'd': tf.TensorSpec([20], np.int32),
            },
            'e': (),
        },
    )

  def test_with_invalid_type(self):
    with self.assertRaises(TypeError):
      type_conversions.type_to_tf_tensor_specs(np.float32(0.0))

  def test_with_unnamed_element(self):
    type_signature = computation_types.StructType([np.int32])
    tensor_specs = type_conversions.type_to_tf_tensor_specs(type_signature)
    self.assertEqual(tensor_specs, (tf.TensorSpec([], np.int32),))


class TypeToTfStructureTest(absltest.TestCase):

  def test_with_names(self):
    expected_structure = collections.OrderedDict([
        ('a', tf.TensorSpec(shape=(), dtype=np.bool_)),
        (
            'b',
            collections.OrderedDict([
                ('c', tf.TensorSpec(shape=(), dtype=np.float32)),
                ('d', tf.TensorSpec(shape=(20,), dtype=np.int32)),
            ]),
        ),
    ])
    type_spec = computation_types.StructWithPythonType(
        [
            ('a', computation_types.TensorType(np.bool_)),
            (
                'b',
                computation_types.StructWithPythonType(
                    [
                        ('c', computation_types.TensorType(np.float32)),
                        ('d', computation_types.TensorType(np.int32, (20,))),
                    ],
                    collections.OrderedDict,
                ),
            ),
        ],
        collections.OrderedDict,
    )
    tf_structure = type_conversions.type_to_tf_structure(type_spec)
    with tf.Graph().as_default():
      ds = tf.data.experimental.from_variant(
          tf.compat.v1.placeholder(tf.variant, shape=[]), structure=tf_structure
      )
      actual_structure = ds.element_spec
      self.assertEqual(expected_structure, actual_structure)

  def test_without_names(self):
    expected_structure = (
        tf.TensorSpec(shape=(), dtype=np.bool_),
        tf.TensorSpec(shape=(), dtype=np.int32),
    )
    type_spec = computation_types.StructType([np.bool_, np.int32])
    tf_structure = type_conversions.type_to_tf_structure(type_spec)
    with tf.Graph().as_default():
      ds = tf.data.experimental.from_variant(
          tf.compat.v1.placeholder(tf.variant, shape=[]), structure=tf_structure
      )
      actual_structure = ds.element_spec
      self.assertEqual(expected_structure, actual_structure)

  def test_with_none(self):
    with self.assertRaises(TypeError):
      type_conversions.type_to_tf_structure(None)

  def test_with_sequence_type(self):
    with self.assertRaises(ValueError):
      type_conversions.type_to_tf_structure(
          computation_types.SequenceType(np.int32)
      )

  def test_with_inconsistently_named_elements(self):
    with self.assertRaises(ValueError):
      type_conversions.type_to_tf_structure(
          computation_types.StructType([('a', np.int32), np.bool_])
      )

  def test_with_no_elements(self):
    tf_structure = type_conversions.type_to_tf_structure(
        computation_types.StructType([])
    )
    self.assertEqual(tf_structure, ())


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


class StructureFromTensorTypeTreeTest(absltest.TestCase):

  def get_incrementing_function(self):
    i = -1

    def fn(ignored):
      del ignored
      nonlocal i
      i += 1
      return i

    return fn

  def test_single_tensor(self):
    def expect_tfint32_return_5(tensor_type):
      type_test_utils.assert_types_identical(
          tensor_type, computation_types.TensorType(np.int32)
      )
      return 5

    result = type_conversions.structure_from_tensor_type_tree(
        expect_tfint32_return_5, np.int32
    )
    self.assertEqual(result, 5)

  def test_dict(self):
    struct_type = computation_types.StructWithPythonType(
        [('a', np.int32), ('b', np.int32)], collections.OrderedDict
    )
    return_incr = self.get_incrementing_function()
    result = type_conversions.structure_from_tensor_type_tree(
        return_incr, struct_type
    )
    self.assertEqual(result, collections.OrderedDict(a=0, b=1))

  def test_nested_python_type(self):
    return_incr = self.get_incrementing_function()
    result = type_conversions.structure_from_tensor_type_tree(
        return_incr, [np.int32, (np.str_, np.int32)]
    )
    self.assertEqual(result, [0, (1, 2)])

  def test_weird_result_elements(self):
    result = type_conversions.structure_from_tensor_type_tree(
        lambda _: set(), [np.int32, (np.str_, np.int32)]
    )
    self.assertEqual(result, [set(), (set(), set())])


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
