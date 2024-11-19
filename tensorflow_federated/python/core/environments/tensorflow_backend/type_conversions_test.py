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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import federated_language
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.environments.tensorflow_backend import type_conversions as tensorflow_type_conversions


class _TestTypedObject(federated_language.TypedObject):

  def __init__(self, type_signature: federated_language.Type):
    self._type_signature = type_signature

  @property
  def type_signature(self) -> federated_language.Type:
    return self._type_signature


class TensorflowInferTypeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'tensor',
          tf.ones(shape=[2, 3], dtype=tf.int32),
          federated_language.TensorType(np.int32, shape=[2, 3]),
      ),
      (
          'tensor_nested',
          [tf.ones(shape=[2, 3], dtype=tf.int32)],
          federated_language.StructWithPythonType(
              [
                  federated_language.TensorType(np.int32, shape=[2, 3]),
              ],
              list,
          ),
      ),
      (
          'tensor_mixed',
          [tf.ones(shape=[2, 3], dtype=tf.int32), 1.0],
          federated_language.StructWithPythonType(
              [
                  federated_language.TensorType(np.int32, shape=[2, 3]),
                  federated_language.TensorType(np.float32),
              ],
              list,
          ),
      ),
      (
          'variable',
          tf.Variable(tf.ones(shape=[2, 3], dtype=tf.int32)),
          federated_language.TensorType(np.int32, shape=[2, 3]),
      ),
      (
          'variable_nested',
          [tf.Variable(tf.ones(shape=[2, 3], dtype=tf.int32))],
          federated_language.StructWithPythonType(
              [
                  federated_language.TensorType(np.int32, shape=[2, 3]),
              ],
              list,
          ),
      ),
      (
          'variable_mixed',
          [tf.Variable(tf.ones(shape=[2, 3], dtype=tf.int32)), 1.0],
          federated_language.StructWithPythonType(
              [
                  federated_language.TensorType(np.int32, shape=[2, 3]),
                  federated_language.TensorType(np.float32),
              ],
              list,
          ),
      ),
      (
          'dataset',
          tf.data.Dataset.from_tensors(tf.ones(shape=[2, 3], dtype=tf.int32)),
          federated_language.SequenceType(
              federated_language.TensorType(np.int32, shape=[2, 3])
          ),
      ),
      (
          'dataset_nested',
          [tf.data.Dataset.from_tensors(tf.ones(shape=[2, 3], dtype=tf.int32))],
          federated_language.StructWithPythonType(
              [
                  federated_language.SequenceType(
                      federated_language.TensorType(np.int32, shape=[2, 3])
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
          federated_language.StructWithPythonType(
              [
                  federated_language.SequenceType(
                      federated_language.TensorType(np.int32, shape=[2, 3])
                  ),
                  federated_language.TensorType(np.float32),
              ],
              list,
          ),
      ),
  )
  def test_returns_result_with_tensorflow_obj(self, obj, expected_result):
    actual_result = tensorflow_type_conversions.tensorflow_infer_type(obj)
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      ('none', None),
      (
          'typed_object',
          _TestTypedObject(
              federated_language.TensorType(np.int32, shape=[2, 3])
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
        federated_language.framework, 'infer_type', autospec=True, spec_set=True
    ) as mock_infer_type:
      tensorflow_type_conversions.tensorflow_infer_type(obj)
      mock_infer_type.assert_called_once_with(obj)


class TypeToTfDtypesAndShapesTest(absltest.TestCase):

  def test_with_int_scalar(self):
    type_signature = federated_language.TensorType(np.int32)
    dtypes, shapes = tensorflow_type_conversions._type_to_tf_dtypes_and_shapes(
        type_signature
    )
    self.assertEqual(dtypes, np.int32)
    self.assertEqual(shapes, ())

  def test_with_int_vector(self):
    type_signature = federated_language.TensorType(np.int32, [10])
    dtypes, shapes = tensorflow_type_conversions._type_to_tf_dtypes_and_shapes(
        type_signature
    )
    self.assertEqual(dtypes, np.int32)
    self.assertEqual(shapes, (10,))

  def test_with_tensor_triple(self):
    type_signature = federated_language.StructWithPythonType(
        [
            ('a', federated_language.TensorType(np.int32, [5])),
            ('b', federated_language.TensorType(np.bool_)),
            ('c', federated_language.TensorType(np.float32, [3])),
        ],
        collections.OrderedDict,
    )
    dtypes, shapes = tensorflow_type_conversions._type_to_tf_dtypes_and_shapes(
        type_signature
    )
    self.assertEqual(dtypes, {'a': np.int32, 'b': np.bool_, 'c': np.float32})
    self.assertEqual(shapes, {'a': [5], 'b': [], 'c': [3]})

  def test_with_two_level_tuple(self):
    type_signature = federated_language.StructWithPythonType(
        [
            ('a', np.bool_),
            (
                'b',
                federated_language.StructWithPythonType(
                    [
                        ('c', federated_language.TensorType(np.float32)),
                        ('d', federated_language.TensorType(np.int32, [20])),
                    ],
                    collections.OrderedDict,
                ),
            ),
            ('e', federated_language.StructType([])),
        ],
        collections.OrderedDict,
    )
    dtypes, shapes = tensorflow_type_conversions._type_to_tf_dtypes_and_shapes(
        type_signature
    )
    self.assertEqual(
        dtypes, {'a': np.bool_, 'b': {'c': np.float32, 'd': np.int32}, 'e': ()}
    )
    self.assertEqual(shapes, {'a': [], 'b': {'c': [], 'd': [20]}, 'e': ()})


class TypeToTfTensorSpecsTest(absltest.TestCase):

  def test_with_int_scalar(self):
    type_signature = federated_language.TensorType(np.int32)
    tensor_specs = tensorflow_type_conversions.type_to_tf_tensor_specs(
        type_signature
    )
    self.assertEqual(tensor_specs, tf.TensorSpec([], np.int32))

  def test_with_int_vector(self):
    type_signature = federated_language.TensorType(np.int32, [10])
    tensor_specs = tensorflow_type_conversions.type_to_tf_tensor_specs(
        type_signature
    )
    self.assertEqual(tensor_specs, tf.TensorSpec([10], np.int32))

  def test_with_tensor_triple(self):
    type_signature = federated_language.StructWithPythonType(
        [
            ('a', federated_language.TensorType(np.int32, [5])),
            ('b', federated_language.TensorType(np.bool_)),
            ('c', federated_language.TensorType(np.float32, [3])),
        ],
        collections.OrderedDict,
    )
    tensor_specs = tensorflow_type_conversions.type_to_tf_tensor_specs(
        type_signature
    )
    self.assertEqual(
        tensor_specs,
        {
            'a': tf.TensorSpec([5], np.int32),
            'b': tf.TensorSpec([], np.bool_),
            'c': tf.TensorSpec([3], np.float32),
        },
    )

  def test_with_two_level_tuple(self):
    type_signature = federated_language.StructWithPythonType(
        [
            ('a', np.bool_),
            (
                'b',
                federated_language.StructWithPythonType(
                    [
                        ('c', federated_language.TensorType(np.float32)),
                        ('d', federated_language.TensorType(np.int32, [20])),
                    ],
                    collections.OrderedDict,
                ),
            ),
            ('e', federated_language.StructType([])),
        ],
        collections.OrderedDict,
    )
    tensor_specs = tensorflow_type_conversions.type_to_tf_tensor_specs(
        type_signature
    )
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
      tensorflow_type_conversions.type_to_tf_tensor_specs(np.float32(0.0))

  def test_with_unnamed_element(self):
    type_signature = federated_language.StructType([np.int32])
    tensor_specs = tensorflow_type_conversions.type_to_tf_tensor_specs(
        type_signature
    )
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
    type_spec = federated_language.StructWithPythonType(
        [
            ('a', federated_language.TensorType(np.bool_)),
            (
                'b',
                federated_language.StructWithPythonType(
                    [
                        ('c', federated_language.TensorType(np.float32)),
                        ('d', federated_language.TensorType(np.int32, (20,))),
                    ],
                    collections.OrderedDict,
                ),
            ),
        ],
        collections.OrderedDict,
    )
    tf_structure = tensorflow_type_conversions.type_to_tf_structure(type_spec)
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
    type_spec = federated_language.StructType([np.bool_, np.int32])
    tf_structure = tensorflow_type_conversions.type_to_tf_structure(type_spec)
    with tf.Graph().as_default():
      ds = tf.data.experimental.from_variant(
          tf.compat.v1.placeholder(tf.variant, shape=[]), structure=tf_structure
      )
      actual_structure = ds.element_spec
      self.assertEqual(expected_structure, actual_structure)

  def test_with_none(self):
    with self.assertRaises(TypeError):
      tensorflow_type_conversions.type_to_tf_structure(None)

  def test_with_sequence_type(self):
    with self.assertRaises(ValueError):
      tensorflow_type_conversions.type_to_tf_structure(
          federated_language.SequenceType(np.int32)
      )

  def test_with_inconsistently_named_elements(self):
    with self.assertRaises(ValueError):
      tensorflow_type_conversions.type_to_tf_structure(
          federated_language.StructType([('a', np.int32), np.bool_])
      )

  def test_with_no_elements(self):
    tf_structure = tensorflow_type_conversions.type_to_tf_structure(
        federated_language.StructType([])
    )
    self.assertEqual(tf_structure, ())


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
      federated_language.framework.assert_types_identical(
          tensor_type, federated_language.TensorType(np.int32)
      )
      return 5

    result = tensorflow_type_conversions.structure_from_tensor_type_tree(
        expect_tfint32_return_5, np.int32
    )
    self.assertEqual(result, 5)

  def test_dict(self):
    struct_type = federated_language.StructWithPythonType(
        [('a', np.int32), ('b', np.int32)], collections.OrderedDict
    )
    return_incr = self.get_incrementing_function()
    result = tensorflow_type_conversions.structure_from_tensor_type_tree(
        return_incr, struct_type
    )
    self.assertEqual(result, collections.OrderedDict(a=0, b=1))

  def test_nested_python_type(self):
    return_incr = self.get_incrementing_function()
    result = tensorflow_type_conversions.structure_from_tensor_type_tree(
        return_incr, [np.int32, (np.str_, np.int32)]
    )
    self.assertEqual(result, [0, (1, 2)])

  def test_weird_result_elements(self):
    result = tensorflow_type_conversions.structure_from_tensor_type_tree(
        lambda _: set(), [np.int32, (np.str_, np.int32)]
    )
    self.assertEqual(result, [set(), (set(), set())])


if __name__ == '__main__':
  absltest.main()
