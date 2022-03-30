# Copyright 2020, The TensorFlow Federated Authors.
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
"""Tests for tensorflow_federated.python.tensorflow_libs.variable_utils."""

import itertools
import operator

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.tensorflow_libs import variable_utils


class RecordVariableCreationScopeTest(tf.test.TestCase):

  def test_variable_capture(self):
    with variable_utils.record_variable_creation_scope() as variable_list:
      v1 = tf.Variable(1.0)
      v2 = tf.Variable('abc', name='my_test_var')
      v3 = tf.compat.v1.get_variable(
          name='v1_var',
          shape=(),
          initializer=tf.compat.v1.initializers.constant)
      # Explicitly add a variable that is not added to any collections.
      v4 = tf.compat.v1.get_variable(
          name='v1_var_no_collections',
          shape=(),
          initializer=tf.compat.v1.initializers.constant,
          collections=[])
    self.assertEqual([v1, v2, v3, v4], variable_list)


class TensorVariableTest(tf.test.TestCase, parameterized.TestCase):

  def assertAllEqual(self, a, b, msg=None):
    self.assertAllClose(a, b, msg=msg, rtol=0.0, atol=0.0)

  @parameterized.named_parameters(
      ('float_default', 1.0, None),
      ('float32_scalar', 1.0, tf.float32),
      ('float64_scalar', 1.0, tf.float64),
      ('float32_tensor', [1.0, 2.0, 3.0], tf.float32),
      ('float64_tensor', [1.0, 2.0, 3.0], tf.float64),
      ('int_default', 1, None),
      ('int32_scalar', 1, tf.int32),
      ('int64_scalar', 1, tf.int64),
      ('int32_tensor', [1, 2, 3], tf.int32),
      ('int64_tensor', [1, 2, 3], tf.int64),
  )
  def test_creation_python_literals(self, python_value, dtype):
    v = variable_utils.TensorVariable(initial_value=python_value, dtype=dtype)
    self.assertIsInstance(v, variable_utils.TensorVariable)
    if dtype is not None:
      self.assertEqual(v.dtype, dtype)
    if isinstance(python_value, list):
      self.assertEqual(v.shape, [len(python_value)])
      self.assertEqual(v.get_shape(), [len(python_value)])
    else:
      self.assertEqual(v.shape, [])
      self.assertEqual(v.get_shape(), [])
    self.assertAllClose(v.value(), python_value)
    self.assertAllClose(v.read_value(), python_value)

  @parameterized.named_parameters(
      ('float32_scalar', 1.0, tf.float32),
      ('float64_scalar', 1.0, tf.float64),
      ('float32_tensor', [1.0, 2.0, 3.0], tf.float32),
      ('float64_tensor', [1.0, 2.0, 3.0], tf.float64),
      ('int32_scalar', 1, tf.int32),
      ('int64_scalar', 1, tf.int64),
      ('int32_tensor', [1, 2, 3], tf.int32),
      ('int64_tensor', [1, 2, 3], tf.int64),
  )
  def test_creation_python_callable(self, python_value, dtype):
    v = variable_utils.TensorVariable(
        initial_value=lambda: python_value, dtype=dtype)
    self.assertIsInstance(v, variable_utils.TensorVariable)
    if dtype is not None:
      self.assertEqual(v.dtype, dtype)
    if isinstance(python_value, list):
      self.assertEqual(v.shape, [len(python_value)])
      self.assertEqual(v.get_shape(), [len(python_value)])
    else:
      self.assertEqual(v.shape, [])
      self.assertEqual(v.get_shape(), [])
    self.assertAllClose(v.value(), python_value)
    self.assertAllClose(v.read_value(), python_value)

  def test_assign_add(self):
    self.assertTrue(tf.executing_eagerly())
    v = variable_utils.TensorVariable(initial_value=1.0)
    result = v.assign_add(1.0, read_value=True)
    self.assertAllClose(result, 2.0)
    self.assertAllClose(result, v.read_value())
    self.assertAllClose(result, v.value())
    result = v.assign_add(1.0, read_value=False)
    self.assertIsNone(result)
    self.assertAllClose(3.0, v.read_value())
    self.assertAllClose(3.0, v.value())

  def test_assign_sub(self):
    self.assertTrue(tf.executing_eagerly())
    v = variable_utils.TensorVariable(initial_value=1.0)
    result = v.assign_sub(1.0, read_value=True)
    self.assertAllClose(result, 0.0)
    self.assertAllClose(result, v.read_value())
    self.assertAllClose(result, v.value())
    result = v.assign_sub(1.0, read_value=False)
    self.assertIsNone(result)
    self.assertAllClose(-1.0, v.read_value())
    self.assertAllClose(-1.0, v.value())

  @parameterized.named_parameters(
      # pylint: disable=g-complex-comprehension,undefined-variable
      ('_'.join([value[0], operator_fn[0]]), value[1], operator_fn[1])
      for value, operator_fn in itertools.product([
          ('scalar_int', 1),
          ('tensor_int', [1, 2, 3]),
          ('scalar_float', 1.0),
          ('tensor_float', [1.0, 2.0, 3.0]),
      ], [
          ('__add__', operator.__add__),
          ('__eq__', operator.__eq__),
          ('__floordiv__', operator.__floordiv__),
          ('__ge__', operator.__ge__),
          ('__gt__', operator.__gt__),
          ('__le__', operator.__le__),
          ('__lt__', operator.__lt__),
          ('__mul__', operator.__mul__),
          ('__ne__', operator.__ne__),
          ('__truediv__', operator.__truediv__),
          ('add', operator.add),
          ('eq', operator.eq),
          ('floordiv', operator.floordiv),
          ('ge', operator.ge),
          ('gt', operator.gt),
          ('le', operator.le),
          ('lt', operator.lt),
          ('mul', operator.mul),
          ('ne', operator.ne),
          ('truediv', operator.truediv),
      ])
      # pylint: enable=g-complex-comprehension,undefined-variable
  )
  def test_binary_operators(self, value, operator_fn):
    value = tf.convert_to_tensor(value)
    variable = tf.Variable(tf.zeros_like(value))
    tensor_variable = variable_utils.TensorVariable(tf.zeros_like(value))
    self.assertAllEqual(variable, tensor_variable)
    variable_result = operator_fn(variable, value)
    tensor_variable_result = operator_fn(tensor_variable, value)
    self.assertAllEqual(variable_result, tensor_variable_result)

  @parameterized.named_parameters(
      # pylint: disable=g-complex-comprehension,undefined-variable
      ('_'.join([value[0], operator_fn[0]]), value[1], operator_fn[1])
      for value, operator_fn in itertools.product([
          ('scalar_int', 1),
          ('tensor_int', [1, 2, 3]),
          ('scalar_bool', True),
          ('tensor_bool', [True, False, True]),
      ], [
          ('__and__', operator.__and__),
          ('and', operator.and_),
          ('__or__', operator.__and__),
          ('or', operator.and_),
      ])
      # pylint: enable=g-complex-comprehension,undefined-variable
  )
  def test_non_float_binary_operators(self, value, operator_fn):
    value = tf.convert_to_tensor(value)
    variable = tf.Variable(tf.zeros_like(value))
    tensor_variable = variable_utils.TensorVariable(tf.zeros_like(value))
    self.assertAllEqual(variable, tensor_variable)
    variable_result = operator_fn(variable, value)
    tensor_variable_result = operator_fn(tensor_variable, value)
    self.assertAllEqual(variable_result, tensor_variable_result)

  @parameterized.named_parameters(
      # pylint: disable=g-complex-comprehension,undefined-variable
      ('_'.join([value[0], operator_fn[0]]), value[1], operator_fn[1])
      for value, operator_fn in itertools.product([
          ('scalar_int', -1),
          ('tensor_int', [1, -2, 3]),
      ], [
          ('__neg__', operator.__neg__),
          ('neg', operator.neg),
          ('__invert__', operator.__invert__),
          ('invert', operator.invert),
          ('__abs__', operator.__abs__),
          ('abs', operator.abs),
      ])
      # pylint: enable=g-complex-comprehension,undefined-variable
  )
  def test_unary_operators(self, value, operator_fn):
    value = tf.convert_to_tensor(value)
    variable = tf.Variable(value)
    tensor_variable = variable_utils.TensorVariable(value)
    self.assertAllEqual(variable, tensor_variable)
    variable_result = operator_fn(variable)
    tensor_variable_result = operator_fn(tensor_variable)
    self.assertAllEqual(variable_result, tensor_variable_result)

  @parameterized.named_parameters(
      ('tensor_int', [1, -2, 3]),
      ('tensor_bool', [[True, False, True]]),
  )
  def test_tensor_getitem(self, value):
    value = tf.convert_to_tensor(value)
    variable = tf.Variable(value)
    tensor_variable = variable_utils.TensorVariable(value)
    self.assertAllEqual(variable, tensor_variable)
    with self.subTest('index'):
      variable_result = variable[0]
      tensor_variable_result = tensor_variable[0]
      self.assertAllEqual(variable_result, tensor_variable_result)
    with self.subTest('slice'):
      variable_result = variable[0:]
      tensor_variable_result = tensor_variable[0:]
      self.assertAllEqual(variable_result, tensor_variable_result)

  @parameterized.named_parameters(
      ('scalar_int', -1),
      ('scalar_bool', True),
  )
  def test_scalar_getitem_fails(self, value):
    value = tf.convert_to_tensor(value)
    tensor_variable = variable_utils.TensorVariable(value)
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                'Index out of range'):
      tensor_variable[0]  # pylint: disable=pointless-statement

  def test_shape_validation_eager(self):
    v = variable_utils.TensorVariable(1.0)
    with self.assertRaises(ValueError):
      v.assign([1.0, 2.0])
    with self.assertRaises(ValueError):
      v.assign_sub([1.0, 2.0])
    with self.assertRaises(ValueError):
      v.assign_add([1.0, 2.0])
    with self.assertRaises(ValueError):
      v = variable_utils.TensorVariable(1.0, shape=[2])
    v.assign(5.0)
    v = variable_utils.TensorVariable(1.0, shape=tf.TensorShape(None))
    v.assign([5.0, 10.0])

  def test_shape_validation_graph(self):
    with tf.Graph().as_default():
      # Unknown shape is compatible with everything.
      p = tf.compat.v1.placeholder(dtype=tf.float32)
      v = variable_utils.TensorVariable(p)
      v.assign([1.0, 2.0])
      v.assign_sub([3.0])
      v.assign_add([[5.0]])
      v.assign(5.0)
      # Fixed shape is not.
      v = variable_utils.TensorVariable(1.0)
      with self.assertRaises(ValueError):
        v.assign([1.0, 2.0])
      with self.assertRaises(ValueError):
        v.assign_sub([1.0, 2.0])
      with self.assertRaises(ValueError):
        v.assign_add([1.0, 2.0])
      with self.assertRaises(ValueError):
        v = variable_utils.TensorVariable(1.0, shape=[2])
      v.assign(5.0)

  def test_hashing(self):
    with self.subTest('eager'):
      with self.assertRaises(TypeError):
        v = tf.Variable(1.0)
        hash(v)
    with self.subTest('graph'):
      with tf.Graph().as_default():
        v = tf.Variable(1.0)
        try:
          hash(v)
        except:  # pylint: disable=bare-except
          self.fail('Failed to compute hash of variable in a graph context.')


if __name__ == '__main__':
  tf.test.main()
