# Copyright 2020, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tensorflow_federated.python.tensorflow_libs.variable_utils."""

import functools
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
          initializer=tf.compat.v1.initializers.constant,
      )
      # Explicitly add a variable that is not added to any collections.
      v4 = tf.compat.v1.get_variable(
          name='v1_var_no_collections',
          shape=(),
          initializer=tf.compat.v1.initializers.constant,
          collections=[],
      )
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
        initial_value=lambda: python_value, dtype=dtype
    )
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

  @parameterized.product(
      value=[
          1,
          [1, 2, 3],
          1.0,
          [1.0, 2.0, 3.0],
      ],
      operator_fn=[
          operator.__add__,
          operator.__eq__,
          operator.__floordiv__,
          operator.__ge__,
          operator.__gt__,
          operator.__le__,
          operator.__lt__,
          operator.__mul__,
          operator.__ne__,
          operator.__truediv__,
          operator.add,
          operator.eq,
          operator.floordiv,
          operator.ge,
          operator.gt,
          operator.le,
          operator.lt,
          operator.mul,
          operator.ne,
          operator.truediv,
      ],
  )
  def test_binary_operators(self, value, operator_fn):
    value = tf.convert_to_tensor(value)
    variable = tf.Variable(tf.zeros_like(value))
    tensor_variable = variable_utils.TensorVariable(tf.zeros_like(value))
    self.assertAllEqual(variable, tensor_variable)
    variable_result = operator_fn(variable, value)
    tensor_variable_result = operator_fn(tensor_variable, value)
    self.assertAllEqual(variable_result, tensor_variable_result)

  @parameterized.product(
      value=[1, [1, 2, 3], True, [True, False, True]],
      operator_fn=[
          operator.__and__,
          operator.and_,
          operator.__and__,
          operator.and_,
      ],
  )
  def test_non_float_binary_operators(self, value, operator_fn):
    value = tf.convert_to_tensor(value)
    variable = tf.Variable(tf.zeros_like(value))
    tensor_variable = variable_utils.TensorVariable(tf.zeros_like(value))
    self.assertAllEqual(variable, tensor_variable)
    variable_result = operator_fn(variable, value)
    tensor_variable_result = operator_fn(tensor_variable, value)
    self.assertAllEqual(variable_result, tensor_variable_result)

  @parameterized.product(
      value=[-1, [1, -2, 3]],
      operator_fn=[
          operator.__neg__,
          operator.neg,
          operator.__invert__,
          operator.invert,
          operator.__abs__,
          operator.abs,
      ],
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
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        'Attempting to slice scalar input',
    ):
      _ = tensor_variable[0]

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
        v = variable_utils.TensorVariable(1.0)
        hash(v)
    with self.subTest('graph'):
      with tf.Graph().as_default():
        v = variable_utils.TensorVariable(1.0)
        try:
          hash(v)
        except TypeError:
          self.fail('Failed to compute hash of variable in a graph context.')

  def test_name(self):
    variable_name = 'test'
    v = variable_utils.TensorVariable(1.0, name=variable_name)
    self.assertEqual(v.name, variable_name)
    with tf.Graph().as_default():
      graph_name = variable_name + '_graph'
      graph_v = variable_utils.TensorVariable(1.0, name=graph_name)
      self.assertEqual(graph_v.name, graph_name)

  def test_partitioned_variable(self):
    test_value = [1, 2, 3]
    with tf.Graph().as_default() as g:
      create_variable = functools.partial(
          tf.compat.v1.get_variable,
          initializer=tf.convert_to_tensor(test_value),
          shape=[3],
          partitioner=tf.compat.v1.min_max_variable_partitioner(
              max_partitions=3, min_slice_size=1
          ),
      )

      v = create_variable(name='partitioned_variable')
      output_v = tf.identity(v)
      with tf.variable_creator_scope(variable_utils.create_tensor_variable):
        tv = create_variable(name='partitioned_tensor_variable')
      output_tv = tf.identity(tv)

    with tf.compat.v1.Session(graph=g) as sess:
      sess.run(fetches=tf.compat.v1.initializers.global_variables())
      output_v, output_tv = sess.run(fetches=[output_v, output_tv])
      self.assertAllEqual(output_v, output_tv)


if __name__ == '__main__':
  tf.test.main()
