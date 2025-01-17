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
from typing import NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import federated_language
import ml_dtypes
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation


def one_arg_fn(x):
  return x > 10


def no_arg_fn():
  return 10


class ToNumpyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('scalar', tf.constant(1), 1),
      ('variable', tf.Variable(1, name='v1'), 1),
      ('array', tf.constant([1, 2, 3]), np.array([1, 2, 3])),
      (
          'list_tensors',
          [tf.constant(1), tf.constant(2), tf.constant(3)],
          [1, 2, 3],
      ),
      (
          'dataset',
          tf.data.Dataset.range(1, 4),
          [1, 2, 3],
      ),
  )
  def test_returns_expected_result(self, value, expected_result):
    actual_result = tensorflow_computation._to_numpy(value)

    if isinstance(actual_result, (np.ndarray, np.generic)):
      np.testing.assert_array_equal(actual_result, expected_result)
    else:
      self.assertEqual(actual_result, expected_result)


class TensorFlowComputationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('tensor_bool', federated_language.TensorType(np.bool_)),
      ('tensor_int8', federated_language.TensorType(np.int8)),
      ('tensor_int16', federated_language.TensorType(np.int16)),
      ('tensor_int32', federated_language.TensorType(np.int32)),
      ('tensor_int64', federated_language.TensorType(np.int64)),
      ('tensor_uint8', federated_language.TensorType(np.uint8)),
      ('tensor_uint16', federated_language.TensorType(np.uint16)),
      ('tensor_uint32', federated_language.TensorType(np.uint32)),
      ('tensor_uint64', federated_language.TensorType(np.uint64)),
      ('tensor_float16', federated_language.TensorType(np.float16)),
      ('tensor_float32', federated_language.TensorType(np.float32)),
      ('tensor_float64', federated_language.TensorType(np.float64)),
      ('tensor_complex64', federated_language.TensorType(np.complex64)),
      ('tensor_complex128', federated_language.TensorType(np.complex128)),
      ('tensor_bfloat16', federated_language.TensorType(ml_dtypes.bfloat16)),
      ('tensor_str', federated_language.TensorType(np.str_)),
      ('tensor_generic', federated_language.TensorType(np.int32)),
      ('tensor_array', federated_language.TensorType(np.int32, shape=[3])),
      ('sequence', federated_language.SequenceType(np.int32)),
  )
  def test_returns_concrete_computation_with_dtype(self, type_spec):
    @tensorflow_computation.tf_computation(type_spec)
    def _comp(x):
      return x

    self.assertIsInstance(
        _comp, federated_language.framework.ConcreteComputation
    )
    expected_type = federated_language.FunctionType(type_spec, type_spec)
    self.assertEqual(_comp.type_signature, expected_type)

  @parameterized.named_parameters(
      ('lambda_with_arg', lambda x: x > 10, tf.int32, '(int32 -> bool)'),
      ('function_with_arg', one_arg_fn, tf.int32, '(int32 -> bool)'),
      (
          'tf_function_with_arg',
          tf.function(one_arg_fn),
          tf.int32,
          '(int32 -> bool)',
      ),
      ('lambda_with_no_args', lambda: 10, None, '( -> int32)'),
      ('function_with_no_args', no_arg_fn, None, '( -> int32)'),
      ('tf_function_with_no_args', tf.function(no_arg_fn), None, '( -> int32)'),
  )
  def test_tf_computation_with_type(
      self, fn, fn_arg_type, expected_representation
  ):
    fn = tensorflow_computation.tf_computation(fn, fn_arg_type)
    self.assertEqual(
        fn.type_signature.compact_representation(), expected_representation
    )

  @parameterized.named_parameters(
      ('lambda', lambda x: x > 10),
      ('function', one_arg_fn),
      ('tf_function', tf.function(one_arg_fn)),
  )
  def test_tf_computation_without_type(self, fn):
    fn = tensorflow_computation.tf_computation(fn)
    concrete_fn = fn.fn_for_argument_type(
        federated_language.TensorType(np.int32)
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(), '(int32 -> bool)'
    )
    concrete_fn = fn.fn_for_argument_type(
        federated_language.TensorType(np.float32)
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(), '(float32 -> bool)'
    )

  def test_decorate_as_typed_fn(self):
    @tensorflow_computation.tf_computation(tf.int32)
    def foo(x):
      return x > 10

    self.assertEqual(
        foo.type_signature.compact_representation(), '(int32 -> bool)'
    )

  def test_decorate_as_polymorphic_fn(self):
    @tensorflow_computation.tf_computation
    def foo(x):
      return x > 10

    concrete_fn = foo.fn_for_argument_type(
        federated_language.TensorType(np.int32)
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(), '(int32 -> bool)'
    )
    concrete_fn = foo.fn_for_argument_type(
        federated_language.TensorType(np.float32)
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(), '(float32 -> bool)'
    )

  def test_decorate_as_no_arg_fn(self):
    @tensorflow_computation.tf_computation
    def foo():
      return 10

    self.assertEqual(foo.type_signature.compact_representation(), '( -> int32)')

  def test_takes_tuple_typed(self):
    @tf.function
    def foo(t):
      return t[0] + t[1]

    foo = tensorflow_computation.tf_computation(foo, (tf.int32, tf.int32))
    self.assertEqual(
        foo.type_signature.compact_representation(), '(<int32,int32> -> int32)'
    )

  def test_takes_tuple_polymorphic(self):
    def foo(t):
      return t[0] + t[1]

    foo = tensorflow_computation.tf_computation(foo)

    concrete_fn = foo.fn_for_argument_type(
        federated_language.StructType([
            federated_language.TensorType(np.int32),
            federated_language.TensorType(np.int32),
        ])
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(),
        '(<int32,int32> -> int32)',
    )
    concrete_fn = foo.fn_for_argument_type(
        federated_language.StructType([
            federated_language.TensorType(np.float32),
            federated_language.TensorType(np.float32),
        ])
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(),
        '(<float32,float32> -> float32)',
    )

  def test_takes_structured_tuple_typed(self):
    class MyType(NamedTuple):
      x: object
      y: object

    @tf.function
    def foo(x, t, l, odict, my_type):
      self.assertIsInstance(x, tf.Tensor)
      self.assertIsInstance(t, tuple)
      self.assertIsInstance(l, list)
      self.assertIsInstance(odict, collections.OrderedDict)
      self.assertIsInstance(my_type, MyType)
      return x + t[0] + l[0] + odict['foo'] + my_type.x

    foo = tensorflow_computation.tf_computation(
        foo,
        [
            tf.int32,
            (tf.int32, tf.int32),
            [tf.int32, tf.int32],
            collections.OrderedDict([('foo', tf.int32), ('bar', tf.int32)]),
            MyType(tf.int32, tf.int32),
        ],
    )
    self.assertEqual(
        foo.type_signature.compact_representation(),
        (
            '(<x=int32,t=<int32,int32>,l=<int32,int32>,odict=<foo=int32,bar=int32>,my_type=<x=int32,y=int32>>'
            ' -> int32)'
        ),
    )

  def test_takes_structured_tuple_polymorphic(self):
    class MyType(NamedTuple):
      x: object
      y: object

    @tf.function
    def foo(x, t, l, odict, my_type):
      self.assertIsInstance(x, tf.Tensor)
      self.assertIsInstance(t, tuple)
      self.assertIsInstance(l, list)
      self.assertIsInstance(odict, collections.OrderedDict)
      self.assertIsInstance(my_type, MyType)
      return x + t[0] + l[0] + odict['foo'] + my_type.x

    foo = tensorflow_computation.tf_computation(foo)

    concrete_fn = foo.fn_for_argument_type(
        federated_language.to_type([
            np.int32,
            (np.int32, np.int32),
            [np.int32, np.int32],
            collections.OrderedDict([('foo', np.int32), ('bar', np.int32)]),
            MyType(np.int32, np.int32),
        ])
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(),
        (
            '(<int32,<int32,int32>,<int32,int32>,<foo=int32,bar=int32>,<x=int32,y=int32>>'
            ' -> int32)'
        ),
    )
    concrete_fn = foo.fn_for_argument_type(
        federated_language.to_type([
            np.float32,
            (np.float32, np.float32),
            [np.float32, np.float32],
            collections.OrderedDict([('foo', np.float32), ('bar', np.float32)]),
            MyType(np.float32, np.float32),
        ])
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(),
        (
            '(<float32,<float32,float32>,<float32,float32>,<foo=float32,bar=float32>,<x=float32,y=float32>>'
            ' -> float32)'
        ),
    )

  def test_returns_tuple_structured(self):
    class MyType(NamedTuple):
      x: object
      y: object

    @tf.function
    def foo():
      return (
          1,
          (2, 3.0),
          [4, 5.0],
          collections.OrderedDict([('foo', 6), ('bar', 7.0)]),
          MyType(True, False),
      )

    foo = tensorflow_computation.tf_computation(foo)

    # pyformat: disable
    self.assertEqual(
        foo.type_signature.compact_representation(),
        '( -> <int32,<int32,float32>,<int32,float32>,<foo=int32,bar=float32>,<x=bool,y=bool>>)'
    )
    # pyformat: enable

  def test_takes_namedtuple_typed(self):
    class MyType(NamedTuple):
      x: object
      y: object

    @tf.function
    def foo(x):
      self.assertIsInstance(x, MyType)
      return x.x + x.y

    foo = tensorflow_computation.tf_computation(foo, MyType(tf.int32, tf.int32))
    self.assertEqual(
        foo.type_signature.compact_representation(),
        '(<x=int32,y=int32> -> int32)',
    )

  def test_takes_namedtuple_polymorphic(self):
    class MyType(NamedTuple):
      x: object
      y: object

    @tf.function
    def foo(t):
      self.assertIsInstance(t, MyType)
      return t.x + t.y

    foo = tensorflow_computation.tf_computation(foo)

    concrete_fn = foo.fn_for_argument_type(
        federated_language.StructWithPythonType(
            [('x', np.int32), ('y', np.int32)], MyType
        )
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(),
        '(<x=int32,y=int32> -> int32)',
    )
    concrete_fn = foo.fn_for_argument_type(
        federated_language.StructWithPythonType(
            [('x', np.float32), ('y', np.float32)], MyType
        )
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(),
        '(<x=float32,y=float32> -> float32)',
    )

  def test_with_variable(self):
    v_slot = []

    @tf.function(autograph=False)
    def foo(x):
      if not v_slot:
        v_slot.append(tf.Variable(0))
      v = v_slot[0]
      v.assign(1)
      return v + x

    foo = tensorflow_computation.tf_computation(foo, tf.int32)
    self.assertEqual(
        foo.type_signature.compact_representation(), '(int32 -> int32)'
    )

  def test_fails_with_bad_types(self):
    function = federated_language.FunctionType(
        None, federated_language.TensorType(np.int32)
    )
    federated = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    tuple_on_function = federated_language.StructType([federated, function])

    def foo(x):
      del x  # Unused.

    with self.assertRaisesRegex(
        TypeError,
        r'you have attempted to create one with the type {int32}@CLIENTS',
    ):
      tensorflow_computation.tf_computation(foo, federated)

    with self.assertRaisesRegex(
        TypeError,
        r'you have attempted to create one with the type \( -> int32\)',
    ):
      tensorflow_computation.tf_computation(foo, function)

    with self.assertRaisesRegex(
        TypeError, r'you have attempted to create one with the type placement'
    ):
      tensorflow_computation.tf_computation(
          foo, federated_language.PlacementType()
      )

    with self.assertRaisesRegex(
        TypeError, r'you have attempted to create one with the type T'
    ):
      tensorflow_computation.tf_computation(
          foo, federated_language.AbstractType('T')
      )

    with self.assertRaisesRegex(
        TypeError,
        r'you have attempted to create one with the type <{int32}@CLIENTS,\( '
        r'-> int32\)>',
    ):
      tensorflow_computation.tf_computation(foo, tuple_on_function)

  def test_stackframes_in_errors(self):
    class DummyError(RuntimeError):
      pass

    with self.assertRaises(DummyError):

      @tensorflow_computation.tf_computation
      def _():
        raise DummyError()

  def test_error_on_non_callable_non_type(self):
    with self.assertRaises(TypeError):
      tensorflow_computation.tf_computation(5)

  def test_stack_resets_on_none_returned(self):
    stack = federated_language.framework.get_context_stack()
    self.assertIsInstance(
        stack.current, federated_language.framework.RuntimeErrorContext
    )

    with self.assertRaises(
        federated_language.framework.ComputationReturnedNoneError
    ):

      @tensorflow_computation.tf_computation()
      def _():
        pass

    self.assertIsInstance(
        stack.current, federated_language.framework.RuntimeErrorContext
    )

  def test_custom_numpy_dtype(self):

    @tensorflow_computation.tf_computation(tf.bfloat16)
    def foo(x):
      return x

    self.assertEqual(
        foo.type_signature,
        federated_language.FunctionType(
            parameter=federated_language.TensorType(tf.bfloat16.as_numpy_dtype),
            result=federated_language.TensorType(tf.bfloat16.as_numpy_dtype),
        ),
    )


if __name__ == '__main__':
  absltest.main()
