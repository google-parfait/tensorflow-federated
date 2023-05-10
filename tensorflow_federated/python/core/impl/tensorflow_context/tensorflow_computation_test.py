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
from typing import Any, NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import attr
import tensorflow as tf

from tensorflow_federated.python.common_libs import golden
from tensorflow_federated.python.core.impl.computation import computation_impl
from tensorflow_federated.python.core.impl.computation import computation_wrapper
from tensorflow_federated.python.core.impl.context_stack import get_context_stack
from tensorflow_federated.python.core.impl.context_stack import runtime_error_context
from tensorflow_federated.python.core.impl.tensorflow_context import tensorflow_computation
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_test_utils
from tensorflow_federated.python.tensorflow_libs import version_check


def one_arg_fn(x):
  return x > 10


def no_arg_fn():
  return 10


class TensorFlowComputationTest(parameterized.TestCase):

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
        computation_types.TensorType(tf.int32)
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(), '(int32 -> bool)'
    )
    concrete_fn = fn.fn_for_argument_type(
        computation_types.TensorType(tf.float32)
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
        computation_types.TensorType(tf.int32)
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(), '(int32 -> bool)'
    )
    concrete_fn = foo.fn_for_argument_type(
        computation_types.TensorType(tf.float32)
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
        computation_types.StructType([
            computation_types.TensorType(tf.int32),
            computation_types.TensorType(tf.int32),
        ])
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(),
        '(<int32,int32> -> int32)',
    )
    concrete_fn = foo.fn_for_argument_type(
        computation_types.StructType([
            computation_types.TensorType(tf.float32),
            computation_types.TensorType(tf.float32),
        ])
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(),
        '(<float32,float32> -> float32)',
    )

  def test_takes_structured_tuple_typed(self):

    class MyType(NamedTuple):
      x: Any
      y: Any

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
      x: Any
      y: Any

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
        computation_types.to_type([
            tf.int32,
            (tf.int32, tf.int32),
            [tf.int32, tf.int32],
            collections.OrderedDict([('foo', tf.int32), ('bar', tf.int32)]),
            MyType(tf.int32, tf.int32),
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
        computation_types.to_type([
            tf.float32,
            (tf.float32, tf.float32),
            [tf.float32, tf.float32],
            collections.OrderedDict([('foo', tf.float32), ('bar', tf.float32)]),
            MyType(tf.float32, tf.float32),
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
      x: Any
      y: Any

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
      x: Any
      y: Any

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
      x: Any
      y: Any

    @tf.function
    def foo(t):
      self.assertIsInstance(t, MyType)
      return t.x + t.y

    foo = tensorflow_computation.tf_computation(foo)

    concrete_fn = foo.fn_for_argument_type(
        computation_types.StructWithPythonType(
            [('x', tf.int32), ('y', tf.int32)], MyType
        )
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(),
        '(<x=int32,y=int32> -> int32)',
    )
    concrete_fn = foo.fn_for_argument_type(
        computation_types.StructWithPythonType(
            [('x', tf.float32), ('y', tf.float32)], MyType
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

  def test_does_not_raise_type_error_with_sequence_inputs_and_outputs(self):
    try:

      @tensorflow_computation.tf_computation(
          computation_types.SequenceType(tf.int32)
      )
      def _(x):
        return x

    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  def test_fails_with_bad_types(self):
    function = computation_types.FunctionType(
        None, computation_types.TensorType(tf.int32)
    )
    federated = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    tuple_on_function = computation_types.StructType([federated, function])

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
          foo, computation_types.PlacementType()
      )

    with self.assertRaisesRegex(
        TypeError, r'you have attempted to create one with the type T'
    ):
      tensorflow_computation.tf_computation(
          foo, computation_types.AbstractType('T')
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

    with golden.check_raises_traceback(
        'tensorflow_computation_traceback.expected', DummyError
    ):

      @tensorflow_computation.tf_computation
      def _():
        raise DummyError()

  def test_error_on_non_callable_non_type(self):
    with golden.check_raises_traceback(
        'non_callable_non_type_traceback.expected', TypeError
    ):
      tensorflow_computation.tf_computation(5)

  def test_stack_resets_on_none_returned(self):
    stack = get_context_stack.get_context_stack()
    self.assertIsInstance(
        stack.current, runtime_error_context.RuntimeErrorContext
    )

    with self.assertRaises(computation_wrapper.ComputationReturnedNoneError):
      @tensorflow_computation.tf_computation()
      def _():
        pass

    self.assertIsInstance(
        stack.current, runtime_error_context.RuntimeErrorContext
    )

  def test_check_returns_type_with_tensorflow_computation_succeeds(self):
    @tensorflow_computation.tf_computation(tf.int32)
    @computation_wrapper.check_returns_type(tf.int32)
    def _(x):
      return x

  def test_check_returns_type_with_tensorflow_computation_fails(self):
    with self.assertRaises(TypeError):
      @tensorflow_computation.tf_computation(tf.int32)
      @computation_wrapper.check_returns_type(tf.int32)
      def _(x):
        return (x, x)

  def test_check_returns_type_with_tensorflow_computation_picking_up_named_parameters(
      self,
  ):
    @tensorflow_computation.tf_computation(tf.int32, tf.int32)
    @computation_wrapper.check_returns_type(tf.int32)
    def f(a, b):
      del b
      return a

    self.assertEqual(
        f.type_signature,
        computation_types.FunctionType(
            collections.OrderedDict(a=tf.int32, b=tf.int32), tf.int32
        ),
    )

  def test_check_returns_type_fails_with_mismatched_container_type(self):
    with golden.check_raises_traceback(
        'returns_type_container_mismatch_traceback.expected', TypeError
    ):
      # This test fails because it `check_returns_type` with a `tuple`,
      # but returns a `list`.
      @tensorflow_computation.tf_computation(tf.int32)
      @computation_wrapper.check_returns_type((tf.int32, tf.int32))
      def _(a):
        return [a, a]

  def test_check_returns_type_fails_with_more_general_tensorspec(self):
    type_with_known_shape = computation_types.TensorType(tf.int32, [1])
    type_with_unknown_shape = computation_types.TensorType(tf.int32, [None])

    with self.assertRaises(TypeError):
      @tensorflow_computation.tf_computation(type_with_known_shape)
      @computation_wrapper.check_returns_type(type_with_unknown_shape)
      def _(a):
        return a

  def test_check_returns_type_attrs_type(self):
    @attr.s(frozen=True, eq=False, slots=True)
    class MyAttrs:
      a = attr.ib()
      b = attr.ib()

    expected_return_type = MyAttrs(a=tf.int32, b=tf.int32)

    @tensorflow_computation.tf_computation
    @computation_wrapper.check_returns_type(expected_return_type)
    def _():
      return MyAttrs(a=0, b=0)


@absltest.skipUnless(
    version_check.is_tensorflow_version_newer('2.11', tf),
    'requires tensorflow 2.11',
)
class TensorFlowFunctionComputationTest(parameterized.TestCase):

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
    fn = tensorflow_computation.experimental_tf_fn_computation(fn, fn_arg_type)
    self.assertEqual(
        fn.type_signature.compact_representation(), expected_representation
    )

  @parameterized.named_parameters(
      ('lambda', lambda x: x > 10),
      ('function', one_arg_fn),
      ('tf_function', tf.function(one_arg_fn)),
  )
  def test_tf_computation_without_type(self, fn):
    fn = tensorflow_computation.experimental_tf_fn_computation(fn)
    concrete_fn = fn.fn_for_argument_type(
        computation_types.TensorType(tf.int32)
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(), '(int32 -> bool)'
    )
    concrete_fn = fn.fn_for_argument_type(
        computation_types.TensorType(tf.float32)
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(), '(float32 -> bool)'
    )

  def test_decorate_sparse_tensor(self):
    x_spec = tf.SparseTensorSpec(shape=[10, 10], dtype=tf.float64)
    y_spec = tf.SparseTensorSpec(shape=[15, 15], dtype=tf.float64)

    @tensorflow_computation.experimental_tf_fn_computation(x_spec, y_spec)
    def sparse_add(x, y):
      return tf.sparse.add(x, y)

    type_test_utils.assert_types_identical(
        sparse_add.type_signature,
        computation_types.FunctionType(
            parameter=collections.OrderedDict(x=x_spec, y=y_spec),
            result=tf.SparseTensorSpec(shape=[15, 15], dtype=tf.float64),
        ),
    )

  def test_decorate_ragged_tensor(self):
    ragged_tensor = tf.RaggedTensor.from_row_splits([0, 1, 2, 3], [0, 1, 4])
    ragged_tensor_spec = tf.RaggedTensorSpec.from_value(ragged_tensor)

    @tensorflow_computation.experimental_tf_fn_computation(ragged_tensor_spec)
    def multiply_two(x):
      return tf.ragged.map_flat_values(lambda y: tf.cast(y * 2, tf.float64), x)

    type_test_utils.assert_types_identical(
        multiply_two.type_signature,
        computation_types.FunctionType(
            parameter=ragged_tensor_spec,
            result=tf.RaggedTensorSpec(
                dtype=tf.float64,
                ragged_rank=1,
                row_splits_dtype=tf.int64,
                flat_values_spec=tf.TensorSpec(shape=None, dtype=tf.float64),
            ),
        ),
    )

  def test_decorate_as_typed_fn(self):
    @tensorflow_computation.experimental_tf_fn_computation(tf.int32)
    def foo(x):
      return x > 10

    self.assertEqual(
        foo.type_signature.compact_representation(), '(int32 -> bool)'
    )

  def test_decorate_as_polymorphic_fn(self):
    @tensorflow_computation.experimental_tf_fn_computation
    def foo(x):
      return x > 10

    concrete_fn = foo.fn_for_argument_type(
        computation_types.TensorType(tf.int32)
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(), '(int32 -> bool)'
    )
    concrete_fn = foo.fn_for_argument_type(
        computation_types.TensorType(tf.float32)
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(), '(float32 -> bool)'
    )

  def test_decorate_as_no_arg_fn(self):
    @tensorflow_computation.experimental_tf_fn_computation
    def foo():
      return 10

    self.assertEqual(foo.type_signature.compact_representation(), '( -> int32)')

  def test_takes_tuple_typed(self):
    @tf.function
    def foo(t):
      return t[0] + t[1]

    foo = tensorflow_computation.experimental_tf_fn_computation(
        foo, (tf.int32, tf.int32)
    )
    self.assertEqual(
        foo.type_signature.compact_representation(), '(<int32,int32> -> int32)'
    )

  def test_takes_tuple_polymorphic(self):
    def foo(t):
      return t[0] + t[1]

    foo = tensorflow_computation.experimental_tf_fn_computation(foo)

    concrete_fn = foo.fn_for_argument_type(
        computation_types.StructType([
            computation_types.TensorType(tf.int32),
            computation_types.TensorType(tf.int32),
        ])
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(),
        '(<int32,int32> -> int32)',
    )
    concrete_fn = foo.fn_for_argument_type(
        computation_types.StructType([
            computation_types.TensorType(tf.float32),
            computation_types.TensorType(tf.float32),
        ])
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(),
        '(<float32,float32> -> float32)',
    )

  def test_takes_structured_tuple_typed(self):

    class MyType(NamedTuple):
      x: Any
      y: Any

    @tf.function
    def foo(x, t, l, odict, my_type):
      self.assertIsInstance(x, tf.Tensor)
      self.assertIsInstance(t, tuple)
      self.assertIsInstance(l, list)
      self.assertIsInstance(odict, collections.OrderedDict)
      self.assertIsInstance(my_type, MyType)
      return x + t[0] + l[0] + odict['foo'] + my_type.x

    foo = tensorflow_computation.experimental_tf_fn_computation(
        foo,
        [
            tf.int32,
            (tf.int32, tf.int32),
            [tf.int32, tf.int32],
            collections.OrderedDict(foo=tf.int32, bar=tf.int32),
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
      x: Any
      y: Any

    @tf.function
    def foo(x, t, l, odict, my_type):
      self.assertIsInstance(x, tf.Tensor)
      self.assertIsInstance(t, tuple)
      self.assertIsInstance(l, list)
      self.assertIsInstance(odict, collections.OrderedDict)
      self.assertIsInstance(my_type, MyType)
      return x + t[0] + l[0] + odict['foo'] + my_type.x

    foo = tensorflow_computation.experimental_tf_fn_computation(foo)

    concrete_fn = foo.fn_for_argument_type(
        computation_types.to_type([
            tf.int32,
            (tf.int32, tf.int32),
            [tf.int32, tf.int32],
            collections.OrderedDict([('foo', tf.int32), ('bar', tf.int32)]),
            MyType(tf.int32, tf.int32),
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
        computation_types.to_type([
            tf.float32,
            (tf.float32, tf.float32),
            [tf.float32, tf.float32],
            collections.OrderedDict([('foo', tf.float32), ('bar', tf.float32)]),
            MyType(tf.float32, tf.float32),
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
      x: Any
      y: Any

    @tf.function
    def foo():
      return (
          1,
          (2, 3.0),
          [4, 5.0],
          collections.OrderedDict([('foo', 6), ('bar', 7.0)]),
          MyType(True, False),
      )

    foo = tensorflow_computation.experimental_tf_fn_computation(foo)

    # pyformat: disable
    self.assertEqual(
        foo.type_signature.compact_representation(),
        '( -> <int32,<int32,float32>,<int32,float32>,<foo=int32,bar=float32>,<x=bool,y=bool>>)'
    )
    # pyformat: enable

  def test_argument_ordering_matches(self):
    """Tests that the argument ordering matches.

    This test asserts classes inherting from `collections.abc.Mapping` result in
    sorted fields for tf.function arguments, everything else does not. If the
    arg_def traversal doesn't correctly match the tff.Type travsel, the dtypes
    of the fields will not match up and this test will fail If the arg_def
    traversal doesn't correctly match the tff.Type travsel, the dtypes of the
    fields will not match up and this test will fail
    """

    @attr.s
    class TestAttrs:
      q = attr.ib()
      p = attr.ib()

    class TestTuple(NamedTuple):
      y: str
      x: int

    class TestDict(collections.OrderedDict):
      ...

    def foo():
      return (
          TestAttrs(q=tf.constant(1.0, tf.float64), p=tf.constant(1, tf.int16)),
          collections.OrderedDict(b=tf.constant(1, tf.int64), c=1.0, a='abc'),
          TestTuple(y='abc', x=tf.constant(5, tf.int8)),
          TestDict(z=tf.constant(1.0, tf.float16), x='abc', y=5),
      )

    foo = tensorflow_computation.experimental_tf_fn_computation(foo)
    type_test_utils.assert_types_identical(
        foo.type_signature,
        computation_types.FunctionType(
            parameter=None,
            result=(
                TestAttrs(
                    q=computation_types.TensorType(tf.float64),
                    p=computation_types.TensorType(tf.int16),
                ),
                collections.OrderedDict(
                    b=computation_types.TensorType(tf.int64),
                    c=computation_types.TensorType(tf.float32),
                    a=computation_types.TensorType(tf.string),
                ),
                TestTuple(
                    y=computation_types.TensorType(tf.string),
                    x=computation_types.TensorType(tf.int8),
                ),
                TestDict(
                    z=computation_types.TensorType(tf.float16),
                    x=computation_types.TensorType(tf.string),
                    y=computation_types.TensorType(tf.int32),
                ),
            ),
        ),
    )

  def test_takes_namedtuple_typed(self):

    class MyType(NamedTuple):
      x: Any
      y: Any

    @tf.function
    def foo(x):
      self.assertIsInstance(x, MyType)
      return x.x + x.y

    foo = tensorflow_computation.experimental_tf_fn_computation(
        foo, MyType(tf.int32, tf.int32)
    )
    self.assertEqual(
        foo.type_signature.compact_representation(),
        '(<x=int32,y=int32> -> int32)',
    )

  def test_takes_namedtuple_polymorphic(self):

    class MyType(NamedTuple):
      x: Any
      y: Any

    @tf.function
    def foo(t):
      self.assertIsInstance(t, MyType)
      return t.x + t.y

    foo = tensorflow_computation.experimental_tf_fn_computation(foo)

    concrete_fn = foo.fn_for_argument_type(
        computation_types.StructWithPythonType(
            [('x', tf.int32), ('y', tf.int32)], MyType
        )
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(),
        '(<x=int32,y=int32> -> int32)',
    )
    concrete_fn = foo.fn_for_argument_type(
        computation_types.StructWithPythonType(
            [('x', tf.float32), ('y', tf.float32)], MyType
        )
    )
    self.assertEqual(
        concrete_fn.type_signature.compact_representation(),
        '(<x=float32,y=float32> -> float32)',
    )

  def test_with_local_variable(self):
    def foo(x):
      v = tf.Variable(0)
      v.assign(x)
      return v

    foo = tensorflow_computation.experimental_tf_fn_computation(foo, tf.int32)
    self.assertEqual(
        foo.type_signature.compact_representation(), '(int32 -> int32)'
    )

  def test_experimental_tf_computation_with_variable_and_layout(self):
    def foo(x):
      v = tf.Variable(0, name='x')
      v.assign(1)
      return v + x

    foo = tensorflow_computation.experimental_tf_fn_computation(
        foo, tf.int32, layout_map={'x': 'unsharded'}
    )
    concrete_foo = computation_impl.ConcreteComputation.get_proto(foo)
    self.assertLen(
        concrete_foo.tensorflow_function.layout_map.name_to_sharding_spec, 1
    )
    self.assertEqual(
        concrete_foo.tensorflow_function.layout_map.name_to_sharding_spec['x'],
        'unsharded',
    )

  def test_experimental_tf_computation_decorator_with_variable_and_layout(self):
    @tensorflow_computation.experimental_tf_fn_computation(
        tf.int32, layout_map={'x': 'unsharded'}
    )
    def foo(x):
      v = tf.Variable(0, name='x')
      v.assign(1)
      return v + x

    concrete_foo = computation_impl.ConcreteComputation.get_proto(foo)
    self.assertLen(
        concrete_foo.tensorflow_function.layout_map.name_to_sharding_spec, 1
    )
    self.assertEqual(
        concrete_foo.tensorflow_function.layout_map.name_to_sharding_spec['x'],
        'unsharded',
    )

  def test_with_local_variable_user_specified_lifting(self):
    def foo(x):
      # The user has explicitly asked for variable lifting, but the
      # _no_lifting_creator should override this and prevent lifting.
      v = tf.Variable(0, experimental_enable_variable_lifting=True)
      v.assign(x)
      return v

    with self.subTest('variable_parameter'):
      foo = tensorflow_computation.experimental_tf_fn_computation(foo, tf.int32)
      self.assertEqual(
          foo.type_signature.compact_representation(), '(int32 -> int32)'
      )

    # Same as above, but using a variable creator scope.
    def _enable_lifting(next_creator_fn, **kwargs):
      kwargs['experimental_enable_variable_lifting'] = True
      return next_creator_fn(**kwargs)

    def bar(x):
      with tf.variable_creator_scope(_enable_lifting):
        v = tf.Variable(0)
      v.assign(x)
      return v

    with self.subTest('variable_scope'):
      bar = tensorflow_computation.experimental_tf_fn_computation(bar, tf.int32)
      self.assertEqual(
          foo.type_signature.compact_representation(), '(int32 -> int32)'
      )

  def test_with_captured_variable_raise_error(self):
    v_slot = []

    def foo(x):
      if not v_slot:
        v_slot.append(tf.Variable(0))
      v = v_slot[0]
      v.assign(1)
      return v + x

    with self.assertRaises(tensorflow_computation.CapturedVariableError):
      tensorflow_computation.experimental_tf_fn_computation(foo, tf.int32)

  def test_tf_computation_with_variable_and_layout(self):
    def foo(x):
      v = tf.Variable(0, name='x')
      v.assign(1)
      return v + x

    foo = tensorflow_computation.tf_computation(
        foo, tf.int32, layout_map={'x': 'unsharded'}
    )
    concrete_foo = computation_impl.ConcreteComputation.get_proto(foo)
    self.assertLen(concrete_foo.tensorflow.layout_map.name_to_sharding_spec, 1)
    self.assertEqual(
        concrete_foo.tensorflow.layout_map.name_to_sharding_spec['x'],
        'unsharded',
    )

  def test_tf_computation_decorator_with_variable_and_layout(self):
    @tensorflow_computation.tf_computation(
        tf.int32, layout_map={'x': 'unsharded'}
    )
    def foo(x):
      v = tf.Variable(0, name='x')
      v.assign(1)
      return v + x

    concrete_foo = computation_impl.ConcreteComputation.get_proto(foo)
    self.assertLen(concrete_foo.tensorflow.layout_map.name_to_sharding_spec, 1)
    self.assertEqual(
        concrete_foo.tensorflow.layout_map.name_to_sharding_spec['x'],
        'unsharded',
    )

  def test_does_not_raise_type_error_with_sequence_inputs_and_outputs(self):
    try:

      @tensorflow_computation.experimental_tf_fn_computation(
          computation_types.SequenceType(tf.int32)
      )
      def foo(x):
        return x

    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  def test_raises_for_non_variant_for_sequence_type(self):
    # TensorFlow will complain that the tf.function is already concretized
    # to a type that is incompatible with the dataset.
    with self.assertRaises((TypeError, ValueError)):
      @tensorflow_computation.experimental_tf_fn_computation(
          computation_types.SequenceType(tf.int32)
      )
      @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.int32)])
      def foo(x):
        return x

  def test_fails_with_bad_types(self):
    function = computation_types.FunctionType(
        None, computation_types.TensorType(tf.int32)
    )
    federated = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    tuple_on_function = computation_types.StructType([federated, function])

    def foo(x):
      del x  # Unused.

    with self.assertRaisesRegex(
        TypeError,
        r'you have attempted to create one with the type {int32}@CLIENTS',
    ):
      tensorflow_computation.experimental_tf_fn_computation(foo, federated)

    with self.assertRaisesRegex(
        TypeError,
        r'you have attempted to create one with the type \( -> int32\)',
    ):
      tensorflow_computation.experimental_tf_fn_computation(foo, function)

    with self.assertRaisesRegex(
        TypeError, r'you have attempted to create one with the type placement'
    ):
      tensorflow_computation.experimental_tf_fn_computation(
          foo, computation_types.PlacementType()
      )

    with self.assertRaisesRegex(
        TypeError, r'you have attempted to create one with the type T'
    ):
      tensorflow_computation.experimental_tf_fn_computation(
          foo, computation_types.AbstractType('T')
      )

    with self.assertRaisesRegex(
        TypeError,
        r'you have attempted to create one with the type <{int32}@CLIENTS,\( '
        r'-> int32\)>',
    ):
      tensorflow_computation.experimental_tf_fn_computation(
          foo, tuple_on_function
      )

  def test_stackframes_in_errors(self):
    self.skipTest(
        'b/257277613: AutoGraph swallows DummyError during tracing and outputs '
        'its own StagingError which is not a public symbol'
    )

    class DummyError(RuntimeError):
      pass

    with golden.check_raises_traceback(
        'tensorflow_function_computation_traceback.expected', DummyError
    ):

      @tensorflow_computation.experimental_tf_fn_computation
      def _():
        raise DummyError()

  def test_error_on_non_callable_non_type(self):
    with golden.check_raises_traceback(
        'function_non_callable_non_type_traceback.expected', TypeError
    ):
      tensorflow_computation.experimental_tf_fn_computation(5)

  def test_stack_resets_on_none_returned(self):
    stack = get_context_stack.get_context_stack()
    self.assertIsInstance(
        stack.current, runtime_error_context.RuntimeErrorContext
    )

    with self.assertRaises(computation_wrapper.ComputationReturnedNoneError):
      @tensorflow_computation.experimental_tf_fn_computation()
      def _():
        pass

    self.assertIsInstance(
        stack.current, runtime_error_context.RuntimeErrorContext
    )

  def test_check_returns_type_with_tensorflow_computation_succeeds(self):
    @tensorflow_computation.experimental_tf_fn_computation(tf.int32)
    @computation_wrapper.check_returns_type(tf.int32)
    def _(x):
      return x

  def test_check_returns_type_with_tensorflow_computation_fails(self):
    with self.assertRaises(TypeError):
      @tensorflow_computation.experimental_tf_fn_computation(tf.int32)
      @computation_wrapper.check_returns_type(tf.int32)
      def _(x):
        return (x, x)

  def test_check_returns_type_with_tensorflow_computation_picking_up_named_parameters(
      self,
  ):
    @tensorflow_computation.experimental_tf_fn_computation(tf.int32, tf.int32)
    @computation_wrapper.check_returns_type(tf.int32)
    def f(a, b):
      del b
      return a

    self.assertEqual(
        f.type_signature,
        computation_types.FunctionType(
            collections.OrderedDict(a=tf.int32, b=tf.int32), tf.int32
        ),
    )

  def test_check_returns_type_fails_with_mismatched_container_type(self):
    self.skipTest(
        'b/257277613: AutoGraph causes non-deterministic stacktrace strings '
        'which will eventually fail golden checks.'
    )
    with golden.check_raises_traceback(
        'function_returns_type_container_mismatch_traceback.expected', TypeError
    ):
      # This test fails because it `check_returns_type` with a `tuple`,
      # but returns a `list`.
      @tensorflow_computation.experimental_tf_fn_computation(tf.int32)
      @computation_wrapper.check_returns_type((tf.int32, tf.int32))
      def _(a):
        return [a, a]

  def test_check_returns_type_fails_with_more_general_tensorspec(self):
    type_with_known_shape = computation_types.TensorType(tf.int32, [1])
    type_with_unknown_shape = computation_types.TensorType(tf.int32, [None])

    with self.assertRaises(TypeError):
      @tensorflow_computation.experimental_tf_fn_computation(
          type_with_known_shape
      )
      @computation_wrapper.check_returns_type(type_with_unknown_shape)
      def _(a):
        return a

  def test_check_returns_type_attrs_type(self):
    @attr.s(frozen=True, eq=False, slots=True)
    class MyAttrs:
      a = attr.ib()
      b = attr.ib()

    expected_return_type = MyAttrs(a=tf.int32, b=tf.int32)

    @tensorflow_computation.experimental_tf_fn_computation
    @computation_wrapper.check_returns_type(expected_return_type)
    def _():
      return MyAttrs(a=0, b=0)


if __name__ == '__main__':
  absltest.main()
