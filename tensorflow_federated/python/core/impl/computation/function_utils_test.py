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
import inspect

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.computation import function_utils
from tensorflow_federated.python.core.impl.types import computation_types


class FunctionUtilsTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('args_only',
       inspect.signature(lambda a: None),
       [tf.int32],
       collections.OrderedDict()),
      ('args_and_kwargs_unnamed',
       inspect.signature(lambda a, b=True: None),
       [tf.int32, tf.bool],
       collections.OrderedDict()),
      ('args_and_kwargs_named',
       inspect.signature(lambda a, b=True: None),
       [tf.int32],
       collections.OrderedDict(b=tf.bool)),
      ('args_and_kwargs_default_int',
       inspect.signature(lambda a=10, b=True: None),
       [tf.int32],
       collections.OrderedDict(b=tf.bool)),
  )
  # pyformat: enable
  def test_is_signature_compatible_with_types_true(
      self, signature, *args, **kwargs
  ):
    self.assertFalse(
        function_utils.is_signature_compatible_with_types(
            signature, *args, **kwargs
        )
    )

  # pyformat: disable
  @parameterized.named_parameters(
      ('args_only',
       inspect.signature(lambda a=True: None),
       [tf.int32],
       collections.OrderedDict()),
      ('args_and_kwargs',
       inspect.signature(lambda a=10, b=True: None),
       [tf.bool],
       collections.OrderedDict(b=tf.bool)),
  )
  # pyformat: enable
  def test_is_signature_compatible_with_types_false(
      self, signature, *args, **kwargs
  ):
    self.assertFalse(
        function_utils.is_signature_compatible_with_types(
            signature, *args, **kwargs
        )
    )

  # pyformat: disable
  @parameterized.named_parameters(
      ('int', tf.int32, False),
      ('tuple_unnamed', [tf.int32, tf.int32], True),
      ('tuple_partially_named', [tf.int32, ('b', tf.int32)], True),
      ('tuple_named', [('a', tf.int32), ('b', tf.int32)], True),
      ('tuple_partially_named_kwargs_first', [('a', tf.int32), tf.int32],
       False),
      ('struct', structure.Struct([(None, 1), ('a', 2)]), True),
      ('struct_kwargs_first', structure.Struct([('a', 1), (None, 2)]), False))
  # pyformat: enable
  def test_is_argument_struct(self, arg, expected_result):
    self.assertEqual(function_utils.is_argument_struct(arg), expected_result)

  # pyformat: disable
  @parameterized.named_parameters(
      ('tuple_unnamed', structure.Struct([(None, 1)]), [1], {}),
      ('tuple_partially_named', structure.Struct([(None, 1), ('a', 2)]),
       [1], {'a': 2}),
  )
  # pyformat: enable
  def test_unpack_args_from_structure(
      self, tuple_with_args, expected_args, expected_kwargs
  ):
    self.assertEqual(
        function_utils.unpack_args_from_struct(tuple_with_args),
        (expected_args, expected_kwargs),
    )

  # pyformat: disable
  @parameterized.named_parameters(
      ('tuple_unnamed_1', [tf.int32], [tf.int32], {}),
      ('tuple_named_1', [('a', tf.int32)], [], {'a': tf.int32}),
      ('tuple_unnamed_2', [tf.int32, tf.bool], [tf.int32, tf.bool], {}),
      ('tuple_partially_named',
       [tf.int32, ('b', tf.bool)], [tf.int32], {'b': tf.bool}),
      ('tuple_named_2',
       [('a', tf.int32), ('b', tf.bool)], [], {'a': tf.int32, 'b': tf.bool}),
  )
  # pyformat: enable
  def test_unpack_args_from_struct_type(
      self, tuple_with_args, expected_args, expected_kwargs
  ):
    args, kwargs = function_utils.unpack_args_from_struct(tuple_with_args)
    self.assertEqual(len(args), len(expected_args))
    for idx, arg in enumerate(args):
      self.assertTrue(
          arg.is_equivalent_to(computation_types.to_type(expected_args[idx]))
      )
    self.assertEqual(set(kwargs.keys()), set(expected_kwargs.keys()))
    for k, v in kwargs.items():
      self.assertTrue(
          v.is_equivalent_to(computation_types.to_type(expected_kwargs[k]))
      )

  def test_pack_args_into_struct_without_type_spec(self):
    self.assertEqual(
        function_utils.pack_args_into_struct([1], {'a': 10}),
        structure.Struct([(None, 1), ('a', 10)]),
    )
    self.assertIn(
        function_utils.pack_args_into_struct([1, 2], {'a': 10, 'b': 20}),
        [
            structure.Struct([
                (None, 1),
                (None, 2),
                ('a', 10),
                ('b', 20),
            ]),
            structure.Struct([
                (None, 1),
                (None, 2),
                ('b', 20),
                ('a', 10),
            ]),
        ],
    )
    self.assertIn(
        function_utils.pack_args_into_struct([], {'a': 10, 'b': 20}),
        [
            structure.Struct([('a', 10), ('b', 20)]),
            structure.Struct([('b', 20), ('a', 10)]),
        ],
    )
    self.assertEqual(
        function_utils.pack_args_into_struct([1], {}),
        structure.Struct([(None, 1)]),
    )

  # pyformat: disable
  @parameterized.named_parameters(
      ('int', [1], {}, [tf.int32], [(None, 1)]),
      ('tuple_unnamed_with_args',
       [1, True], {}, [tf.int32, tf.bool], [(None, 1), (None, True)]),
      ('tuple_named_with_args', [1, True], {},
       [('x', tf.int32), ('y', tf.bool)], [('x', 1), ('y', True)]),
      ('tuple_named_with_args_and_kwargs', [1], {'y': True},
       [('x', tf.int32), ('y', tf.bool)], [('x', 1), ('y', True)]),
      ('tuple_with_kwargs', [], {'x': 1, 'y': True},
       [('x', tf.int32), ('y', tf.bool)], [('x', 1), ('y', True)]),
      ('tuple_with_args_odict', [], collections.OrderedDict([('y', True), ('x', 1)]),
       [('x', tf.int32), ('y', tf.bool)], [('x', 1), ('y', True)]))
  # pyformat: enable
  def test_pack_args_into_struct_with_type_spec_expect_success(
      self, args, kwargs, type_spec, elements
  ):
    self.assertEqual(
        function_utils.pack_args_into_struct(args, kwargs, type_spec),
        structure.Struct(elements),
    )

  def test_pack_args_into_struct_named_to_unnamed_fails(self):
    with self.assertRaises(TypeError):
      function_utils.pack_args_into_struct(
          [], {'x': 1, 'y': True}, [tf.int32, tf.bool]
      )

  # pyformat: disable
  @parameterized.named_parameters(
      ('none', None, [], {}, 'None'),
      ('int', tf.int32, [1], {}, '1'),
      ('tuple_unnamed', [tf.int32, tf.bool], [1, True], {}, '<1,True>'),
      ('tuple_named_with_args', [('x', tf.int32), ('y', tf.bool)], [1, True],
       {}, '<x=1,y=True>'),
      ('tuple_named_with_kwargs', [('x', tf.int32), ('y', tf.bool)], [1],
       {'y': True}, '<x=1,y=True>'),
      ('tuple_with_args_struct', [tf.int32, tf.bool],
       [structure.Struct([(None, 1), (None, True)])], {}, '<1,True>'))
  # pyformat: enable
  def test_pack_args(self, parameter_type, args, kwargs, expected_value_string):
    self.assertEqual(
        str(function_utils.pack_args(parameter_type, args, kwargs)),
        expected_value_string,
    )

  # pyformat: disable
  @parameterized.named_parameters(
      ('const', lambda: 10, None, None, None, 10),
      ('add_const', lambda x=1: x + 10, None, None, None, 11),
      ('add_const_with_type',
       lambda x=1: x + 10,
       computation_types.TensorType(tf.int32),
       None,
       20,
       30),
      ('add',
       lambda x, y: x + y,
       computation_types.StructType([tf.int32, tf.int32]),
       None,
       structure.Struct([('x', 5), ('y', 6)]),
       11),
      ('str_tuple',
       lambda *args: str(args),
       computation_types.StructType([tf.int32, tf.int32]),
       True,
       structure.Struct([(None, 5), (None, 6)]),
       '(5, 6)'),
      ('str_tuple_with_named_type',
       lambda *args: str(args),
       computation_types.StructType([('x', tf.int32), ('y', tf.int32)]),
       False,
       structure.Struct([('x', 5), ('y', 6)]),
       '(Struct([(\'x\', 5), (\'y\', 6)]),)'),
      ('str_ing',
       lambda x: str(x),  # pylint: disable=unnecessary-lambda
       computation_types.StructWithPythonType([tf.int32], list),
       None,
       structure.Struct([(None, 10)]),
       '[10]'),
  )
  # pyformat: enable
  def test_wrap_as_zero_or_one_arg_callable(
      self, fn, parameter_type, unpack, arg, expected_result
  ):
    unpack_arguments = function_utils.create_argument_unpacking_fn(
        fn, parameter_type, unpack
    )
    args, kwargs = unpack_arguments(arg)
    actual_result = fn(*args, **kwargs)
    self.assertEqual(actual_result, expected_result)


if __name__ == '__main__':
  absltest.main()
