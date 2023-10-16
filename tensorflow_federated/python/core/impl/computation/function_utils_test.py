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
import numpy as np

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.computation import function_utils
from tensorflow_federated.python.core.impl.types import computation_types


class FunctionUtilsTest(parameterized.TestCase):

  # pyformat: disable
  @parameterized.named_parameters(
      ('args_only',
       inspect.signature(lambda a: None),
       [np.int32],
       collections.OrderedDict()),
      ('args_and_kwargs_unnamed',
       inspect.signature(lambda a, b=True: None),
       [np.int32, np.bool_],
       collections.OrderedDict()),
      ('args_and_kwargs_named',
       inspect.signature(lambda a, b=True: None),
       [np.int32],
       collections.OrderedDict(b=np.bool_)),
      ('args_and_kwargs_default_int',
       inspect.signature(lambda a=10, b=True: None),
       [np.int32],
       collections.OrderedDict(b=np.bool_)),
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
       [np.int32],
       collections.OrderedDict()),
      ('args_and_kwargs',
       inspect.signature(lambda a=10, b=True: None),
       [np.bool_],
       collections.OrderedDict(b=np.bool_)),
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
      ('int', np.int32, False),
      ('tuple_unnamed', [np.int32, np.int32], True),
      ('tuple_partially_named', [np.int32, ('b', np.int32)], True),
      ('tuple_named', [('a', np.int32), ('b', np.int32)], True),
      ('tuple_partially_named_kwargs_first', [('a', np.int32), np.int32],
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
      ('tuple_unnamed_1', [np.int32], [np.int32], {}),
      ('tuple_named_1', [('a', np.int32)], [], {'a': np.int32}),
      ('tuple_unnamed_2', [np.int32, np.bool_], [np.int32, np.bool_], {}),
      ('tuple_partially_named',
       [np.int32, ('b', np.bool_)], [np.int32], {'b': np.bool_}),
      ('tuple_named_2',
       [('a', np.int32), ('b', np.bool_)], [], {'a': np.int32, 'b': np.bool_}),
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
      ('int', [1], {}, [np.int32], [(None, 1)]),
      ('tuple_unnamed_with_args',
       [1, True], {}, [np.int32, np.bool_], [(None, 1), (None, True)]),
      ('tuple_named_with_args', [1, True], {},
       [('x', np.int32), ('y', np.bool_)], [('x', 1), ('y', True)]),
      ('tuple_named_with_args_and_kwargs', [1], {'y': True},
       [('x', np.int32), ('y', np.bool_)], [('x', 1), ('y', True)]),
      ('tuple_with_kwargs', [], {'x': 1, 'y': True},
       [('x', np.int32), ('y', np.bool_)], [('x', 1), ('y', True)]),
      ('tuple_with_args_odict', [], collections.OrderedDict([('y', True), ('x', 1)]),
       [('x', np.int32), ('y', np.bool_)], [('x', 1), ('y', True)]))
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
          [], {'x': 1, 'y': True}, [np.int32, np.bool_]
      )

  # pyformat: disable
  @parameterized.named_parameters(
      ('none', None, [], {}, 'None'),
      ('int', np.int32, [1], {}, '1'),
      ('tuple_unnamed', [np.int32, np.bool_], [1, True], {}, '<1,True>'),
      ('tuple_named_with_args', [('x', np.int32), ('y', np.bool_)], [1, True],
       {}, '<x=1,y=True>'),
      ('tuple_named_with_kwargs', [('x', np.int32), ('y', np.bool_)], [1],
       {'y': True}, '<x=1,y=True>'),
      ('tuple_with_args_struct', [np.int32, np.bool_],
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
       computation_types.TensorType(np.int32),
       None,
       20,
       30),
      ('add',
       lambda x, y: x + y,
       computation_types.StructType([np.int32, np.int32]),
       None,
       structure.Struct([('x', 5), ('y', 6)]),
       11),
      ('str_tuple',
       lambda *args: str(args),
       computation_types.StructType([np.int32, np.int32]),
       True,
       structure.Struct([(None, 5), (None, 6)]),
       '(5, 6)'),
      ('str_tuple_with_named_type',
       lambda *args: str(args),
       computation_types.StructType([('x', np.int32), ('y', np.int32)]),
       False,
       structure.Struct([('x', 5), ('y', 6)]),
       '(Struct([(\'x\', 5), (\'y\', 6)]),)'),
      ('str_ing',
       lambda x: str(x),  # pylint: disable=unnecessary-lambda
       computation_types.StructWithPythonType([np.int32], list),
       None,
       structure.Struct([(None, 10)]),
       '[10]'),
  )
  # pyformat: enable
  def test_wrap_as_zero_or_one_arg_callable(
      self, fn, parameter_type, unpack, arg, expected_result
  ):
    wrapped_fn = function_utils.wrap_as_zero_or_one_arg_callable(
        fn, parameter_type, unpack
    )
    actual_result = wrapped_fn(arg) if parameter_type else wrapped_fn()
    self.assertEqual(actual_result, expected_result)


if __name__ == '__main__':
  absltest.main()
