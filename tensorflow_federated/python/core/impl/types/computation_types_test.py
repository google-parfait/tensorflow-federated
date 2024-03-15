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
from typing import NamedTuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import attrs
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import golden
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


_ALL_INTERNED_TYPES = [
    computation_types.AbstractType,
    computation_types.FederatedType,
    computation_types.FunctionType,
    computation_types.PlacementType,
    computation_types.SequenceType,
    computation_types.StructType,
    computation_types.StructWithPythonType,
    computation_types.TensorType,
]


@attrs.define
class TestAttrs:
  a: int = 1
  a: bool = True


class TestNamedTuple(NamedTuple):
  a: int = 1
  b: bool = True


class TypeMismatchErrorMessageTest(absltest.TestCase):

  def test_short_compact_repr(self):
    first = computation_types.TensorType(np.int32)
    second = computation_types.TensorType(np.bool_)
    actual = computation_types.type_mismatch_error_message(
        first, second, computation_types.TypeRelation.EQUIVALENT
    )
    golden.check_string('short_compact_repr.expected', actual)

  def test_long_formatted_with_diff(self):
    int32 = computation_types.TensorType(np.int32)
    first = computation_types.StructType([(None, int32)] * 20)
    second = computation_types.StructType([(None, int32)] * 21)
    actual = computation_types.type_mismatch_error_message(
        first, second, computation_types.TypeRelation.EQUIVALENT
    )
    golden.check_string('long_formatted_with_diff.expected', actual)

  def test_container_types_full_repr(self):
    first = computation_types.StructWithPythonType([], list)
    second = computation_types.StructWithPythonType([], tuple)
    actual = computation_types.type_mismatch_error_message(
        first, second, computation_types.TypeRelation.EQUIVALENT
    )
    golden.check_string('container_types_full_repr.expected', actual)


class InternTest(parameterized.TestCase):

  @parameterized.named_parameters(
      [(cls.__name__, cls) for cls in _ALL_INTERNED_TYPES]
  )
  def test_hashable_from_init_args_has_correct_parameters(self, cls):
    hashable_from_init_args_signature = inspect.signature(
        cls._hashable_from_init_args
    )
    actual_parameters = hashable_from_init_args_signature.parameters
    init_signature = inspect.signature(cls.__init__)
    # A copy of the parameters is created because `mappingproxy` object does not
    # support item deletion.
    expected_parameters = init_signature.parameters.copy()
    del expected_parameters['self']
    self.assertEqual(actual_parameters, expected_parameters)

  def test_call_raises_type_error_with_unhashable_key(self):

    class Foo(metaclass=computation_types._Intern):  # pylint: disable=undefined-variable

      @classmethod
      def _hashable_from_init_args(cls, *args, **kwargs):
        del args, kwargs  # Unused.
        return []

    with self.assertRaises(TypeError):
      _ = Foo()


class TypeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'tensor_type_same_dtype_and_shape',
          computation_types.TensorType(np.int32),
          computation_types.TensorType(np.int32),
      ),
  )
  def test_check_equivalent_to_does_not_raise_types_not_equivalent_error(
      self, type_spec, other
  ):
    try:
      type_spec.check_equivalent_to(other)
    except computation_types.TypesNotEquivalentError:
      self.fail('Raised `TypesNotEquivalentError` unexpectedly.')

  @parameterized.named_parameters(
      (
          'tensor_type_different_dtype',
          computation_types.TensorType(np.int32),
          computation_types.TensorType(np.bool_),
      ),
      (
          'tensor_type_different_shape',
          computation_types.TensorType(np.int32),
          computation_types.TensorType(np.int32, (10,)),
      ),
  )
  def test_check_equivalent_to_returns_false(self, type_spec, other):
    with self.assertRaises(computation_types.TypesNotEquivalentError):
      type_spec.check_equivalent_to(other)


class TensorTypeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'tensor_type',
          computation_types.TensorType(np.int32),
          computation_types.TensorType(np.int32),
      ),
      (
          'tensor_type_ndims_unknown',
          computation_types.TensorType(np.int32, (None,)),
          computation_types.TensorType(np.int32, (None,)),
      ),
  )
  def test_interned(self, type_spec_1, type_spec_2):
    self.assertIs(type_spec_1, type_spec_2)

  def test_init_infers_shape(self):
    type_spec = computation_types.TensorType(np.int32)
    self.assertEqual(type_spec.shape, ())

  @parameterized.named_parameters(
      (
          'rank_unknown',
          computation_types.TensorType(np.int32),
          'int32',
      ),
      (
          'ndims_unknown',
          computation_types.TensorType(np.int32, (None,)),
          'int32[?]',
      ),
      (
          'ndims_10',
          computation_types.TensorType(np.int32, (10,)),
          'int32[10]',
      ),
  )
  def test_str(self, type_spec, expected_str):
    actual_str = str(type_spec)
    self.assertEqual(actual_str, expected_str)

  @parameterized.named_parameters(
      (
          'rank_unknown',
          computation_types.TensorType(np.int32),
          'TensorType(np.int32)',
      ),
      (
          'ndims_unknown',
          computation_types.TensorType(np.int32, (None,)),
          'TensorType(np.int32, (None,))',
      ),
      (
          'ndims_ten',
          computation_types.TensorType(np.int32, (10,)),
          'TensorType(np.int32, (10,))',
      ),
  )
  def test_repr(self, type_spec, expected_repr):
    actual_repr = repr(type_spec)
    self.assertEqual(actual_repr, expected_repr)

  @parameterized.named_parameters(
      (
          'same_dtype_and_shape',
          computation_types.TensorType(np.int32),
          computation_types.TensorType(np.int32),
          True,
      ),
      (
          'different_dtype',
          computation_types.TensorType(np.int32),
          computation_types.TensorType(np.bool_),
          False,
      ),
      (
          'different_shape',
          computation_types.TensorType(np.int32),
          computation_types.TensorType(np.int32, (10,)),
          False,
      ),
  )
  def test_eq(self, type_spec, other, expected_result):
    actual_result = type_spec == other
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      (
          'same_dtype_and_shape',
          computation_types.TensorType(np.int32),
          computation_types.TensorType(np.int32),
          True,
      ),
      (
          'different_dtype',
          computation_types.TensorType(np.int32),
          computation_types.TensorType(np.bool_),
          False,
      ),
      (
          'different_shape',
          computation_types.TensorType(np.int32),
          computation_types.TensorType(np.int32, (10,)),
          False,
      ),
      (
          'ndims_unknown_from_known',
          computation_types.TensorType(np.int32, (None,)),
          computation_types.TensorType(np.int32, (10,)),
          True,
      ),
      (
          'ndims_known_from_unknown',
          computation_types.TensorType(np.int32, (10,)),
          computation_types.TensorType(np.int32, (None,)),
          False,
      ),
  )
  def test_is_assignable_from(self, type_spec, other, expected_result):
    actual_result = type_spec.is_assignable_from(other)
    self.assertEqual(actual_result, expected_result)


class StructTypeTest(parameterized.TestCase):

  def test_interned(self):
    type_spec_1 = computation_types.StructType([np.int32, np.bool_])
    type_spec_2 = computation_types.StructType([np.int32, np.bool_])
    self.assertIs(type_spec_1, type_spec_2)

  @parameterized.named_parameters(
      (
          'unnamed',
          computation_types.StructType([np.int32, np.bool_]),
          '<int32,bool>',
      ),
      (
          'named',
          computation_types.StructType([('a', np.int32), ('b', np.bool_)]),
          '<a=int32,b=bool>',
      ),
  )
  def test_str(self, type_spec, expected_str):
    actual_str = str(type_spec)
    self.assertEqual(actual_str, expected_str)

  @parameterized.named_parameters(
      (
          'unnamed',
          computation_types.StructType([np.int32, np.bool_]),
          'StructType([TensorType(np.int32), TensorType(np.bool_)])',
      ),
      (
          'named',
          computation_types.StructType([('a', np.int32), ('b', np.bool_)]),
          (
              'StructType(['
              "('a', TensorType(np.int32)), "
              "('b', TensorType(np.bool_))"
              '])'
          ),
      ),
  )
  def test_repr(self, type_spec, expected_repr):
    actual_repr = repr(type_spec)
    self.assertEqual(actual_repr, expected_repr)

  @parameterized.named_parameters(
      (
          'same_elements_unnamed',
          computation_types.StructType([np.int32, np.bool_]),
          computation_types.StructType([np.int32, np.bool_]),
          True,
      ),
      (
          'same_elements_named',
          computation_types.StructType([('a', np.int32), ('b', np.bool_)]),
          computation_types.StructType([('a', np.int32), ('b', np.bool_)]),
          True,
      ),
      (
          'different_elements_unnamed',
          computation_types.StructType([np.int32, np.bool_]),
          computation_types.StructType([np.int32, np.int32]),
          False,
      ),
      (
          'different_elements_named',
          computation_types.StructType([('a', np.int32), ('b', np.bool_)]),
          computation_types.StructType([('a', np.int32), ('b', np.int32)]),
          False,
      ),
      (
          'same_elements_different_names',
          computation_types.StructType([('a', np.int32), ('b', np.bool_)]),
          computation_types.StructType([('a', np.int32), ('c', np.bool_)]),
          False,
      ),
  )
  def test_eq(self, type_spec, other, expected_result):
    actual_result = type_spec == other
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      (
          'same_elements_unnamed',
          computation_types.StructType([np.int32, np.bool_]),
          computation_types.StructType([np.int32, np.bool_]),
          True,
      ),
      (
          'same_elements_named',
          computation_types.StructType([('a', np.int32), ('b', np.bool_)]),
          computation_types.StructType([('a', np.int32), ('b', np.bool_)]),
          True,
      ),
      (
          'different_elements_unnamed',
          computation_types.StructType([np.int32, np.bool_]),
          computation_types.StructType([np.int32, np.int32]),
          False,
      ),
      (
          'different_elements_named',
          computation_types.StructType([('a', np.int32), ('b', np.bool_)]),
          computation_types.StructType([('a', np.int32), ('b', np.int32)]),
          False,
      ),
      (
          'same_elements_different_names',
          computation_types.StructType([('a', np.int32), ('b', np.bool_)]),
          computation_types.StructType([('a', np.int32), ('c', np.bool_)]),
          False,
      ),
      (
          'same_elements_unnamed_from_named',
          computation_types.StructType([np.int32, np.bool_]),
          computation_types.StructType([('a', np.int32), ('b', np.bool_)]),
          False,
      ),
      (
          'same_elements_named_from_unnamed',
          computation_types.StructType([('a', np.int32), ('b', np.bool_)]),
          computation_types.StructType([np.int32, np.bool_]),
          True,
      ),
  )
  def test_is_assignable_from(self, type_spec, other, expected_result):
    actual_result = type_spec.is_assignable_from(other)
    self.assertEqual(actual_result, expected_result)


class StructWithPythonTypeTest(parameterized.TestCase):

  def test_interned(self):
    type_spec_1 = computation_types.StructWithPythonType(
        [np.int32, np.bool_], list
    )
    type_spec_2 = computation_types.StructWithPythonType(
        [np.int32, np.bool_], list
    )
    self.assertIs(type_spec_1, type_spec_2)

  @parameterized.named_parameters(
      (
          'list_unnamed',
          computation_types.StructWithPythonType([np.int32, np.bool_], list),
          '<int32,bool>',
      ),
      (
          'list_named',
          computation_types.StructWithPythonType(
              [('a', np.int32), ('b', np.bool_)], list
          ),
          '<a=int32,b=bool>',
      ),
      (
          'tuple',
          computation_types.StructWithPythonType([np.int32, np.bool_], tuple),
          '<int32,bool>',
      ),
      (
          'dict',
          computation_types.StructWithPythonType(
              [('a', np.int32), ('b', np.bool_)], dict
          ),
          '<a=int32,b=bool>',
      ),
      (
          'ordered_dict',
          computation_types.StructWithPythonType(
              [('a', np.int32), ('b', np.bool_)], collections.OrderedDict
          ),
          '<a=int32,b=bool>',
      ),
      (
          'attrs',
          computation_types.StructWithPythonType(
              [('a', np.int32), ('b', np.bool_)], TestAttrs
          ),
          '<a=int32,b=bool>',
      ),
      (
          'named_tuple',
          computation_types.StructWithPythonType(
              [('a', np.int32), ('b', np.bool_)], TestNamedTuple
          ),
          '<a=int32,b=bool>',
      ),
  )
  def test_str(self, type_spec, expected_str):
    actual_str = str(type_spec)
    self.assertEqual(actual_str, expected_str)

  @parameterized.named_parameters(
      (
          'list_unnamed',
          computation_types.StructWithPythonType([np.int32, np.bool_], list),
          'StructType([TensorType(np.int32), TensorType(np.bool_)]) as list',
      ),
      (
          'list_named',
          computation_types.StructWithPythonType(
              [('a', np.int32), ('b', np.bool_)], list
          ),
          (
              'StructType(['
              "('a', TensorType(np.int32)), "
              "('b', TensorType(np.bool_))"
              ']) as list'
          ),
      ),
      (
          'tuple',
          computation_types.StructWithPythonType([np.int32, np.bool_], tuple),
          'StructType([TensorType(np.int32), TensorType(np.bool_)]) as tuple',
      ),
      (
          'dict',
          computation_types.StructWithPythonType(
              [('a', np.int32), ('b', np.bool_)], dict
          ),
          (
              'StructType(['
              "('a', TensorType(np.int32)), "
              "('b', TensorType(np.bool_))"
              ']) as dict'
          ),
      ),
      (
          'ordered_dict',
          computation_types.StructWithPythonType(
              [('a', np.int32), ('b', np.bool_)], collections.OrderedDict
          ),
          (
              'StructType(['
              "('a', TensorType(np.int32)), "
              "('b', TensorType(np.bool_))"
              ']) as OrderedDict'
          ),
      ),
      (
          'attrs',
          computation_types.StructWithPythonType(
              [('a', np.int32), ('b', np.bool_)], TestAttrs
          ),
          (
              'StructType(['
              "('a', TensorType(np.int32)), "
              "('b', TensorType(np.bool_))"
              ']) as TestAttrs'
          ),
      ),
      (
          'named_tuple',
          computation_types.StructWithPythonType(
              [('a', np.int32), ('b', np.bool_)], TestNamedTuple
          ),
          (
              'StructType(['
              "('a', TensorType(np.int32)), "
              "('b', TensorType(np.bool_))"
              ']) as TestNamedTuple'
          ),
      ),
  )
  def test_repr(self, type_spec, expected_repr):
    actual_repr = repr(type_spec)
    self.assertEqual(actual_repr, expected_repr)

  @parameterized.named_parameters(
      (
          'same_elements_and_container_type_unnamed',
          computation_types.StructWithPythonType([np.int32, np.bool_], list),
          computation_types.StructWithPythonType([np.int32, np.bool_], list),
          True,
      ),
      (
          'same_elements_and_container_type_named',
          computation_types.StructWithPythonType(
              [('a', np.int32), ('b', np.bool_)], list
          ),
          computation_types.StructWithPythonType(
              [('a', np.int32), ('b', np.bool_)], list
          ),
          True,
      ),
      (
          'different_elements',
          computation_types.StructWithPythonType([np.int32, np.bool_], list),
          computation_types.StructWithPythonType([np.int32, np.int32], list),
          False,
      ),
      (
          'different_container_type',
          computation_types.StructWithPythonType([np.int32, np.bool_], list),
          computation_types.StructWithPythonType([np.int32, np.bool_], tuple),
          False,
      ),
      (
          'same_elements_and_container_type_different_names',
          computation_types.StructWithPythonType(
              [('a', np.int32), ('b', np.bool_)], list
          ),
          computation_types.StructWithPythonType(
              [('a', np.int32), ('c', np.bool_)], list
          ),
          False,
      ),
  )
  def test_eq(self, type_spec, other, expected_result):
    actual_result = type_spec == other
    self.assertEqual(actual_result, expected_result)


class SequenceTypeTest(parameterized.TestCase):

  def test_interned(self):
    type_spec_1 = computation_types.SequenceType(np.int32)
    type_spec_2 = computation_types.SequenceType(np.int32)
    self.assertIs(type_spec_1, type_spec_2)

  def test_init_converts_struct_with_list_to_struct_with_tuple_with_list(self):
    type_spec = computation_types.SequenceType(
        computation_types.StructWithPythonType([np.int32, np.bool_], list)
    )
    self.assertIs(type_spec.element.python_container, tuple)

  def test_init_converts_struct_with_list_to_struct_with_tuple_with_list_nested(
      self,
  ):
    type_spec = computation_types.SequenceType(
        computation_types.StructWithPythonType(
            [
                computation_types.StructWithPythonType(
                    [np.int32, np.bool_], list
                ),
                computation_types.StructWithPythonType(
                    [np.int32, np.bool_], list
                ),
            ],
            list,
        )
    )
    self.assertIs(type_spec.element.python_container, tuple)
    first_element, second_element = type_spec.element
    self.assertIs(first_element.python_container, tuple)
    self.assertIs(second_element.python_container, tuple)

  @parameterized.named_parameters([
      ('abstract_type', computation_types.AbstractType('T')),
      ('struct_type', computation_types.StructType([np.int32] * 3)),
      (
          'struct_with_python_type',
          computation_types.StructWithPythonType([np.int32] * 3, list),
      ),
      ('placement_type', computation_types.PlacementType()),
      ('tensor_type', computation_types.TensorType(np.int32)),
  ])
  def test_init_does_not_raise_value_error(self, element):
    try:
      computation_types.SequenceType(element)
    except ValueError:
      self.fail('Raised `ValueError` unexpectedly.')

  @parameterized.named_parameters([
      (
          'federated_type',
          computation_types.FederatedType(np.int32, placements.CLIENTS),
      ),
      (
          'function_type',
          computation_types.FunctionType(np.int32, np.int32),
      ),
      (
          'function_type_nested',
          computation_types.StructType([
              computation_types.FunctionType(np.int32, np.int32),
          ]),
      ),
      (
          'sequence_type',
          computation_types.SequenceType([np.int32]),
      ),
  ])
  def test_init_raises_value_error(self, element):
    with self.assertRaises(ValueError):
      computation_types.SequenceType(element)

  @parameterized.named_parameters(
      (
          'tensor_type',
          computation_types.SequenceType(np.int32),
          'int32*',
      ),
      (
          'struct_type',
          computation_types.SequenceType(
              computation_types.StructType([np.int32, np.bool_])
          ),
          '<int32,bool>*',
      ),
  )
  def test_str(self, type_spec, expected_str):
    actual_str = str(type_spec)
    self.assertEqual(actual_str, expected_str)

  @parameterized.named_parameters(
      (
          'tensor_type',
          computation_types.SequenceType(np.int32),
          'SequenceType(TensorType(np.int32))',
      ),
      (
          'struct_type',
          computation_types.SequenceType(
              computation_types.StructType([np.int32, np.bool_])
          ),
          (
              'SequenceType(StructType([TensorType(np.int32),'
              ' TensorType(np.bool_)]))'
          ),
      ),
  )
  def test_repr(self, type_spec, expected_repr):
    actual_repr = repr(type_spec)
    self.assertEqual(actual_repr, expected_repr)

  @parameterized.named_parameters(
      (
          'same_element',
          computation_types.SequenceType(np.int32),
          computation_types.SequenceType(np.int32),
          True,
      ),
      (
          'different_element',
          computation_types.SequenceType(np.int32),
          computation_types.SequenceType(np.bool_),
          False,
      ),
  )
  def test_eq(self, type_spec, other, expected_result):
    actual_result = type_spec == other
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      (
          'same_element',
          computation_types.SequenceType(np.int32),
          computation_types.SequenceType(np.int32),
          True,
      ),
      (
          'different_element',
          computation_types.SequenceType(np.int32),
          computation_types.SequenceType(np.bool_),
          False,
      ),
  )
  def test_is_assignable_from(self, type_spec, other, expected_result):
    actual_result = type_spec.is_assignable_from(other)
    self.assertEqual(actual_result, expected_result)


class FunctionTypeTest(parameterized.TestCase):

  def test_interned(self):
    type_spec_1 = computation_types.FunctionType(np.int32, np.int32)
    type_spec_2 = computation_types.FunctionType(np.int32, np.int32)
    self.assertIs(type_spec_1, type_spec_2)

  @parameterized.named_parameters(
      (
          'with_parameter',
          computation_types.FunctionType(np.int32, np.bool_),
          '(int32 -> bool)',
      ),
      (
          'without_parameter',
          computation_types.FunctionType(None, np.bool_),
          '( -> bool)',
      ),
  )
  def test_str(self, type_spec, expected_str):
    actual_str = str(type_spec)
    self.assertEqual(actual_str, expected_str)

  @parameterized.named_parameters(
      (
          'with_parameter',
          computation_types.FunctionType(np.int32, np.bool_),
          'FunctionType(TensorType(np.int32), TensorType(np.bool_))',
      ),
      (
          'without_parameter',
          computation_types.FunctionType(None, np.bool_),
          'FunctionType(None, TensorType(np.bool_))',
      ),
  )
  def test_repr(self, type_spec, expected_repr):
    actual_repr = repr(type_spec)
    self.assertEqual(actual_repr, expected_repr)

  @parameterized.named_parameters(
      (
          'same_parameter_and_result',
          computation_types.FunctionType(np.int32, np.bool_),
          computation_types.FunctionType(np.int32, np.bool_),
          True,
      ),
      (
          'different_parameter',
          computation_types.FunctionType(np.int32, np.bool_),
          computation_types.FunctionType(np.bool_, np.bool_),
          False,
      ),
      (
          'different_result',
          computation_types.FunctionType(np.int32, np.bool_),
          computation_types.FunctionType(np.int32, np.int32),
          False,
      ),
  )
  def test_eq(self, type_spec, other, expected_result):
    actual_result = type_spec == other
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      (
          'same_parameter_and_result',
          computation_types.FunctionType(np.int32, np.bool_),
          computation_types.FunctionType(np.int32, np.bool_),
          True,
      ),
      (
          'different_parameter',
          computation_types.FunctionType(np.int32, np.bool_),
          computation_types.FunctionType(np.bool_, np.bool_),
          False,
      ),
      (
          'different_result',
          computation_types.FunctionType(np.int32, np.bool_),
          computation_types.FunctionType(np.int32, np.int32),
          False,
      ),
  )
  def test_is_assignable_from(self, type_spec, other, expected_result):
    actual_result = type_spec.is_assignable_from(other)
    self.assertEqual(actual_result, expected_result)


class AbstractTypeTest(parameterized.TestCase):

  def test_interned(self):
    type_spec_1 = computation_types.AbstractType('T')
    type_spec_2 = computation_types.AbstractType('T')
    self.assertIs(type_spec_1, type_spec_2)

  def test_str(self):
    type_spec = computation_types.AbstractType('T')
    actual_str = str(type_spec)
    self.assertEqual(actual_str, 'T')

  def test_repr(self):
    type_spec = computation_types.AbstractType('T')
    actual_str = repr(type_spec)
    self.assertEqual(actual_str, "AbstractType('T')")

  @parameterized.named_parameters(
      (
          'same_label',
          computation_types.AbstractType('T'),
          computation_types.AbstractType('T'),
          True,
      ),
      (
          'different_label',
          computation_types.AbstractType('T'),
          computation_types.AbstractType('U'),
          False,
      ),
  )
  def test_eq(self, type_spec, other, expected_result):
    actual_result = type_spec == other
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      (
          'same_label',
          computation_types.AbstractType('T'),
          computation_types.AbstractType('T'),
      ),
      (
          'different_label',
          computation_types.AbstractType('T'),
          computation_types.AbstractType('U'),
      ),
  )
  def test_is_assignable_from(self, type_spec, other):
    with self.assertRaises(TypeError):
      type_spec.is_assignable_from(other)


class PlacementTypeTest(parameterized.TestCase):

  def test_interned(self):
    type_spec_1 = computation_types.PlacementType()
    type_spec_2 = computation_types.PlacementType()
    self.assertIs(type_spec_1, type_spec_2)

  def test_str(self):
    type_spec = computation_types.PlacementType()
    actual_str = str(type_spec)
    self.assertEqual(actual_str, 'placement')

  def test_repr(self):
    type_spec = computation_types.PlacementType()
    actual_str = repr(type_spec)
    self.assertEqual(actual_str, 'PlacementType()')

  @parameterized.named_parameters(
      (
          'placement_type',
          computation_types.PlacementType(),
          computation_types.PlacementType(),
          True,
      ),
  )
  def test_eq(self, type_spec, other, expected_result):
    actual_result = type_spec == other
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      (
          'placement_type',
          computation_types.PlacementType(),
          computation_types.PlacementType(),
          True,
      ),
  )
  def test_is_assignable_from(self, type_spec, other, expected_result):
    actual_result = type_spec.is_assignable_from(other)
    self.assertEqual(actual_result, expected_result)


class FederatedTypeTest(parameterized.TestCase):

  def test_interned(self):
    type_spec_1 = computation_types.FederatedType(np.int32, placements.CLIENTS)
    type_spec_2 = computation_types.FederatedType(np.int32, placements.CLIENTS)
    self.assertIs(type_spec_1, type_spec_2)

  @parameterized.named_parameters(
      ('clients', placements.CLIENTS, False),
      ('server', placements.SERVER, True),
  )
  def test_init_infers_all_equal(self, placement, expected_all_equal):
    type_spec = computation_types.FederatedType(np.int32, placement)
    self.assertEqual(type_spec.all_equal, expected_all_equal)

  @parameterized.named_parameters([
      ('abstract_type', computation_types.AbstractType('T')),
      ('placement_type', computation_types.PlacementType()),
      ('sequence_type', computation_types.SequenceType([np.int32])),
      ('struct_type', computation_types.StructType([np.int32] * 3)),
      (
          'struct_with_python_type',
          computation_types.StructWithPythonType([np.int32] * 3, list),
      ),
      ('tensor_type', computation_types.TensorType(np.int32)),
  ])
  def test_init_does_not_raise_value_error(self, member):
    try:
      computation_types.FederatedType(member, placements.CLIENTS)
    except ValueError:
      self.fail('Raised `ValueError` unexpectedly.')

  @parameterized.named_parameters([
      (
          'federated_type',
          computation_types.FederatedType(np.int32, placements.CLIENTS),
      ),
      (
          'function_type',
          computation_types.FunctionType(np.int32, np.int32),
      ),
      (
          'function_type_nested',
          computation_types.StructType([
              computation_types.FunctionType(np.int32, np.int32),
          ]),
      ),
  ])
  def test_init_raises_value_error(self, member):
    with self.assertRaises(ValueError):
      computation_types.FederatedType(member, placements.CLIENTS)

  @parameterized.named_parameters(
      (
          'clients_and_all_equal_true',
          computation_types.FederatedType(np.int32, placements.CLIENTS, True),
          'int32@CLIENTS',
      ),
      (
          'clients_and_all_equal_false',
          computation_types.FederatedType(np.int32, placements.CLIENTS, False),
          '{int32}@CLIENTS',
      ),
      (
          'server_and_all_equal_true',
          computation_types.FederatedType(np.int32, placements.SERVER, True),
          'int32@SERVER',
      ),
      (
          'server_and_all_equal_false',
          computation_types.FederatedType(np.int32, placements.SERVER, False),
          '{int32}@SERVER',
      ),
  )
  def test_str(self, type_spec, expected_str):
    actual_str = str(type_spec)
    self.assertEqual(actual_str, expected_str)

  @parameterized.named_parameters(
      (
          'clients_and_all_equal_true',
          computation_types.FederatedType(np.int32, placements.CLIENTS, True),
          (
              "FederatedType(TensorType(np.int32), PlacementLiteral('clients'),"
              ' True)'
          ),
      ),
      (
          'clients_and_all_equal_false',
          computation_types.FederatedType(np.int32, placements.CLIENTS, False),
          (
              "FederatedType(TensorType(np.int32), PlacementLiteral('clients'),"
              ' False)'
          ),
      ),
      (
          'server_and_all_equal_true',
          computation_types.FederatedType(np.int32, placements.SERVER, True),
          (
              "FederatedType(TensorType(np.int32), PlacementLiteral('server'),"
              ' True)'
          ),
      ),
      (
          'server_and_all_equal_false',
          computation_types.FederatedType(np.int32, placements.SERVER, False),
          (
              "FederatedType(TensorType(np.int32), PlacementLiteral('server'),"
              ' False)'
          ),
      ),
  )
  def test_repr(self, type_spec, expected_repr):
    actual_repr = repr(type_spec)
    self.assertEqual(actual_repr, expected_repr)

  @parameterized.named_parameters(
      (
          'same_member_and_placement_and_all_equal',
          computation_types.FederatedType(np.int32, placements.CLIENTS),
          computation_types.FederatedType(np.int32, placements.CLIENTS),
          True,
      ),
      (
          'different_member',
          computation_types.FederatedType(np.int32, placements.CLIENTS),
          computation_types.FederatedType(np.bool_, placements.CLIENTS),
          False,
      ),
      (
          'different_placement',
          computation_types.FederatedType(np.int32, placements.CLIENTS),
          computation_types.FederatedType(np.int32, placements.SERVER),
          False,
      ),
      (
          'different_all_equals',
          computation_types.FederatedType(np.int32, placements.CLIENTS, True),
          computation_types.FederatedType(np.int32, placements.CLIENTS, False),
          False,
      ),
  )
  def test_eq(self, type_spec, other, expected_result):
    actual_result = type_spec == other
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      (
          'same_member_and_placement_and_all_equal',
          computation_types.FederatedType(np.int32, placements.CLIENTS),
          computation_types.FederatedType(np.int32, placements.CLIENTS),
          True,
      ),
      (
          'different_member',
          computation_types.FederatedType(np.int32, placements.CLIENTS),
          computation_types.FederatedType(np.bool_, placements.CLIENTS),
          False,
      ),
      (
          'different_placement',
          computation_types.FederatedType(np.int32, placements.CLIENTS),
          computation_types.FederatedType(np.int32, placements.SERVER),
          False,
      ),
      (
          'different_all_equals',
          computation_types.FederatedType(np.int32, placements.CLIENTS, True),
          computation_types.FederatedType(np.int32, placements.CLIENTS, False),
          False,
      ),
  )
  def test_is_assignable_from(self, type_spec, other, expected_result):
    actual_result = type_spec.is_assignable_from(other)
    self.assertEqual(actual_result, expected_result)


class ToTypeTest(parameterized.TestCase):

  def test_tensor_type(self):
    s = computation_types.TensorType(np.int32)
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.TensorType)
    self.assertEqual(str(t), 'int32')

  def test_tf_type(self):
    s = np.int32
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.TensorType)
    self.assertEqual(str(t), 'int32')

  def test_tf_type_and_shape(self):
    s = (np.int32, [10])
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.TensorType)
    self.assertEqual(str(t), 'int32[10]')

  def test_tf_type_and_shape_with_unknown_dimension(self):
    s = (np.int32, [None])
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.TensorType)
    self.assertEqual(str(t), 'int32[?]')

  def test_list_of_tf_types(self):
    s = [np.int32, np.bool_]
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertEqual(str(t), '<int32,bool>')

  def test_tuple_of_tf_types(self):
    s = (np.int32, np.bool_)
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, tuple)
    self.assertEqual(str(t), '<int32,bool>')

  def test_singleton_named_tf_type(self):
    s = ('a', np.int32)
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, tuple)
    self.assertEqual(str(t), '<a=int32>')

  def test_list_of_named_tf_types(self):
    s = [('a', np.int32), ('b', np.bool_)]
    t = computation_types.to_type(s)
    # Note: list of pairs should be interpreted as a plain StructType, and
    # not try to convert into a python list afterwards.
    self.assertNotIsInstance(t, computation_types.StructWithPythonType)
    self.assertEqual(str(t), '<a=int32,b=bool>')

  def test_list_of_partially_named_tf_types(self):
    s = [np.bool_, ('a', np.int32)]
    t = computation_types.to_type(s)
    # Note: list of pairs should be interpreted as a plain StructType, and
    # not try to convert into a python list afterwards.
    self.assertNotIsInstance(t, computation_types.StructWithPythonType)
    self.assertEqual(str(t), '<bool,a=int32>')

  def test_ordered_dict_of_tf_types(self):
    s = collections.OrderedDict([('a', np.int32), ('b', np.bool_)])
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, collections.OrderedDict)
    self.assertEqual(str(t), '<a=int32,b=bool>')

  def test_nested_tuple_of_tf_types(self):
    s = (np.int32, (np.float32, np.bool_))
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, tuple)
    self.assertEqual(str(t), '<int32,<float32,bool>>')

  def test_nested_tuple_of_named_tf_types(self):
    s = (np.int32, (('x', np.float32), np.bool_))
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, tuple)
    self.assertNotIsInstance(t[1], computation_types.StructWithPythonType)
    self.assertEqual(str(t), '<int32,<x=float32,bool>>')

  def test_nested_tuple_of_named_nonscalar_tf_types(self):
    s = ((np.int32, [1]), (('x', (np.float32, [2])), (np.bool_, [3])))
    t = computation_types.to_type(s)
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, tuple)
    self.assertNotIsInstance(t[1], computation_types.StructWithPythonType)
    self.assertEqual(str(t), '<int32[1],<x=float32[2],bool[3]>>')

  def test_namedtuple_elements_two_tuples(self):
    elems = [np.int32 for _ in range(10)]
    t = computation_types.to_type(elems)
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, list)
    for k in structure.iter_elements(t):
      self.assertLen(k, 2)

  def test_namedtuples_addressable_by_name(self):
    elems = [('item' + str(k), np.int32) for k in range(5)]
    t = computation_types.to_type(elems)
    # Note: list of pairs should be interpreted as a plain StructType, and
    # not try to convert into a python list afterwards.
    self.assertNotIsInstance(t, computation_types.StructWithPythonType)
    self.assertIsInstance(t.item0, computation_types.TensorType)
    self.assertEqual(t.item0, t[0])

  def test_namedtuple_unpackable(self):
    elems = [('item' + str(k), np.int32) for k in range(2)]
    t = computation_types.to_type(elems)
    a, b = t
    self.assertIsInstance(a, computation_types.TensorType)
    self.assertIsInstance(b, computation_types.TensorType)

  def test_attrs_instance(self):

    @attrs.define
    class TestFoo:
      a: object
      b: object

    t = computation_types.to_type(TestFoo(a=np.int32, b=(np.float32, [2])))
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, TestFoo)
    self.assertEqual(str(t), '<a=int32,b=float32[2]>')

  def test_nested_attrs_class(self):

    @attrs.define
    class TestFoo:
      a: object
      b: object

    @attrs.define
    class TestFoo2:
      c: object

    t = computation_types.to_type(
        TestFoo(a=[np.int32, np.bool_], b=TestFoo2(c=(np.float32, [2])))
    )
    self.assertIsInstance(t, computation_types.StructWithPythonType)
    self.assertIs(t.python_container, TestFoo)
    self.assertIsInstance(t.a, computation_types.StructWithPythonType)
    self.assertIs(t.a.python_container, list)
    self.assertIsInstance(t.b, computation_types.StructWithPythonType)
    self.assertIs(t.b.python_container, TestFoo2)
    self.assertEqual(str(t), '<a=<int32,bool>,b=<c=float32[2]>>')

  def test_struct(self):
    t = computation_types.to_type(
        structure.Struct((
            (None, np.int32),
            ('b', np.int64),
        ))
    )
    self.assertEqual(
        t,
        computation_types.StructType([
            (None, computation_types.TensorType(np.int32)),
            ('b', computation_types.TensorType(np.int64)),
        ]),
    )

  def test_with_np_int32(self):
    t = computation_types.to_type(np.int32)
    self.assertIsInstance(t, computation_types.TensorType)
    self.assertEqual(t.dtype, np.int32)
    self.assertEqual(t.shape, ())

  def test_with_np_int32_in_tensor_spec(self):
    t = computation_types.to_type((np.int32, [5]))
    self.assertIsInstance(t, computation_types.TensorType)
    self.assertEqual(t.dtype, np.int32)
    self.assertEqual(t.shape, (5,))

  def test_with_np_int32_in_dict(self):
    t = computation_types.to_type(collections.OrderedDict([('foo', np.int32)]))
    self.assertIsInstance(t, computation_types.StructType)
    self.assertIsInstance(t.foo, computation_types.TensorType)
    self.assertEqual(t.foo.dtype, np.int32)
    self.assertEqual(t.foo.shape, ())

  @parameterized.named_parameters(
      ('none', None),
      ('object', object()),
  )
  def test_raises_type_error(self, obj):
    with self.assertRaises(TypeError):
      _ = computation_types.to_type(obj)


class TensorflowToTypeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'dtype',
          tf.int32,
          computation_types.TensorType(np.int32),
      ),
      (
          'dtype_nested',
          [tf.int32],
          computation_types.StructWithPythonType(
              [
                  computation_types.TensorType(np.int32),
              ],
              list,
          ),
      ),
      (
          'dtype_mixed',
          [tf.int32, np.float32],
          computation_types.StructWithPythonType(
              [
                  computation_types.TensorType(np.int32),
                  computation_types.TensorType(np.float32),
              ],
              list,
          ),
      ),
      (
          'tensor_like_shape_fully_defined',
          (tf.int32, tf.TensorShape([2, 3])),
          computation_types.TensorType(np.int32, shape=[2, 3]),
      ),
      (
          'tensor_like_shape_partially_defined',
          (tf.int32, tf.TensorShape([2, None])),
          computation_types.TensorType(np.int32, shape=[2, None]),
      ),
      (
          'tensor_like_shape_unknown',
          (tf.int32, tf.TensorShape(None)),
          computation_types.TensorType(np.int32, shape=None),
      ),
      (
          'tensor_like_shape_scalar',
          (tf.int32, tf.TensorShape([])),
          computation_types.TensorType(np.int32),
      ),
      (
          'tensor_like_dtype_only',
          (tf.int32, [2, 3]),
          computation_types.TensorType(np.int32, shape=[2, 3]),
      ),
      (
          'tensor_like_shape_only',
          (np.int32, tf.TensorShape([2, 3])),
          computation_types.TensorType(np.int32, shape=[2, 3]),
      ),
      (
          'tensor_like_nested',
          [(tf.int32, tf.TensorShape([2, 3]))],
          computation_types.StructWithPythonType(
              [
                  computation_types.TensorType(np.int32, shape=[2, 3]),
              ],
              list,
          ),
      ),
      (
          'tensor_like_mixed',
          [(tf.int32, tf.TensorShape([2, 3])), np.float32],
          computation_types.StructWithPythonType(
              [
                  computation_types.TensorType(np.int32, shape=[2, 3]),
                  computation_types.TensorType(np.float32),
              ],
              list,
          ),
      ),
      (
          'tensor_spec',
          tf.TensorSpec(shape=[2, 3], dtype=tf.int32),
          computation_types.TensorType(np.int32, shape=[2, 3]),
      ),
      (
          'tensor_spec_nested',
          [tf.TensorSpec(shape=[2, 3], dtype=tf.int32)],
          computation_types.StructWithPythonType(
              [
                  computation_types.TensorType(np.int32, shape=[2, 3]),
              ],
              list,
          ),
      ),
      (
          'tensor_spec_mixed',
          [tf.TensorSpec(shape=[2, 3], dtype=tf.int32), np.float32],
          computation_types.StructWithPythonType(
              [
                  computation_types.TensorType(np.int32, shape=[2, 3]),
                  computation_types.TensorType(np.float32),
              ],
              list,
          ),
      ),
      (
          'dataset_spec',
          tf.data.DatasetSpec(tf.TensorSpec(shape=[2, 3], dtype=tf.int32)),
          computation_types.SequenceType(
              computation_types.TensorType(np.int32, shape=[2, 3])
          ),
      ),
      (
          'dataset_spec_nested',
          [
              tf.data.DatasetSpec(tf.TensorSpec(shape=[2, 3], dtype=tf.int32)),
          ],
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
          'dataset_spec_mixed',
          [
              tf.data.DatasetSpec(tf.TensorSpec(shape=[2, 3], dtype=tf.int32)),
              np.float32,
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
    actual_result = computation_types.tensorflow_to_type(obj)
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      ('type', computation_types.TensorType(np.int32)),
      ('dtype', np.int32),
      ('tensor_like', (np.int32, [2, 3])),
      ('sequence_unnamed', [np.bool_, np.int32, np.str_]),
      ('sequence_named', [('a', np.bool_), ('b', np.int32), ('c', np.str_)]),
      ('mapping', {'a': np.bool_, 'b': np.int32, 'c': np.str_}),
  )
  def test_delegates_result_with_obj(self, obj):

    with mock.patch.object(
        computation_types, 'to_type', autospec=True, spec_set=True
    ) as mock_to_type:
      computation_types.tensorflow_to_type(obj)
      mock_to_type.assert_called_once_with(obj)


class RepresentationTest(absltest.TestCase):

  def test_returns_string_for_abstract_type(self):
    type_spec = computation_types.AbstractType('T')

    self.assertEqual(type_spec.compact_representation(), 'T')
    self.assertEqual(type_spec.formatted_representation(), 'T')

  def test_returns_string_for_federated_type_clients(self):
    type_spec = computation_types.FederatedType(np.int32, placements.CLIENTS)

    self.assertEqual(type_spec.compact_representation(), '{int32}@CLIENTS')
    self.assertEqual(type_spec.formatted_representation(), '{int32}@CLIENTS')

  def test_returns_string_for_federated_type_server(self):
    type_spec = computation_types.FederatedType(np.int32, placements.SERVER)

    self.assertEqual(type_spec.compact_representation(), 'int32@SERVER')
    self.assertEqual(type_spec.formatted_representation(), 'int32@SERVER')

  def test_returns_string_for_function_type(self):
    type_spec = computation_types.FunctionType(np.int32, np.float32)

    self.assertEqual(type_spec.compact_representation(), '(int32 -> float32)')
    self.assertEqual(type_spec.formatted_representation(), '(int32 -> float32)')

  def test_returns_string_for_function_type_with_named_tuple_type_parameter(
      self,
  ):
    parameter = computation_types.StructType((np.int32, np.float32))
    type_spec = computation_types.FunctionType(parameter, np.bool_)

    self.assertEqual(
        type_spec.compact_representation(), '(<int32,float32> -> bool)'
    )
    # pyformat: disable
    self.assertEqual(
        type_spec.formatted_representation(),
        '(<\n'
        '  int32,\n'
        '  float32\n'
        '> -> bool)'
    )
    # pyformat: enable

  def test_returns_string_for_function_type_with_named_tuple_type_result(self):
    result = computation_types.StructType((np.int32, np.float32))
    type_spec = computation_types.FunctionType(np.bool_, result)

    self.assertEqual(
        type_spec.compact_representation(), '(bool -> <int32,float32>)'
    )
    # pyformat: disable
    self.assertEqual(
        type_spec.formatted_representation(),
        '(bool -> <\n'
        '  int32,\n'
        '  float32\n'
        '>)'
    )
    # pyformat: enable

  def test_returns_string_for_function_type_with_named_tuple_type_parameter_and_result(
      self,
  ):
    parameter = computation_types.StructType((np.int32, np.float32))
    result = computation_types.StructType((np.bool_, np.str_))
    type_spec = computation_types.FunctionType(parameter, result)

    self.assertEqual(
        type_spec.compact_representation(), '(<int32,float32> -> <bool,str>)'
    )
    # pyformat: disable
    self.assertEqual(
        type_spec.formatted_representation(),
        '(<\n'
        '  int32,\n'
        '  float32\n'
        '> -> <\n'
        '  bool,\n'
        '  str\n'
        '>)'
    )
    # pyformat: enable

  def test_returns_string_for_named_tuple_type_unnamed(self):
    type_spec = computation_types.StructType((np.int32, np.float32))

    self.assertEqual(type_spec.compact_representation(), '<int32,float32>')
    # pyformat: disable
    self.assertEqual(
        type_spec.formatted_representation(),
        '<\n'
        '  int32,\n'
        '  float32\n'
        '>'
    )
    # pyformat: enable

  def test_returns_string_for_named_tuple_type_named(self):
    type_spec = computation_types.StructType(
        (('a', np.int32), ('b', np.float32))
    )

    self.assertEqual(type_spec.compact_representation(), '<a=int32,b=float32>')
    # pyformat: disable
    self.assertEqual(
        type_spec.formatted_representation(),
        '<\n'
        '  a=int32,\n'
        '  b=float32\n'
        '>'
    )
    # pyformat: enable

  def test_returns_string_for_named_tuple_type_nested(self):
    type_spec_1 = computation_types.StructType((np.int32, np.float32))
    type_spec_2 = computation_types.StructType((type_spec_1, np.bool_))
    type_spec_3 = computation_types.StructType((type_spec_2, np.str_))
    type_spec = type_spec_3

    self.assertEqual(
        type_spec.compact_representation(), '<<<int32,float32>,bool>,str>'
    )
    # pyformat: disable
    self.assertEqual(
        type_spec.formatted_representation(),
        '<\n'
        '  <\n'
        '    <\n'
        '      int32,\n'
        '      float32\n'
        '    >,\n'
        '    bool\n'
        '  >,\n'
        '  str\n'
        '>'
    )
    # pyformat: enable

  def test_returns_string_for_named_tuple_type_with_one_element(self):
    type_spec = computation_types.StructType((np.int32,))

    self.assertEqual(type_spec.compact_representation(), '<int32>')
    # pyformat: disable
    self.assertEqual(
        type_spec.formatted_representation(),
        '<\n'
        '  int32\n'
        '>'
    )
    # pyformat: enable

  def test_returns_string_for_named_tuple_type_with_no_element(self):
    type_spec = computation_types.StructType([])

    self.assertEqual(type_spec.compact_representation(), '<>')
    self.assertEqual(type_spec.formatted_representation(), '<>')

  def test_returns_string_for_placement_type(self):
    type_spec = computation_types.PlacementType()

    self.assertEqual(type_spec.compact_representation(), 'placement')
    self.assertEqual(type_spec.formatted_representation(), 'placement')

  def test_returns_string_for_sequence_type_int(self):
    type_spec = computation_types.SequenceType(np.int32)

    self.assertEqual(type_spec.compact_representation(), 'int32*')
    self.assertEqual(type_spec.formatted_representation(), 'int32*')

  def test_returns_string_for_sequence_type_float(self):
    type_spec = computation_types.SequenceType(np.float32)

    self.assertEqual(type_spec.compact_representation(), 'float32*')
    self.assertEqual(type_spec.formatted_representation(), 'float32*')

  def test_returns_string_for_sequence_type_named_tuple_type(self):
    element = computation_types.StructType((np.int32, np.float32))
    type_spec = computation_types.SequenceType(element)

    self.assertEqual(type_spec.compact_representation(), '<int32,float32>*')
    # pyformat: disable
    self.assertEqual(
        type_spec.formatted_representation(),
        '<\n'
        '  int32,\n'
        '  float32\n'
        '>*'
    )
    # pyformat: enable

  def test_returns_string_for_tensor_type_int(self):
    type_spec = computation_types.TensorType(np.int32)

    self.assertEqual(type_spec.compact_representation(), 'int32')
    self.assertEqual(type_spec.formatted_representation(), 'int32')

  def test_returns_string_for_tensor_type_float(self):
    type_spec = computation_types.TensorType(np.float32)

    self.assertEqual(type_spec.compact_representation(), 'float32')
    self.assertEqual(type_spec.formatted_representation(), 'float32')


if __name__ == '__main__':
  absltest.main()
