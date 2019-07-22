# Lint as: python3
# Copyright 2019, The TensorFlow Federated Authors.
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
"""Tests for computation_constructing_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import six
from six.moves import range
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_constructing_utils
from tensorflow_federated.python.core.impl import computation_wrapper_instances
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import placement_literals
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl import value_impl


class UniqueNameGeneratorTest(absltest.TestCase):

  def test_does_not_raise_type_error_with_none_comp(self):
    try:
      computation_constructing_utils.unique_name_generator(None)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  def test_returns_unique_names_with_none_comp_and_none_prefix(self):
    name_generator = computation_constructing_utils.unique_name_generator(
        None, prefix=None)
    names = set(six.next(name_generator) for _ in range(10))
    first_name = list(names)[0]
    prefix = first_name[:3]
    self.assertLen(names, 10)
    self.assertTrue(all(n.startswith(prefix) for n in names))

  def test_returns_unique_names_with_none_comp_and_unset_prefix(self):
    name_generator = computation_constructing_utils.unique_name_generator(None)
    names = set(six.next(name_generator) for _ in range(10))
    self.assertLen(names, 10)
    self.assertTrue(all(n.startswith('_var') for n in names))

  def test_returns_unique_names_with_none_comp_and_prefix(self):
    name_generator = computation_constructing_utils.unique_name_generator(
        None, prefix='_test')
    names = set(six.next(name_generator) for _ in range(10))
    self.assertLen(names, 10)
    self.assertTrue(all(n.startswith('_test') for n in names))

  def test_returns_unique_names_with_comp_and_none_prefix(self):
    ref = computation_building_blocks.Reference('a', tf.int32)
    comp = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
    name_generator = computation_constructing_utils.unique_name_generator(
        comp, prefix=None)
    names = set(six.next(name_generator) for _ in range(10))
    first_name = list(names)[0]
    prefix = first_name[:3]
    self.assertLen(names, 10)
    self.assertTrue(all(n.startswith(prefix) for n in names))

  def test_returns_unique_names_with_comp_and_unset_prefix(self):
    ref = computation_building_blocks.Reference('a', tf.int32)
    comp = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
    name_generator = computation_constructing_utils.unique_name_generator(comp)
    names = set(six.next(name_generator) for _ in range(10))
    self.assertLen(names, 10)
    self.assertTrue(all(n.startswith('_var') for n in names))

  def test_returns_unique_names_with_comp_and_prefix(self):
    ref = computation_building_blocks.Reference('a', tf.int32)
    comp = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
    name_generator = computation_constructing_utils.unique_name_generator(
        comp, prefix='_test')
    names = set(six.next(name_generator) for _ in range(10))
    self.assertLen(names, 10)
    self.assertTrue(all(n.startswith('_test') for n in names))

  def test_returns_unique_names_with_conflicting_prefix(self):
    ref = computation_building_blocks.Reference('_test', tf.int32)
    comp = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
    name_generator = computation_constructing_utils.unique_name_generator(
        comp, prefix='_test')
    names = set(six.next(name_generator) for _ in range(10))
    first_name = list(names)[0]
    prefix = first_name[:3]
    self.assertNotEqual(prefix, '_test')
    self.assertTrue(all(n.startswith(prefix) for n in names))


class CreateCompiledEmptyTupleTest(absltest.TestCase):

  def test_constructs_correct_type(self):
    empty_tuple = computation_constructing_utils.create_compiled_empty_tuple()
    self.assertEqual(empty_tuple.type_signature,
                     computation_building_blocks.Tuple([]).type_signature)

  def test_constructs_called_graph(self):
    empty_tuple = computation_constructing_utils.create_compiled_empty_tuple()
    self.assertIsInstance(empty_tuple, computation_building_blocks.Call)
    self.assertIsNone(empty_tuple.argument)
    self.assertIsInstance(empty_tuple.function,
                          computation_building_blocks.CompiledComputation)


class CreateCompiledIdentityTest(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_compiled_identity(None)

  def test_raises_on_federated_type(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_compiled_identity(
          computation_types.FederatedType(tf.int32, placement_literals.SERVER))

  def test_raises_on_federated_type_under_tuple(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_compiled_identity([
          computation_types.FederatedType(tf.int32, placement_literals.SERVER)
      ])

  def test_integer_identity_type_signature(self):
    int_identity = computation_constructing_utils.create_compiled_identity(
        tf.int32)
    self.assertIsInstance(int_identity,
                          computation_building_blocks.CompiledComputation)
    expected_type_signature = computation_types.FunctionType(tf.int32, tf.int32)
    self.assertEqual(int_identity.type_signature, expected_type_signature)

  def test_integer_identity_acts_as_identity(self):
    int_identity = computation_constructing_utils.create_compiled_identity(
        tf.int32)
    executable_identity = computation_wrapper_instances.building_block_to_computation(
        int_identity)
    for k in range(10):
      self.assertEqual(executable_identity(k), k)

  def test_unnamed_tuple_identity_type_signature(self):
    tuple_type = [tf.int32, tf.float32]
    tuple_identity = computation_constructing_utils.create_compiled_identity(
        tuple_type)
    self.assertIsInstance(tuple_identity,
                          computation_building_blocks.CompiledComputation)
    expected_type_signature = computation_types.FunctionType(
        tuple_type, tuple_type)
    self.assertEqual(tuple_identity.type_signature, expected_type_signature)

  def test_unnamed_tuple_identity_acts_as_identity(self):
    tuple_type = [tf.int32, tf.float32]
    tuple_identity = computation_constructing_utils.create_compiled_identity(
        tuple_type)
    executable_identity = computation_wrapper_instances.building_block_to_computation(
        tuple_identity)
    for k in range(10):
      self.assertEqual(executable_identity([k, 10. - k])[0], k)
      self.assertEqual(executable_identity([k, 10. - k])[1], 10. - k)

  def test_named_tuple_identity_type_signature(self):
    tuple_type = [('a', tf.int32), ('b', tf.float32)]
    tuple_identity = computation_constructing_utils.create_compiled_identity(
        tuple_type)
    self.assertIsInstance(tuple_identity,
                          computation_building_blocks.CompiledComputation)
    expected_type_signature = computation_types.FunctionType(
        tuple_type, tuple_type)
    self.assertEqual(tuple_identity.type_signature, expected_type_signature)

  def test_named_tuple_identity_acts_as_identity(self):
    tuple_type = [('a', tf.int32), ('b', tf.float32)]
    tuple_identity = computation_constructing_utils.create_compiled_identity(
        tuple_type)
    executable_identity = computation_wrapper_instances.building_block_to_computation(
        tuple_identity)
    for k in range(10):
      self.assertEqual(executable_identity({'a': k, 'b': 10. - k}).a, k)
      self.assertEqual(executable_identity({'a': k, 'b': 10. - k}).b, 10. - k)

  def test_sequence_identity_type_signature(self):
    sequence_type = computation_types.SequenceType(tf.int32)
    sequence_identity = computation_constructing_utils.create_compiled_identity(
        sequence_type)
    self.assertIsInstance(sequence_identity,
                          computation_building_blocks.CompiledComputation)
    expected_type_signature = computation_types.FunctionType(
        sequence_type, sequence_type)
    self.assertEqual(sequence_identity.type_signature, expected_type_signature)

  def test_sequence_identity_acts_as_identity(self):
    sequence_type = computation_types.SequenceType(tf.int32)
    sequence_identity = computation_constructing_utils.create_compiled_identity(
        sequence_type)
    executable_identity = computation_wrapper_instances.building_block_to_computation(
        sequence_identity)
    seq = list(range(10))
    self.assertEqual(executable_identity(seq), seq)


class CreateCompiledInputReplicationTest(absltest.TestCase):

  def test_raises_on_none_type(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_compiled_input_replication(None, 2)

  def test_raises_on_none_int(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_compiled_input_replication(
          tf.int32, None)

  def test_raises_on_federated_type(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_compiled_input_replication(
          computation_types.FederatedType(tf.int32, placement_literals.SERVER),
          2)

  def test_raises_on_federated_type_under_tuple(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_compiled_input_replication([
          computation_types.FederatedType(tf.int32, placement_literals.SERVER)
      ])

  def test_integer_input_duplicate_type_signature(self):
    int_duplicate_input = computation_constructing_utils.create_compiled_input_replication(
        tf.int32, 2)
    self.assertIsInstance(int_duplicate_input,
                          computation_building_blocks.CompiledComputation)
    expected_type_signature = computation_types.FunctionType(
        tf.int32, [tf.int32, tf.int32])
    self.assertEqual(int_duplicate_input.type_signature,
                     expected_type_signature)

  def test_integer_input_duplicate_duplicates_input(self):
    int_duplicate_input = computation_constructing_utils.create_compiled_input_replication(
        tf.int32, 2)
    executable_duplicate_input = computation_wrapper_instances.building_block_to_computation(
        int_duplicate_input)
    for k in range(10):
      self.assertEqual(executable_duplicate_input(k)[0], k)
      self.assertEqual(executable_duplicate_input(k)[1], k)
      self.assertLen(executable_duplicate_input(k), 2)

  def test_integer_input_triplicate_type_signature(self):
    int_duplicate_input = computation_constructing_utils.create_compiled_input_replication(
        tf.int32, 3)
    self.assertIsInstance(int_duplicate_input,
                          computation_building_blocks.CompiledComputation)
    expected_type_signature = computation_types.FunctionType(
        tf.int32, [tf.int32, tf.int32, tf.int32])
    self.assertEqual(int_duplicate_input.type_signature,
                     expected_type_signature)

  def test_integer_input_triplicate_triplicates_input(self):
    int_duplicate_input = computation_constructing_utils.create_compiled_input_replication(
        tf.int32, 3)
    executable_duplicate_input = computation_wrapper_instances.building_block_to_computation(
        int_duplicate_input)
    for k in range(10):
      self.assertEqual(executable_duplicate_input(k)[0], k)
      self.assertEqual(executable_duplicate_input(k)[1], k)
      self.assertEqual(executable_duplicate_input(k)[2], k)
      self.assertLen(executable_duplicate_input(k), 3)

  def test_unnamed_tuple_input_duplicate_type_signature(self):
    tuple_type = [tf.int32, tf.float32]
    tuple_duplicate_input = computation_constructing_utils.create_compiled_input_replication(
        tuple_type, 2)
    self.assertIsInstance(tuple_duplicate_input,
                          computation_building_blocks.CompiledComputation)
    expected_type_signature = computation_types.FunctionType(
        tuple_type, [tuple_type, tuple_type])
    self.assertEqual(tuple_duplicate_input.type_signature,
                     expected_type_signature)

  def test_unnamed_tuple_input_duplicate_duplicates_input(self):
    tuple_type = [tf.int32, tf.float32]
    tuple_duplicate_input = computation_constructing_utils.create_compiled_input_replication(
        tuple_type, 2)
    executable_duplicate_input = computation_wrapper_instances.building_block_to_computation(
        tuple_duplicate_input)
    for k in range(10):
      self.assertEqual(executable_duplicate_input([k, 10. - k])[0][0], k)
      self.assertEqual(executable_duplicate_input([k, 10. - k])[1][0], k)
      self.assertEqual(executable_duplicate_input([k, 10. - k])[0][1], 10. - k)
      self.assertEqual(executable_duplicate_input([k, 10. - k])[1][1], 10. - k)
      self.assertLen(executable_duplicate_input([k, 10. - k]), 2)

  def test_named_tuple_input_duplicate_type_signature(self):
    tuple_type = [('a', tf.int32), ('b', tf.float32)]
    tuple_duplicate_input = computation_constructing_utils.create_compiled_input_replication(
        tuple_type, 2)
    self.assertIsInstance(tuple_duplicate_input,
                          computation_building_blocks.CompiledComputation)
    expected_type_signature = computation_types.FunctionType(
        tuple_type, [tuple_type, tuple_type])
    self.assertEqual(tuple_duplicate_input.type_signature,
                     expected_type_signature)

  def test_named_tuple_input_duplicate_duplicates_input(self):
    tuple_type = [('a', tf.int32), ('b', tf.float32)]
    tuple_duplicate_input = computation_constructing_utils.create_compiled_input_replication(
        tuple_type, 2)
    executable_duplicate_input = computation_wrapper_instances.building_block_to_computation(
        tuple_duplicate_input)
    for k in range(10):
      self.assertEqual(executable_duplicate_input([k, 10. - k])[0].a, k)
      self.assertEqual(executable_duplicate_input([k, 10. - k])[1].a, k)
      self.assertEqual(executable_duplicate_input([k, 10. - k])[0].b, 10. - k)
      self.assertEqual(executable_duplicate_input([k, 10. - k])[1].b, 10. - k)
      self.assertLen(executable_duplicate_input([k, 10. - k]), 2)

  def test_sequence_input_duplicate_type_signature(self):
    sequence_type = computation_types.SequenceType(tf.int32)
    sequence_duplicate_input = computation_constructing_utils.create_compiled_input_replication(
        sequence_type, 2)
    self.assertIsInstance(sequence_duplicate_input,
                          computation_building_blocks.CompiledComputation)
    expected_type_signature = computation_types.FunctionType(
        sequence_type, [sequence_type, sequence_type])
    self.assertEqual(sequence_duplicate_input.type_signature,
                     expected_type_signature)

  def test_sequence_input_duplicate_duplicates_input(self):
    sequence_type = computation_types.SequenceType(tf.int32)
    sequence_duplicate_input = computation_constructing_utils.create_compiled_input_replication(
        sequence_type, 2)
    executable_duplicate_input = computation_wrapper_instances.building_block_to_computation(
        sequence_duplicate_input)
    seq = list(range(10))
    self.assertEqual(executable_duplicate_input(seq)[0], seq)
    self.assertEqual(executable_duplicate_input(seq)[1], seq)
    self.assertLen(executable_duplicate_input(seq), 2)


class CreateFederatedGetitemCompTest(parameterized.TestCase):

  def test_fails_value(self):
    x = computation_building_blocks.Reference(
        'x', computation_types.to_type([tf.int32]))
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_getitem_comp(
          value_impl.to_value(x), 0)

  @parameterized.named_parameters(('clients', placement_literals.CLIENTS),
                                  ('server', placement_literals.SERVER))
  def test_returns_comp(self, placement):
    federated_value = computation_building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32), ('b', tf.bool)],
                                        placement))
    get_0_comp = computation_constructing_utils.create_federated_getitem_comp(
        federated_value, 0)
    self.assertEqual(str(get_0_comp), '(x -> x[0])')
    get_slice_comp = computation_constructing_utils.create_federated_getitem_comp(
        federated_value, slice(None, None, -1))
    self.assertEqual(str(get_slice_comp), '(x -> <b=x[1],a=x[0]>)')


class CreateFederatedGetattrCompTest(parameterized.TestCase):

  def test_fails_value(self):
    x = computation_building_blocks.Reference(
        'x', computation_types.to_type([('x', tf.int32)]))
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_getattr_comp(
          value_impl.to_value(x), 'x')

  @parameterized.named_parameters(('clients', placement_literals.CLIENTS),
                                  ('server', placement_literals.SERVER))
  def test_returns_comp(self, placement):
    federated_value = computation_building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32), ('b', tf.bool)],
                                        placement))
    get_a_comp = computation_constructing_utils.create_federated_getattr_comp(
        federated_value, 'a')
    self.assertEqual(str(get_a_comp), '(x -> x.a)')
    get_b_comp = computation_constructing_utils.create_federated_getattr_comp(
        federated_value, 'b')
    self.assertEqual(str(get_b_comp), '(x -> x.b)')
    non_federated_arg = computation_building_blocks.Reference(
        'test',
        computation_types.NamedTupleType([('a', tf.int32), ('b', tf.bool)]))
    with self.assertRaises(TypeError):
      _ = computation_constructing_utils.create_federated_getattr_comp(
          non_federated_arg, 'a')
    with self.assertRaisesRegex(ValueError, 'has no element of name c'):
      _ = computation_constructing_utils.create_federated_getattr_comp(
          federated_value, 'c')


class CreateFederatedGetattrCallTest(parameterized.TestCase):

  def test_fails_value(self):
    x = computation_building_blocks.Reference(
        'x', computation_types.to_type([('x', tf.int32)]))
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_getattr_call(
          value_impl.to_value(x), 'x')

  @parameterized.named_parameters(('clients', placement_literals.CLIENTS),
                                  ('server', placement_literals.SERVER))
  def test_returns_named(self, placement):
    federated_comp_named = computation_building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32),
                                         ('b', tf.bool), tf.int32], placement))
    self.assertEqual(
        str(federated_comp_named.type_signature.member),
        '<a=int32,b=bool,int32>')
    name_a = computation_constructing_utils.create_federated_getattr_call(
        federated_comp_named, 'a')
    name_b = computation_constructing_utils.create_federated_getattr_call(
        federated_comp_named, 'b')
    self.assertIsInstance(name_a.type_signature,
                          computation_types.FederatedType)
    self.assertIsInstance(name_b.type_signature,
                          computation_types.FederatedType)
    self.assertEqual(str(name_a.type_signature.member), 'int32')
    self.assertEqual(str(name_b.type_signature.member), 'bool')
    type_utils.check_federated_value_placement(
        value_impl.to_value(name_a, None, context_stack_impl.context_stack),
        placement)
    type_utils.check_federated_value_placement(
        value_impl.to_value(name_b, None, context_stack_impl.context_stack),
        placement)
    with self.assertRaisesRegex(ValueError, 'has no element of name c'):
      _ = computation_constructing_utils.create_federated_getattr_call(
          federated_comp_named, 'c')


class CreateFederatedGetitemCallTest(parameterized.TestCase):

  def test_fails_value(self):
    x = computation_building_blocks.Reference(
        'x', computation_types.to_type([tf.int32]))
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_getitem_call(
          value_impl.to_value(x), 0)

  @parameterized.named_parameters(('clients', placement_literals.CLIENTS),
                                  ('server', placement_literals.SERVER))
  def test_returns_named(self, placement):
    federated_comp_named = computation_building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32), ('b', tf.bool)],
                                        placement))
    self.assertEqual(
        str(federated_comp_named.type_signature.member), '<a=int32,b=bool>')
    idx_0 = computation_constructing_utils.create_federated_getitem_call(
        federated_comp_named, 0)
    idx_1 = computation_constructing_utils.create_federated_getitem_call(
        federated_comp_named, 1)
    self.assertIsInstance(idx_0.type_signature, computation_types.FederatedType)
    self.assertIsInstance(idx_1.type_signature, computation_types.FederatedType)
    self.assertEqual(str(idx_0.type_signature.member), 'int32')
    self.assertEqual(str(idx_1.type_signature.member), 'bool')
    type_utils.check_federated_value_placement(
        value_impl.to_value(idx_0, None, context_stack_impl.context_stack),
        placement)
    type_utils.check_federated_value_placement(
        value_impl.to_value(idx_1, None, context_stack_impl.context_stack),
        placement)
    flipped = computation_constructing_utils.create_federated_getitem_call(
        federated_comp_named, slice(None, None, -1))
    self.assertIsInstance(flipped.type_signature,
                          computation_types.FederatedType)
    self.assertEqual(str(flipped.type_signature.member), '<b=bool,a=int32>')
    type_utils.check_federated_value_placement(
        value_impl.to_value(flipped, None, context_stack_impl.context_stack),
        placement)

  @parameterized.named_parameters(('clients', placement_literals.CLIENTS),
                                  ('server', placement_literals.SERVER))
  def test_returns_unnamed(self, placement):
    federated_comp_unnamed = computation_building_blocks.Reference(
        'test', computation_types.FederatedType([tf.int32, tf.bool], placement))
    self.assertEqual(
        str(federated_comp_unnamed.type_signature.member), '<int32,bool>')
    unnamed_idx_0 = computation_constructing_utils.create_federated_getitem_call(
        federated_comp_unnamed, 0)
    unnamed_idx_1 = computation_constructing_utils.create_federated_getitem_call(
        federated_comp_unnamed, 1)
    self.assertIsInstance(unnamed_idx_0.type_signature,
                          computation_types.FederatedType)
    self.assertIsInstance(unnamed_idx_1.type_signature,
                          computation_types.FederatedType)
    self.assertEqual(str(unnamed_idx_0.type_signature.member), 'int32')
    self.assertEqual(str(unnamed_idx_1.type_signature.member), 'bool')
    type_utils.check_federated_value_placement(
        value_impl.to_value(unnamed_idx_0, None,
                            context_stack_impl.context_stack), placement)
    type_utils.check_federated_value_placement(
        value_impl.to_value(unnamed_idx_1, None,
                            context_stack_impl.context_stack), placement)
    unnamed_flipped = computation_constructing_utils.create_federated_getitem_call(
        federated_comp_unnamed, slice(None, None, -1))
    self.assertIsInstance(unnamed_flipped.type_signature,
                          computation_types.FederatedType)
    self.assertEqual(str(unnamed_flipped.type_signature.member), '<bool,int32>')
    type_utils.check_federated_value_placement(
        value_impl.to_value(unnamed_flipped, None,
                            context_stack_impl.context_stack), placement)


class CreateFederatedSetitemLambdaTest(parameterized.TestCase):

  def test_fails_on_bad_type(self):
    bad_type = computation_types.FederatedType([('a', tf.int32)],
                                               placement_literals.CLIENTS)
    value_comp = computation_building_blocks.Data('x', tf.int32)
    with self.assertRaises(TypeError):
      _ = computation_constructing_utils.create_named_tuple_setattr_lambda(
          bad_type, 'a', value_comp)

  def test_fails_on_none_name(self):
    good_type = computation_types.NamedTupleType([('a', tf.int32)])
    value_comp = computation_building_blocks.Data('x', tf.int32)
    with self.assertRaises(TypeError):
      _ = computation_constructing_utils.create_named_tuple_setattr_lambda(
          good_type, None, value_comp)

  def test_fails_on_none_value(self):
    good_type = computation_types.NamedTupleType([('a', tf.int32)])
    with self.assertRaises(TypeError):
      _ = computation_constructing_utils.create_named_tuple_setattr_lambda(
          good_type, 'a', None)

  def test_fails_implicit_type_conversion(self):
    good_type = computation_types.NamedTupleType([('a', tf.int32),
                                                  ('b', tf.bool)])
    value_comp = computation_building_blocks.Data('x', tf.int32)
    with self.assertRaisesRegex(TypeError, 'incompatible type'):
      _ = computation_constructing_utils.create_named_tuple_setattr_lambda(
          good_type, 'b', value_comp)

  def test_fails_unknown_name(self):
    good_type = computation_types.NamedTupleType([('a', tf.int32),
                                                  ('b', tf.bool)])
    value_comp = computation_building_blocks.Data('x', tf.int32)
    with self.assertRaises(AttributeError):
      _ = computation_constructing_utils.create_named_tuple_setattr_lambda(
          good_type, 'c', value_comp)

  def test_replaces_single_element(self):
    good_type = computation_types.NamedTupleType([('a', tf.int32),
                                                  ('b', tf.bool)])
    value_comp = computation_building_blocks.Data('x', tf.int32)
    lam = computation_constructing_utils.create_named_tuple_setattr_lambda(
        good_type, 'a', value_comp)
    # pyformat: disable
    self.assertEqual(
        lam.formatted_representation(),
        '(let\n'
        '  value_comp_placeholder=x\n'
        ' in (lambda_arg -> <\n'
        '  a=value_comp_placeholder,\n'
        '  b=lambda_arg[1]\n'
        '>))'
    )
    # pyformat: enable

  def test_skips_unnamed_element(self):
    good_type = computation_types.NamedTupleType([('a', tf.int32),
                                                  (None, tf.float32),
                                                  ('b', tf.bool)])
    value_comp = computation_building_blocks.Data('x', tf.int32)
    lam = computation_constructing_utils.create_named_tuple_setattr_lambda(
        good_type, 'a', value_comp)
    # pyformat: disable
    self.assertEqual(
        lam.formatted_representation(),
        '(let\n'
        '  value_comp_placeholder=x\n'
        ' in (lambda_arg -> <\n'
        '  a=value_comp_placeholder,\n'
        '  lambda_arg[1],\n'
        '  b=lambda_arg[2]\n'
        '>))'
    )
    # pyformat: enable

  def test_leaves_type_signature_unchanged(self):
    good_type = computation_types.NamedTupleType([('a', tf.int32),
                                                  (None, tf.float32),
                                                  ('b', tf.bool)])
    value_comp = computation_building_blocks.Data('x', tf.int32)
    lam = computation_constructing_utils.create_named_tuple_setattr_lambda(
        good_type, 'a', value_comp)
    self.assertTrue(
        type_utils.are_equivalent_types(lam.type_signature.parameter,
                                        lam.type_signature.result))


class CreateFederatedSetatterCallTest(parameterized.TestCase):

  def test_fails_on_none_federated_comp(self):
    value_comp = computation_building_blocks.Data('x', tf.int32)
    with self.assertRaises(TypeError):
      _ = computation_constructing_utils.create_federated_setattr_call(
          None, 'a', value_comp)

  def test_fails_non_federated_type(self):
    bad_type = computation_types.NamedTupleType([('a', tf.int32),
                                                 (None, tf.float32),
                                                 ('b', tf.bool)])
    bad_comp = computation_building_blocks.Data('data', bad_type)
    value_comp = computation_building_blocks.Data('x', tf.int32)

    with self.assertRaises(TypeError):
      _ = computation_constructing_utils.create_federated_setattr_call(
          bad_comp, 'a', value_comp)

  def test_fails_on_none_name(self):
    named_tuple_type = computation_types.NamedTupleType([('a', tf.int32),
                                                         (None, tf.float32),
                                                         ('b', tf.bool)])
    good_type = computation_types.FederatedType(named_tuple_type,
                                                placement_literals.CLIENTS)
    acceptable_comp = computation_building_blocks.Data('data', good_type)
    value_comp = computation_building_blocks.Data('x', tf.int32)

    with self.assertRaises(TypeError):
      _ = computation_constructing_utils.create_federated_setattr_call(
          acceptable_comp, None, value_comp)

  def test_fails_on_none_value(self):
    named_tuple_type = computation_types.NamedTupleType([('a', tf.int32),
                                                         (None, tf.float32),
                                                         ('b', tf.bool)])
    good_type = computation_types.FederatedType(named_tuple_type,
                                                placement_literals.CLIENTS)
    acceptable_comp = computation_building_blocks.Data('data', good_type)

    with self.assertRaises(TypeError):
      _ = computation_constructing_utils.create_federated_setattr_call(
          acceptable_comp, 'a', None)

  def test_constructs_correct_intrinsic_clients(self):
    named_tuple_type = computation_types.NamedTupleType([('a', tf.int32),
                                                         (None, tf.float32),
                                                         ('b', tf.bool)])
    good_type = computation_types.FederatedType(named_tuple_type,
                                                placement_literals.CLIENTS)
    federated_comp = computation_building_blocks.Data('federated_comp',
                                                      good_type)
    value_comp = computation_building_blocks.Data('x', tf.int32)

    federated_setattr = computation_constructing_utils.create_federated_setattr_call(
        federated_comp, 'a', value_comp)
    self.assertEqual(federated_setattr.function.uri,
                     intrinsic_defs.FEDERATED_MAP.uri)

  def test_constructs_correct_intrinsic_server(self):
    named_tuple_type = computation_types.NamedTupleType([('a', tf.int32),
                                                         (None, tf.float32),
                                                         ('b', tf.bool)])
    good_type = computation_types.FederatedType(named_tuple_type,
                                                placement_literals.SERVER)
    federated_comp = computation_building_blocks.Data('federated_comp',
                                                      good_type)
    value_comp = computation_building_blocks.Data('x', tf.int32)

    federated_setattr = computation_constructing_utils.create_federated_setattr_call(
        federated_comp, 'a', value_comp)
    self.assertEqual(federated_setattr.function.uri,
                     intrinsic_defs.FEDERATED_APPLY.uri)

  @parameterized.named_parameters(('clients', placement_literals.CLIENTS),
                                  ('server', placement_literals.SERVER))
  def test_leaves_type_signatures_alone(self, placement):
    named_tuple_type = computation_types.NamedTupleType([('a', tf.int32),
                                                         (None, tf.float32),
                                                         ('b', tf.bool)])
    good_type = computation_types.FederatedType(named_tuple_type, placement)
    federated_comp = computation_building_blocks.Data('federated_comp',
                                                      good_type)
    value_comp = computation_building_blocks.Data('x', tf.int32)

    federated_setattr = computation_constructing_utils.create_federated_setattr_call(
        federated_comp, 'a', value_comp)
    self.assertTrue(
        type_utils.are_equivalent_types(federated_setattr.type_signature,
                                        federated_comp.type_signature))

  def test_constructs_correct_computation_clients(self):
    named_tuple_type = computation_types.NamedTupleType([('a', tf.int32),
                                                         (None, tf.float32),
                                                         ('b', tf.bool)])
    good_type = computation_types.FederatedType(named_tuple_type,
                                                placement_literals.CLIENTS)
    federated_comp = computation_building_blocks.Data('federated_comp',
                                                      good_type)
    value_comp = computation_building_blocks.Data('x', tf.int32)

    federated_setattr = computation_constructing_utils.create_federated_setattr_call(
        federated_comp, 'a', value_comp)
    # pyformat: disable
    self.assertEqual(
        federated_setattr.formatted_representation(),
        'federated_map(<\n'
        '  (let\n'
        '    value_comp_placeholder=x\n'
        '   in (lambda_arg -> <\n'
        '    a=value_comp_placeholder,\n'
        '    lambda_arg[1],\n'
        '    b=lambda_arg[2]\n'
        '  >)),\n'
        '  federated_comp\n'
        '>)'
    )
    # pyformat: enable

  def test_constructs_correct_computation_server(self):
    named_tuple_type = computation_types.NamedTupleType([('a', tf.int32),
                                                         (None, tf.float32),
                                                         ('b', tf.bool)])
    good_type = computation_types.FederatedType(named_tuple_type,
                                                placement_literals.SERVER)
    federated_comp = computation_building_blocks.Data('federated_comp',
                                                      good_type)
    value_comp = computation_building_blocks.Data('x', tf.int32)

    federated_setattr = computation_constructing_utils.create_federated_setattr_call(
        federated_comp, 'a', value_comp)
    # pyformat: disable
    self.assertEqual(
        federated_setattr.formatted_representation(),
        'federated_apply(<\n'
        '  (let\n'
        '    value_comp_placeholder=x\n'
        '   in (lambda_arg -> <\n'
        '    a=value_comp_placeholder,\n'
        '    lambda_arg[1],\n'
        '    b=lambda_arg[2]\n'
        '  >)),\n'
        '  federated_comp\n'
        '>)'
    )
    # pyformat: enable


class CreateComputationAppendingTest(absltest.TestCase):

  def test_raises_type_error_with_none_comp1(self):
    comp2 = computation_building_blocks.Data('y', tf.int32)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_computation_appending(None, comp2)

  def test_raises_type_error_with_none_comp2(self):
    value = computation_building_blocks.Data('x', tf.int32)
    comp1 = computation_building_blocks.Tuple([value, value])
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_computation_appending(comp1, None)

  def test_raises_type_error_with_comp1_bad_type(self):
    comp1 = computation_building_blocks.Data('x', tf.int32)
    comp2 = computation_building_blocks.Data('y', tf.int32)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_computation_appending(comp1, comp2)

  def test_returns_comp_unnamed(self):
    value = computation_building_blocks.Data('x', tf.int32)
    comp1 = computation_building_blocks.Tuple([value, value])
    comp2 = computation_building_blocks.Data('y', tf.int32)
    comp = computation_constructing_utils.create_computation_appending(
        comp1, comp2)
    self.assertEqual(
        comp.compact_representation(),
        '(let comps=<<x,x>,y> in <comps[0][0],comps[0][1],comps[1]>)')
    self.assertEqual(str(comp.type_signature), '<int32,int32,int32>')

  def test_returns_comp_named(self):
    value = computation_building_blocks.Data('x', tf.int32)
    comp1 = computation_building_blocks.Tuple((
        ('a', value),
        ('b', value),
    ))
    comp2 = computation_building_blocks.Data('y', tf.int32)
    comp = computation_constructing_utils.create_computation_appending(
        comp1, ('c', comp2))
    self.assertEqual(
        comp.compact_representation(),
        '(let comps=<<a=x,b=x>,y> in <a=comps[0][0],b=comps[0][1],c=comps[1]>)')
    self.assertEqual(str(comp.type_signature), '<a=int32,b=int32,c=int32>')


class CreateFederatedAggregateTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    zero = computation_building_blocks.Data('z', tf.int32)
    accumulate_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    accumulate_result = computation_building_blocks.Data('a', tf.int32)
    accumulate = computation_building_blocks.Lambda('x', accumulate_type,
                                                    accumulate_result)
    merge_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    merge_result = computation_building_blocks.Data('m', tf.int32)
    merge = computation_building_blocks.Lambda('x', merge_type, merge_result)
    report_ref = computation_building_blocks.Reference('r', tf.int32)
    report = computation_building_blocks.Lambda(report_ref.name,
                                                report_ref.type_signature,
                                                report_ref)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_aggregate(
          None, zero, accumulate, merge, report)

  def test_raises_type_error_with_none_zero(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    accumulate_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    accumulate_result = computation_building_blocks.Data('a', tf.int32)
    accumulate = computation_building_blocks.Lambda('x', accumulate_type,
                                                    accumulate_result)
    merge_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    merge_result = computation_building_blocks.Data('m', tf.int32)
    merge = computation_building_blocks.Lambda('x', merge_type, merge_result)
    report_ref = computation_building_blocks.Reference('r', tf.int32)
    report = computation_building_blocks.Lambda(report_ref.name,
                                                report_ref.type_signature,
                                                report_ref)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_aggregate(
          value, None, accumulate, merge, report)

  def test_raises_type_error_with_none_accumulate(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    zero = computation_building_blocks.Data('z', tf.int32)
    merge_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    merge_result = computation_building_blocks.Data('m', tf.int32)
    merge = computation_building_blocks.Lambda('x', merge_type, merge_result)
    report_ref = computation_building_blocks.Reference('r', tf.int32)
    report = computation_building_blocks.Lambda(report_ref.name,
                                                report_ref.type_signature,
                                                report_ref)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_aggregate(
          value, zero, None, merge, report)

  def test_raises_type_error_with_none_merge(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    zero = computation_building_blocks.Data('z', tf.int32)
    accumulate_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    accumulate_result = computation_building_blocks.Data('a', tf.int32)
    accumulate = computation_building_blocks.Lambda('x', accumulate_type,
                                                    accumulate_result)
    report_ref = computation_building_blocks.Reference('r', tf.int32)
    report = computation_building_blocks.Lambda(report_ref.name,
                                                report_ref.type_signature,
                                                report_ref)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_aggregate(
          value, zero, accumulate, None, report)

  def test_raises_type_error_with_none_report(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    zero = computation_building_blocks.Data('z', tf.int32)
    accumulate_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    accumulate_result = computation_building_blocks.Data('a', tf.int32)
    accumulate = computation_building_blocks.Lambda('x', accumulate_type,
                                                    accumulate_result)
    merge_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    merge_result = computation_building_blocks.Data('m', tf.int32)
    merge = computation_building_blocks.Lambda('x', merge_type, merge_result)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_aggregate(
          value, zero, accumulate, merge, None)

  def test_returns_federated_aggregate(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    zero = computation_building_blocks.Data('z', tf.int32)
    accumulate_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    accumulate_result = computation_building_blocks.Data('a', tf.int32)
    accumulate = computation_building_blocks.Lambda('x', accumulate_type,
                                                    accumulate_result)
    merge_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    merge_result = computation_building_blocks.Data('m', tf.int32)
    merge = computation_building_blocks.Lambda('x', merge_type, merge_result)
    report_ref = computation_building_blocks.Reference('r', tf.int32)
    report = computation_building_blocks.Lambda(report_ref.name,
                                                report_ref.type_signature,
                                                report_ref)
    comp = computation_constructing_utils.create_federated_aggregate(
        value, zero, accumulate, merge, report)
    self.assertEqual(comp.compact_representation(),
                     'federated_aggregate(<v,z,(x -> a),(x -> m),(r -> r)>)')
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')


class CreateFederatedApplyTest(absltest.TestCase):

  def test_raises_type_error_with_none_fn(self):
    arg = computation_building_blocks.Data('y', tf.int32)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_apply(None, arg)

  def test_raises_type_error_with_none_arg(self):
    ref = computation_building_blocks.Reference('x', tf.int32)
    fn = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_apply(fn, None)

  def test_returns_federated_apply(self):
    ref = computation_building_blocks.Reference('x', tf.int32)
    fn = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg_type = computation_types.FederatedType(tf.int32, placements.SERVER)
    arg = computation_building_blocks.Data('y', arg_type)
    comp = computation_constructing_utils.create_federated_apply(fn, arg)
    self.assertEqual(comp.compact_representation(),
                     'federated_apply(<(x -> x),y>)')
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')


class CreateFederatedBroadcastTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_broadcast(None)

  def test_returns_federated_broadcast(self):
    value_type = computation_types.FederatedType(tf.int32, placements.SERVER)
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_broadcast(value)
    self.assertEqual(comp.compact_representation(), 'federated_broadcast(v)')
    self.assertEqual(str(comp.type_signature), 'int32@CLIENTS')


class CreateFederatedCollectTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_collect(None)

  def test_returns_federated_collect(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_collect(value)
    self.assertEqual(comp.compact_representation(), 'federated_collect(v)')
    self.assertEqual(str(comp.type_signature), 'int32*@SERVER')


class CreateFederatedMapTest(absltest.TestCase):

  def test_raises_type_error_with_none_fn(self):
    arg = computation_building_blocks.Data('y', tf.int32)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_map(None, arg)

  def test_raises_type_error_with_none_arg(self):
    ref = computation_building_blocks.Reference('x', tf.int32)
    fn = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_map(fn, None)

  def test_returns_federated_map(self):
    ref = computation_building_blocks.Reference('x', tf.int32)
    fn = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = computation_building_blocks.Data('y', arg_type)
    comp = computation_constructing_utils.create_federated_map(fn, arg)
    self.assertEqual(comp.compact_representation(),
                     'federated_map(<(x -> x),y>)')
    self.assertEqual(str(comp.type_signature), '{int32}@CLIENTS')


class CreateFederatedMapAllEqualTest(absltest.TestCase):

  def test_raises_type_error_with_none_fn(self):
    arg = computation_building_blocks.Data('y', tf.int32)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_map_all_equal(None, arg)

  def test_raises_type_error_with_none_arg(self):
    ref = computation_building_blocks.Reference('x', tf.int32)
    fn = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_map_all_equal(fn, None)

  def test_returns_federated_map_all_equal(self):
    ref = computation_building_blocks.Reference('x', tf.int32)
    fn = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg_type = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=True)
    arg = computation_building_blocks.Data('y', arg_type)
    comp = computation_constructing_utils.create_federated_map_all_equal(
        fn, arg)
    self.assertEqual(comp.compact_representation(),
                     'federated_map_all_equal(<(x -> x),y>)')
    self.assertEqual(str(comp.type_signature), 'int32@CLIENTS')


class CreateFederatedMapOrApplyTest(absltest.TestCase):

  def test_raises_type_error_with_none_fn(self):
    arg = computation_building_blocks.Data('y', tf.int32)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_map_or_apply(None, arg)

  def test_raises_type_error_with_none_arg(self):
    ref = computation_building_blocks.Reference('x', tf.int32)
    fn = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_map_or_apply(fn, None)

  def test_returns_federated_apply(self):
    ref = computation_building_blocks.Reference('x', tf.int32)
    fn = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg_type = computation_types.FederatedType(tf.int32, placements.SERVER)
    arg = computation_building_blocks.Data('y', arg_type)
    comp = computation_constructing_utils.create_federated_map_or_apply(fn, arg)
    self.assertEqual(comp.compact_representation(),
                     'federated_apply(<(x -> x),y>)')
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')

  def test_returns_federated_map(self):
    ref = computation_building_blocks.Reference('x', tf.int32)
    fn = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = computation_building_blocks.Data('y', arg_type)
    comp = computation_constructing_utils.create_federated_map_or_apply(fn, arg)
    self.assertEqual(comp.compact_representation(),
                     'federated_map(<(x -> x),y>)')
    self.assertEqual(str(comp.type_signature), '{int32}@CLIENTS')


class CreateFederatedMeanTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_mean(None, None)

  def test_returns_federated_mean(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_mean(value, None)
    self.assertEqual(comp.compact_representation(), 'federated_mean(v)')
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')

  def test_returns_federated_weighted_mean(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    weight_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    weight = computation_building_blocks.Data('w', weight_type)
    comp = computation_constructing_utils.create_federated_mean(value, weight)
    self.assertEqual(comp.compact_representation(),
                     'federated_weighted_mean(<v,w>)')
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')


class CreateFederatedReduceTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    zero = computation_building_blocks.Data('z', tf.int32)
    op_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    op_result = computation_building_blocks.Data('o', tf.int32)
    op = computation_building_blocks.Lambda('x', op_type, op_result)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_reduce(None, zero, op)

  def test_raises_type_error_with_none_zero(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    op_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    op_result = computation_building_blocks.Data('o', tf.int32)
    op = computation_building_blocks.Lambda('x', op_type, op_result)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_reduce(value, None, op)

  def test_raises_type_error_with_none_op(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    zero = computation_building_blocks.Data('z', tf.int32)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_reduce(value, zero, None)

  def test_returns_federated_reduce(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    zero = computation_building_blocks.Data('z', tf.int32)
    op_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    op_result = computation_building_blocks.Data('o', tf.int32)
    op = computation_building_blocks.Lambda('x', op_type, op_result)
    comp = computation_constructing_utils.create_federated_reduce(
        value, zero, op)
    self.assertEqual(comp.compact_representation(),
                     'federated_reduce(<v,z,(x -> o)>)')
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')


class CreateFederatedSumTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_sum(None)

  def test_returns_federated_sum(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_sum(value)
    self.assertEqual(comp.compact_representation(), 'federated_sum(v)')
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')


class CreateFederatedUnzipTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_unzip(None)

  def test_returns_tuple_federated_map_with_empty_value(self):
    value_type = computation_types.FederatedType([], placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    with self.assertRaises(ValueError):
      computation_constructing_utils.create_federated_unzip(value)

  def test_returns_tuple_federated_map_with_one_value_unnamed(self):
    value_type = computation_types.FederatedType((tf.int32,),
                                                 placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_unzip(value)
    self.assertEqual(
        comp.compact_representation(),
        '(let value=v in <federated_map(<(arg -> arg[0]),value>)>)')
    self.assertEqual(str(comp.type_signature), '<{int32}@CLIENTS>')

  def test_returns_tuple_federated_map_with_one_value_named(self):
    type_signature = computation_types.NamedTupleType((('a', tf.int32),))
    value_type = computation_types.FederatedType(type_signature,
                                                 placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_unzip(value)
    self.assertEqual(
        comp.compact_representation(),
        '(let value=v in <a=federated_map(<(arg -> arg[0]),value>)>)')
    self.assertEqual(str(comp.type_signature), '<a={int32}@CLIENTS>')

  def test_returns_tuple_federated_map_with_two_values_unnamed(self):
    value_type = computation_types.FederatedType((tf.int32, tf.int32),
                                                 placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_unzip(value)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        '(let\n'
        '  value=v\n'
        ' in <\n'
        '  federated_map(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> arg[1]),\n'
        '    value\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(
        str(comp.type_signature), '<{int32}@CLIENTS,{int32}@CLIENTS>')

  def test_returns_tuple_federated_map_with_two_values_named(self):
    type_signature = computation_types.NamedTupleType(
        (('a', tf.int32), ('b', tf.int32)))
    value_type = computation_types.FederatedType(type_signature,
                                                 placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_unzip(value)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        '(let\n'
        '  value=v\n'
        ' in <\n'
        '  a=federated_map(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >),\n'
        '  b=federated_map(<\n'
        '    (arg -> arg[1]),\n'
        '    value\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(
        str(comp.type_signature), '<a={int32}@CLIENTS,b={int32}@CLIENTS>')

  def test_returns_tuple_federated_map_with_two_values_different_typed(self):
    value_type = computation_types.FederatedType((tf.int32, tf.bool),
                                                 placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_unzip(value)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        '(let\n'
        '  value=v\n'
        ' in <\n'
        '  federated_map(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> arg[1]),\n'
        '    value\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(
        str(comp.type_signature), '<{int32}@CLIENTS,{bool}@CLIENTS>')

  def test_returns_tuple_federated_apply_with_one_value_unnamed(self):
    value_type = computation_types.FederatedType((tf.int32,), placements.SERVER)
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_unzip(value)
    self.assertEqual(
        comp.compact_representation(),
        '(let value=v in <federated_apply(<(arg -> arg[0]),value>)>)')
    self.assertEqual(str(comp.type_signature), '<int32@SERVER>')

  def test_returns_tuple_federated_apply_with_one_value_named(self):
    type_signature = computation_types.NamedTupleType((('a', tf.int32),))
    value_type = computation_types.FederatedType(type_signature,
                                                 placements.SERVER)
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_unzip(value)
    self.assertEqual(
        comp.compact_representation(),
        '(let value=v in <a=federated_apply(<(arg -> arg[0]),value>)>)')
    self.assertEqual(str(comp.type_signature), '<a=int32@SERVER>')

  def test_returns_tuple_federated_apply_with_two_values_unnamed(self):
    value_type = computation_types.FederatedType((tf.int32, tf.int32),
                                                 placements.SERVER)
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_unzip(value)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        '(let\n'
        '  value=v\n'
        ' in <\n'
        '  federated_apply(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >),\n'
        '  federated_apply(<\n'
        '    (arg -> arg[1]),\n'
        '    value\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(str(comp.type_signature), '<int32@SERVER,int32@SERVER>')

  def test_returns_tuple_federated_apply_with_two_values_named(self):
    type_signature = computation_types.NamedTupleType(
        (('a', tf.int32), ('b', tf.int32)))
    value_type = computation_types.FederatedType(type_signature,
                                                 placements.SERVER)
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_unzip(value)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        '(let\n'
        '  value=v\n'
        ' in <\n'
        '  a=federated_apply(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >),\n'
        '  b=federated_apply(<\n'
        '    (arg -> arg[1]),\n'
        '    value\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(
        str(comp.type_signature), '<a=int32@SERVER,b=int32@SERVER>')

  def test_returns_tuple_federated_apply_with_two_values_different_typed(self):
    value_type = computation_types.FederatedType((tf.int32, tf.bool),
                                                 placements.SERVER)
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_unzip(value)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        '(let\n'
        '  value=v\n'
        ' in <\n'
        '  federated_apply(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >),\n'
        '  federated_apply(<\n'
        '    (arg -> arg[1]),\n'
        '    value\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(str(comp.type_signature), '<int32@SERVER,bool@SERVER>')


class CreateFederatedValueTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_value(
          None, placement_literals.CLIENTS)

  def test_raises_type_error_with_none_placement(self):
    value = computation_building_blocks.Data('v', tf.int32)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_value(value, None)

  def test_raises_type_error_with_unknown_placement(self):
    value = computation_building_blocks.Data('v', tf.int32)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_value(value, 'unknown')

  def test_returns_federated_value_at_clients(self):
    value = computation_building_blocks.Data('v', tf.int32)
    comp = computation_constructing_utils.create_federated_value(
        value, placement_literals.CLIENTS)
    self.assertEqual(comp.compact_representation(),
                     'federated_value_at_clients(v)')
    self.assertEqual(str(comp.type_signature), 'int32@CLIENTS')

  def test_returns_federated_value_at_server(self):
    value = computation_building_blocks.Data('v', tf.int32)
    comp = computation_constructing_utils.create_federated_value(
        value, placement_literals.SERVER)
    self.assertEqual(comp.compact_representation(),
                     'federated_value_at_server(v)')
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')


class CreateFederatedZipTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_federated_zip(None)

  def test_raises_value_error_with_empty_value(self):
    value_type = computation_types.NamedTupleType([])
    value = computation_building_blocks.Data('v', value_type)
    with self.assertRaises(ValueError):
      computation_constructing_utils.create_federated_zip(value)

  def test_returns_federated_map_with_one_value_unnamed(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    value_type = computation_types.NamedTupleType((type_signature,))
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_zip(value)
    self.assertEqual(comp.compact_representation(),
                     'federated_map(<(arg -> <arg>),v[0]>)')
    self.assertEqual(str(comp.type_signature), '{<int32>}@CLIENTS')

  def test_returns_federated_map_with_one_value_unnamed_tuple(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    tup = computation_building_blocks.Tuple((value,))
    comp = computation_constructing_utils.create_federated_zip(tup)
    self.assertEqual(comp.compact_representation(),
                     'federated_map(<(arg -> <arg>),<v>[0]>)')
    self.assertEqual(str(comp.type_signature), '{<int32>}@CLIENTS')

  def test_returns_federated_map_with_one_value_named(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    value_type = computation_types.NamedTupleType((('a', type_signature),))
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_zip(value)
    self.assertEqual(comp.compact_representation(),
                     'federated_map(<(arg -> <a=arg>),v[0]>)')
    self.assertEqual(str(comp.type_signature), '{<a=int32>}@CLIENTS')

  def test_returns_federated_map_with_one_value_named_tuple(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    tup = computation_building_blocks.Tuple((('a', value),))
    comp = computation_constructing_utils.create_federated_zip(tup)
    self.assertEqual(comp.compact_representation(),
                     'federated_map(<(arg -> <a=arg>),<a=v>[0]>)')
    self.assertEqual(str(comp.type_signature), '{<a=int32>}@CLIENTS')

  def test_returns_federated_zip_at_clients_with_two_values_unnamed(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    value_type = computation_types.NamedTupleType(
        (type_signature, type_signature))
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_zip(value)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        'federated_map(<\n'
        '  (x -> <\n'
        '    x[0],\n'
        '    x[1]\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> arg),\n'
        '    (let\n'
        '      value=v\n'
        '     in federated_zip_at_clients(<\n'
        '      value[0],\n'
        '      value[1]\n'
        '    >))\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(str(comp.type_signature), '{<int32,int32>}@CLIENTS')

  def test_returns_federated_zip_at_clients_with_two_values_unnamed_tuple(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    tup = computation_building_blocks.Tuple((value, value))
    comp = computation_constructing_utils.create_federated_zip(tup)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        'federated_map(<\n'
        '  (x -> <\n'
        '    x[0],\n'
        '    x[1]\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> arg),\n'
        '    (let\n'
        '      value=<\n'
        '        v,\n'
        '        v\n'
        '      >\n'
        '     in federated_zip_at_clients(<\n'
        '      value[0],\n'
        '      value[1]\n'
        '    >))\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(str(comp.type_signature), '{<int32,int32>}@CLIENTS')

  def test_returns_federated_zip_at_clients_with_two_values_named(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    value_type = computation_types.NamedTupleType(
        (('a', type_signature), ('b', type_signature)))
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_zip(value)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        'federated_map(<\n'
        '  (x -> <\n'
        '    a=x[0],\n'
        '    b=x[1]\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> arg),\n'
        '    (let\n'
        '      value=v\n'
        '     in federated_zip_at_clients(<\n'
        '      value[0],\n'
        '      value[1]\n'
        '    >))\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(str(comp.type_signature), '{<a=int32,b=int32>}@CLIENTS')

  def test_returns_federated_zip_at_clients_with_two_values_named_tuple(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    tup = computation_building_blocks.Tuple((('a', value), ('b', value)))
    comp = computation_constructing_utils.create_federated_zip(tup)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        'federated_map(<\n'
        '  (x -> <\n'
        '    a=x[0],\n'
        '    b=x[1]\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> arg),\n'
        '    (let\n'
        '      value=<\n'
        '        a=v,\n'
        '        b=v\n'
        '      >\n'
        '     in federated_zip_at_clients(<\n'
        '      value[0],\n'
        '      value[1]\n'
        '    >))\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(str(comp.type_signature), '{<a=int32,b=int32>}@CLIENTS')

  def test_returns_federated_zip_at_clients_with_three_values_unnamed(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    value_type = computation_types.NamedTupleType(
        (type_signature, type_signature, type_signature))
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_zip(value)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        'federated_map(<\n'
        '  (x -> <\n'
        '    x[0],\n'
        '    x[1],\n'
        '    x[2]\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> (let\n'
        '      comps=<\n'
        '        (arg -> arg)(arg[0]),\n'
        '        arg[1]\n'
        '      >\n'
        '     in <\n'
        '      comps[0][0],\n'
        '      comps[0][1],\n'
        '      comps[1]\n'
        '    >)),\n'
        '    (let\n'
        '      value=v\n'
        '     in federated_zip_at_clients(<\n'
        '      federated_zip_at_clients(<\n'
        '        value[0],\n'
        '        value[1]\n'
        '      >),\n'
        '      value[2]\n'
        '    >))\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(str(comp.type_signature), '{<int32,int32,int32>}@CLIENTS')

  def test_returns_federated_zip_at_clients_with_three_values_unnamed_tuple(
      self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    tup = computation_building_blocks.Tuple((value, value, value))
    comp = computation_constructing_utils.create_federated_zip(tup)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        'federated_map(<\n'
        '  (x -> <\n'
        '    x[0],\n'
        '    x[1],\n'
        '    x[2]\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> (let\n'
        '      comps=<\n'
        '        (arg -> arg)(arg[0]),\n'
        '        arg[1]\n'
        '      >\n'
        '     in <\n'
        '      comps[0][0],\n'
        '      comps[0][1],\n'
        '      comps[1]\n'
        '    >)),\n'
        '    (let\n'
        '      value=<\n'
        '        v,\n'
        '        v,\n'
        '        v\n'
        '      >\n'
        '     in federated_zip_at_clients(<\n'
        '      federated_zip_at_clients(<\n'
        '        value[0],\n'
        '        value[1]\n'
        '      >),\n'
        '      value[2]\n'
        '    >))\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(str(comp.type_signature), '{<int32,int32,int32>}@CLIENTS')

  def test_returns_federated_zip_at_clients_with_three_values_named(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    value_type = computation_types.NamedTupleType((
        ('a', type_signature),
        ('b', type_signature),
        ('c', type_signature),
    ))
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_zip(value)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        'federated_map(<\n'
        '  (x -> <\n'
        '    a=x[0],\n'
        '    b=x[1],\n'
        '    c=x[2]\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> (let\n'
        '      comps=<\n'
        '        (arg -> arg)(arg[0]),\n'
        '        arg[1]\n'
        '      >\n'
        '     in <\n'
        '      comps[0][0],\n'
        '      comps[0][1],\n'
        '      comps[1]\n'
        '    >)),\n'
        '    (let\n'
        '      value=v\n'
        '     in federated_zip_at_clients(<\n'
        '      federated_zip_at_clients(<\n'
        '        value[0],\n'
        '        value[1]\n'
        '      >),\n'
        '      value[2]\n'
        '    >))\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(
        str(comp.type_signature), '{<a=int32,b=int32,c=int32>}@CLIENTS')

  def test_returns_federated_zip_at_clients_with_three_values_named_tuple(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = computation_building_blocks.Data('v', value_type)
    tup = computation_building_blocks.Tuple((
        ('a', value),
        ('b', value),
        ('c', value),
    ))
    comp = computation_constructing_utils.create_federated_zip(tup)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        'federated_map(<\n'
        '  (x -> <\n'
        '    a=x[0],\n'
        '    b=x[1],\n'
        '    c=x[2]\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> (let\n'
        '      comps=<\n'
        '        (arg -> arg)(arg[0]),\n'
        '        arg[1]\n'
        '      >\n'
        '     in <\n'
        '      comps[0][0],\n'
        '      comps[0][1],\n'
        '      comps[1]\n'
        '    >)),\n'
        '    (let\n'
        '      value=<\n'
        '        a=v,\n'
        '        b=v,\n'
        '        c=v\n'
        '      >\n'
        '     in federated_zip_at_clients(<\n'
        '      federated_zip_at_clients(<\n'
        '        value[0],\n'
        '        value[1]\n'
        '      >),\n'
        '      value[2]\n'
        '    >))\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(
        str(comp.type_signature), '{<a=int32,b=int32,c=int32>}@CLIENTS')

  def test_returns_federated_zip_at_clients_with_three_values_different_typed(
      self):
    type_signature1 = computation_types.FederatedType(tf.int32,
                                                      placements.CLIENTS)
    type_signature2 = computation_types.FederatedType(tf.float32,
                                                      placements.CLIENTS)
    type_signature3 = computation_types.FederatedType(tf.bool,
                                                      placements.CLIENTS)
    value_type = computation_types.NamedTupleType(
        (type_signature1, type_signature2, type_signature3))
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_zip(value)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        'federated_map(<\n'
        '  (x -> <\n'
        '    x[0],\n'
        '    x[1],\n'
        '    x[2]\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> (let\n'
        '      comps=<\n'
        '        (arg -> arg)(arg[0]),\n'
        '        arg[1]\n'
        '      >\n'
        '     in <\n'
        '      comps[0][0],\n'
        '      comps[0][1],\n'
        '      comps[1]\n'
        '    >)),\n'
        '    (let\n'
        '      value=v\n'
        '     in federated_zip_at_clients(<\n'
        '      federated_zip_at_clients(<\n'
        '        value[0],\n'
        '        value[1]\n'
        '      >),\n'
        '      value[2]\n'
        '    >))\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(str(comp.type_signature), '{<int32,float32,bool>}@CLIENTS')

  def test_returns_federated_zip_at_clients_with_three_values_different_typed_tuple(
      self):
    value_type1 = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value1 = computation_building_blocks.Data('v1', value_type1)
    value_type2 = computation_types.FederatedType(tf.float32,
                                                  placements.CLIENTS)
    value2 = computation_building_blocks.Data('v2', value_type2)
    value_type3 = computation_types.FederatedType(tf.bool, placements.CLIENTS)
    value3 = computation_building_blocks.Data('v3', value_type3)
    tup = computation_building_blocks.Tuple((value1, value2, value3))
    comp = computation_constructing_utils.create_federated_zip(tup)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        'federated_map(<\n'
        '  (x -> <\n'
        '    x[0],\n'
        '    x[1],\n'
        '    x[2]\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> (let\n'
        '      comps=<\n'
        '        (arg -> arg)(arg[0]),\n'
        '        arg[1]\n'
        '      >\n'
        '     in <\n'
        '      comps[0][0],\n'
        '      comps[0][1],\n'
        '      comps[1]\n'
        '    >)),\n'
        '    (let\n'
        '      value=<\n'
        '        v1,\n'
        '        v2,\n'
        '        v3\n'
        '      >\n'
        '     in federated_zip_at_clients(<\n'
        '      federated_zip_at_clients(<\n'
        '        value[0],\n'
        '        value[1]\n'
        '      >),\n'
        '      value[2]\n'
        '    >))\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(str(comp.type_signature), '{<int32,float32,bool>}@CLIENTS')

  def test_returns_federated_apply_with_one_value_unnamed(self):
    value_type = computation_types.FederatedType(tf.int32, placements.SERVER)
    value = computation_building_blocks.Data('v', value_type)
    tup = computation_building_blocks.Tuple((value,))
    comp = computation_constructing_utils.create_federated_zip(tup)
    self.assertEqual(comp.compact_representation(),
                     'federated_apply(<(arg -> <arg>),<v>[0]>)')
    self.assertEqual(str(comp.type_signature), '<int32>@SERVER')

  def test_returns_federated_apply_with_one_value_named(self):
    value_type = computation_types.FederatedType(tf.int32, placements.SERVER)
    value = computation_building_blocks.Data('v', value_type)
    tup = computation_building_blocks.Tuple((('a', value),))
    comp = computation_constructing_utils.create_federated_zip(tup)
    self.assertEqual(comp.compact_representation(),
                     'federated_apply(<(arg -> <a=arg>),<a=v>[0]>)')
    self.assertEqual(str(comp.type_signature), '<a=int32>@SERVER')

  def test_returns_federated_zip_at_server_with_two_values_unnamed(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.SERVER)
    value_type = computation_types.NamedTupleType(
        (type_signature, type_signature))
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_zip(value)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        'federated_apply(<\n'
        '  (x -> <\n'
        '    x[0],\n'
        '    x[1]\n'
        '  >),\n'
        '  federated_apply(<\n'
        '    (arg -> arg),\n'
        '    (let\n'
        '      value=v\n'
        '     in federated_zip_at_server(<\n'
        '      value[0],\n'
        '      value[1]\n'
        '    >))\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(str(comp.type_signature), '<int32,int32>@SERVER')

  def test_returns_federated_zip_at_server_with_two_values_named(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.SERVER)
    value_type = computation_types.NamedTupleType(
        (('a', type_signature), ('b', type_signature)))
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_zip(value)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        'federated_apply(<\n'
        '  (x -> <\n'
        '    a=x[0],\n'
        '    b=x[1]\n'
        '  >),\n'
        '  federated_apply(<\n'
        '    (arg -> arg),\n'
        '    (let\n'
        '      value=v\n'
        '     in federated_zip_at_server(<\n'
        '      value[0],\n'
        '      value[1]\n'
        '    >))\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(str(comp.type_signature), '<a=int32,b=int32>@SERVER')

  def test_returns_federated_zip_at_server_with_three_values_unnamed(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.SERVER)
    value_type = computation_types.NamedTupleType(
        (type_signature, type_signature, type_signature))
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_zip(value)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        'federated_apply(<\n'
        '  (x -> <\n'
        '    x[0],\n'
        '    x[1],\n'
        '    x[2]\n'
        '  >),\n'
        '  federated_apply(<\n'
        '    (arg -> (let\n'
        '      comps=<\n'
        '        (arg -> arg)(arg[0]),\n'
        '        arg[1]\n'
        '      >\n'
        '     in <\n'
        '      comps[0][0],\n'
        '      comps[0][1],\n'
        '      comps[1]\n'
        '    >)),\n'
        '    (let\n'
        '      value=v\n'
        '     in federated_zip_at_server(<\n'
        '      federated_zip_at_server(<\n'
        '        value[0],\n'
        '        value[1]\n'
        '      >),\n'
        '      value[2]\n'
        '    >))\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(str(comp.type_signature), '<int32,int32,int32>@SERVER')

  def test_returns_federated_zip_at_server_with_three_values_named(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.SERVER)
    value_type = computation_types.NamedTupleType((
        ('a', type_signature),
        ('b', type_signature),
        ('c', type_signature),
    ))
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_zip(value)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        'federated_apply(<\n'
        '  (x -> <\n'
        '    a=x[0],\n'
        '    b=x[1],\n'
        '    c=x[2]\n'
        '  >),\n'
        '  federated_apply(<\n'
        '    (arg -> (let\n'
        '      comps=<\n'
        '        (arg -> arg)(arg[0]),\n'
        '        arg[1]\n'
        '      >\n'
        '     in <\n'
        '      comps[0][0],\n'
        '      comps[0][1],\n'
        '      comps[1]\n'
        '    >)),\n'
        '    (let\n'
        '      value=v\n'
        '     in federated_zip_at_server(<\n'
        '      federated_zip_at_server(<\n'
        '        value[0],\n'
        '        value[1]\n'
        '      >),\n'
        '      value[2]\n'
        '    >))\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(
        str(comp.type_signature), '<a=int32,b=int32,c=int32>@SERVER')

  def test_returns_federated_zip_at_server_with_three_values_different_typed(
      self):
    type_signature1 = computation_types.FederatedType(tf.int32,
                                                      placements.SERVER)
    type_signature2 = computation_types.FederatedType(tf.float32,
                                                      placements.SERVER)
    type_signature3 = computation_types.FederatedType(tf.bool,
                                                      placements.SERVER)
    value_type = computation_types.NamedTupleType(
        (type_signature1, type_signature2, type_signature3))
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_federated_zip(value)
    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        'federated_apply(<\n'
        '  (x -> <\n'
        '    x[0],\n'
        '    x[1],\n'
        '    x[2]\n'
        '  >),\n'
        '  federated_apply(<\n'
        '    (arg -> (let\n'
        '      comps=<\n'
        '        (arg -> arg)(arg[0]),\n'
        '        arg[1]\n'
        '      >\n'
        '     in <\n'
        '      comps[0][0],\n'
        '      comps[0][1],\n'
        '      comps[1]\n'
        '    >)),\n'
        '    (let\n'
        '      value=v\n'
        '     in federated_zip_at_server(<\n'
        '      federated_zip_at_server(<\n'
        '        value[0],\n'
        '        value[1]\n'
        '      >),\n'
        '      value[2]\n'
        '    >))\n'
        '  >)\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(str(comp.type_signature), '<int32,float32,bool>@SERVER')


class CreateGenericConstantTest(absltest.TestCase):

  def test_raises_on_none_type(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_generic_constant(None, 0)

  def test_raises_non_scalar(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_generic_constant([tf.int32], [0])

  def test_constructs_tensor_zero(self):
    tensor_type = computation_types.TensorType(tf.float32, [2, 2])
    tensor_zero = computation_constructing_utils.create_generic_constant(
        tensor_type, 0)
    self.assertEqual(tensor_zero.type_signature, tensor_type)
    self.assertIsInstance(tensor_zero, computation_building_blocks.Call)
    executable_noarg_fn = computation_wrapper_instances.building_block_to_computation(
        tensor_zero.function)
    self.assertTrue(np.array_equal(executable_noarg_fn(), np.zeros([2, 2])))

  def test_create_unnamed_tuple_zero(self):
    tuple_type = [computation_types.TensorType(tf.float32, [2, 2])] * 2
    tuple_zero = computation_constructing_utils.create_generic_constant(
        tuple_type, 0)
    self.assertEqual(tuple_zero.type_signature,
                     computation_types.to_type(tuple_type))
    self.assertIsInstance(tuple_zero, computation_building_blocks.Call)
    executable_noarg_fn = computation_wrapper_instances.building_block_to_computation(
        tuple_zero.function)
    self.assertLen(executable_noarg_fn(), 2)
    self.assertTrue(np.array_equal(executable_noarg_fn()[0], np.zeros([2, 2])))
    self.assertTrue(np.array_equal(executable_noarg_fn()[1], np.zeros([2, 2])))

  def test_create_named_tuple_one(self):
    tuple_type = [('a', computation_types.TensorType(tf.float32, [2, 2])),
                  ('b', computation_types.TensorType(tf.float32, [2, 2]))]
    tuple_zero = computation_constructing_utils.create_generic_constant(
        tuple_type, 1)
    self.assertEqual(tuple_zero.type_signature,
                     computation_types.to_type(tuple_type))
    self.assertIsInstance(tuple_zero, computation_building_blocks.Call)
    executable_noarg_fn = computation_wrapper_instances.building_block_to_computation(
        tuple_zero.function)
    self.assertLen(executable_noarg_fn(), 2)
    self.assertTrue(np.array_equal(executable_noarg_fn().a, np.ones([2, 2])))
    self.assertTrue(np.array_equal(executable_noarg_fn().b, np.ones([2, 2])))

  def test_create_federated_tensor_one(self):
    fed_type = computation_types.FederatedType(
        computation_types.TensorType(tf.float32, [2, 2]),
        placement_literals.CLIENTS)
    fed_zero = computation_constructing_utils.create_generic_constant(
        fed_type, 1)
    self.assertEqual(fed_zero.type_signature.member, fed_type.member)
    self.assertEqual(fed_zero.type_signature.placement, fed_type.placement)
    self.assertTrue(fed_zero.type_signature.all_equal)
    self.assertIsInstance(fed_zero, computation_building_blocks.Call)
    self.assertIsInstance(fed_zero.function,
                          computation_building_blocks.Intrinsic)
    self.assertEqual(fed_zero.function.uri,
                     intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri)
    self.assertIsInstance(fed_zero.argument, computation_building_blocks.Call)
    executable_unplaced_fn = computation_wrapper_instances.building_block_to_computation(
        fed_zero.argument.function)
    self.assertTrue(np.array_equal(executable_unplaced_fn(), np.ones([2, 2])))

  def test_create_federated_named_tuple_one(self):
    tuple_type = [('a', computation_types.TensorType(tf.float32, [2, 2])),
                  ('b', computation_types.TensorType(tf.float32, [2, 2]))]
    fed_type = computation_types.FederatedType(tuple_type,
                                               placement_literals.SERVER)
    fed_zero = computation_constructing_utils.create_generic_constant(
        fed_type, 1)
    self.assertEqual(fed_zero.type_signature.member, fed_type.member)
    self.assertEqual(fed_zero.type_signature.placement, fed_type.placement)
    self.assertTrue(fed_zero.type_signature.all_equal)
    self.assertIsInstance(fed_zero, computation_building_blocks.Call)
    self.assertIsInstance(fed_zero.function,
                          computation_building_blocks.Intrinsic)
    self.assertEqual(fed_zero.function.uri,
                     intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri)
    self.assertIsInstance(fed_zero.argument, computation_building_blocks.Call)
    executable_unplaced_fn = computation_wrapper_instances.building_block_to_computation(
        fed_zero.argument.function)
    self.assertLen(executable_unplaced_fn(), 2)
    self.assertTrue(np.array_equal(executable_unplaced_fn().a, np.ones([2, 2])))
    self.assertTrue(np.array_equal(executable_unplaced_fn().b, np.ones([2, 2])))

  def test_create_named_tuple_of_federated_tensors_zero(self):
    fed_type = computation_types.FederatedType(
        computation_types.TensorType(tf.float32, [2, 2]),
        placement_literals.CLIENTS,
        all_equal=True)
    tuple_type = [('a', fed_type), ('b', fed_type)]
    zero = computation_constructing_utils.create_generic_constant(tuple_type, 0)
    fed_zero = zero.argument[0]

    self.assertEqual(zero.type_signature, computation_types.to_type(tuple_type))
    self.assertIsInstance(fed_zero.function,
                          computation_building_blocks.Intrinsic)
    self.assertEqual(fed_zero.function.uri,
                     intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri)
    self.assertIsInstance(fed_zero.argument, computation_building_blocks.Call)
    executable_unplaced_fn = computation_wrapper_instances.building_block_to_computation(
        fed_zero.argument.function)
    self.assertTrue(np.array_equal(executable_unplaced_fn(), np.zeros([2, 2])))


class CreateSequenceMapTest(absltest.TestCase):

  def test_raises_type_error_with_none_fn(self):
    arg_type = computation_types.SequenceType(tf.int32)
    arg = computation_building_blocks.Data('y', arg_type)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_sequence_map(None, arg)

  def test_raises_type_error_with_none_arg(self):
    ref = computation_building_blocks.Reference('x', tf.int32)
    fn = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_sequence_map(fn, None)

  def test_returns_sequence_map(self):
    ref = computation_building_blocks.Reference('x', tf.int32)
    fn = computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg_type = computation_types.SequenceType(tf.int32)
    arg = computation_building_blocks.Data('y', arg_type)
    comp = computation_constructing_utils.create_sequence_map(fn, arg)
    self.assertEqual(comp.compact_representation(),
                     'sequence_map(<(x -> x),y>)')
    self.assertEqual(str(comp.type_signature), 'int32*')


class CreateSequenceReduceTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    zero = computation_building_blocks.Data('z', tf.int32)
    op_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    op_result = computation_building_blocks.Data('o', tf.int32)
    op = computation_building_blocks.Lambda('x', op_type, op_result)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_sequence_reduce(None, zero, op)

  def test_raises_type_error_with_none_zero(self):
    value_type = computation_types.SequenceType(tf.int32)
    value = computation_building_blocks.Data('v', value_type)
    op_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    op_result = computation_building_blocks.Data('o', tf.int32)
    op = computation_building_blocks.Lambda('x', op_type, op_result)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_sequence_reduce(value, None, op)

  def test_raises_type_error_with_none_op(self):
    value_type = computation_types.SequenceType(tf.int32)
    value = computation_building_blocks.Data('v', value_type)
    zero = computation_building_blocks.Data('z', tf.int32)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_sequence_reduce(value, zero, None)

  def test_returns_sequence_reduce(self):
    value_type = computation_types.SequenceType(tf.int32)
    value = computation_building_blocks.Data('v', value_type)
    zero = computation_building_blocks.Data('z', tf.int32)
    op_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    op_result = computation_building_blocks.Data('o', tf.int32)
    op = computation_building_blocks.Lambda('x', op_type, op_result)
    comp = computation_constructing_utils.create_sequence_reduce(
        value, zero, op)
    self.assertEqual(comp.compact_representation(),
                     'sequence_reduce(<v,z,(x -> o)>)')
    self.assertEqual(str(comp.type_signature), 'int32')


class CreateSequenceSumTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_sequence_sum(None)

  def test_returns_federated_sum(self):
    value_type = computation_types.SequenceType(tf.int32)
    value = computation_building_blocks.Data('v', value_type)
    comp = computation_constructing_utils.create_sequence_sum(value)
    self.assertEqual(comp.compact_representation(), 'sequence_sum(v)')
    self.assertEqual(str(comp.type_signature), 'int32')


class CreateNamedFederatedTupleTest(parameterized.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_named_federated_tuple(None, ['a'])

  def test_raises_non_federated_type(self):
    bad_comp = computation_building_blocks.Data(
        'x', computation_types.to_type(tf.int32))
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_named_federated_tuple(
          bad_comp, ['a'])

  def test_raises_federated_non_tuple(self):
    bad_comp = computation_building_blocks.Data(
        'x',
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_named_federated_tuple(
          bad_comp, ['a'])

  def test_raises_on_naked_string(self):
    data_tuple = computation_building_blocks.Data(
        'x',
        computation_types.FederatedType([tf.int32], placement_literals.CLIENTS))
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_named_federated_tuple(
          data_tuple, 'a')

  def test_raises_list_of_ints(self):
    data_tuple = computation_building_blocks.Data(
        'x',
        computation_types.FederatedType([tf.int32], placement_literals.SERVER))
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_named_federated_tuple(
          data_tuple, [1])

  def test_raises_wrong_list_length(self):
    data_tuple = computation_building_blocks.Data(
        'x',
        computation_types.FederatedType([tf.int32], placement_literals.SERVER))
    with self.assertRaises(ValueError):
      computation_constructing_utils.create_named_federated_tuple(
          data_tuple, ['a', 'b'])

  @parameterized.named_parameters(
      ('server', placement_literals.SERVER),
      ('clients', placement_literals.CLIENTS),
  )
  def test_constructs_correct_type_from_unnamed_tuple(self, placement):
    fed_type = computation_types.FederatedType([tf.int32, tf.float32],
                                               placement)
    data_tuple = computation_building_blocks.Data('x', fed_type)
    named_tuple = computation_constructing_utils.create_named_federated_tuple(
        data_tuple, ['a', 'b'])
    expected_result_type = computation_types.FederatedType([('a', tf.int32),
                                                            ('b', tf.float32)],
                                                           placement)
    self.assertEqual(expected_result_type, named_tuple.type_signature)

  @parameterized.named_parameters(
      ('server', placement_literals.SERVER),
      ('clients', placement_literals.CLIENTS),
  )
  def test_constructs_correct_type_from_named_tuple(self, placement):
    fed_type = computation_types.FederatedType([('c', tf.int32),
                                                ('d', tf.float32)], placement)
    data_tuple = computation_building_blocks.Data('x', fed_type)
    named_tuple = computation_constructing_utils.create_named_federated_tuple(
        data_tuple, ['a', 'b'])
    expected_result_type = computation_types.FederatedType([('a', tf.int32),
                                                            ('b', tf.float32)],
                                                           placement)
    self.assertEqual(expected_result_type, named_tuple.type_signature)

  @parameterized.named_parameters(
      ('server', placement_literals.SERVER),
      ('clients', placement_literals.CLIENTS),
  )
  def test_only_names_unnamed_tuple(self, placement):
    ntt = computation_types.FederatedType([tf.int32, tf.float32], placement)
    data_tuple = computation_building_blocks.Data('data', ntt)
    named_tuple = computation_constructing_utils.create_named_federated_tuple(
        data_tuple, ['a', 'b'])
    self.assertRegexMatch(
        named_tuple.compact_representation(),
        [r'federated_(map|apply)\(<\(x -> <a=x\[0\],b=x\[1\]>\),data>\)'])

  @parameterized.named_parameters(
      ('server', placement_literals.SERVER),
      ('clients', placement_literals.CLIENTS),
  )
  def test_only_overwrites_existing_names_in_tuple(self, placement):
    fed_type = computation_types.FederatedType([('c', tf.int32),
                                                ('d', tf.float32)], placement)
    data_tuple = computation_building_blocks.Data('data', fed_type)
    named_tuple = computation_constructing_utils.create_named_federated_tuple(
        data_tuple, ['a', 'b'])
    self.assertRegexMatch(
        named_tuple.compact_representation(),
        [r'federated_(map|apply)\(<\(x -> <a=x\[0\],b=x\[1\]>\),data>\)'])


class CreateNamedTupleTest(absltest.TestCase):

  def test_raises_type_error_with_none_comp(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_named_tuple(None, ('a',))

  def test_raises_type_error_with_wrong_comp_type(self):
    comp = computation_building_blocks.Data('data', tf.int32)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_named_tuple(comp, ('a',))

  def test_raises_type_error_with_wrong_names_type_string(self):
    type_signature = computation_types.NamedTupleType((tf.int32, tf.int32))
    comp = computation_building_blocks.Data('data', type_signature)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_named_tuple(comp, 'a')

  def test_raises_type_error_with_wrong_names_type_ints(self):
    type_signature = computation_types.NamedTupleType((tf.int32, tf.int32))
    comp = computation_building_blocks.Data('data', type_signature)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_named_tuple(comp, 'a')

  def test_raises_value_error_with_wrong_lengths(self):
    type_signature = computation_types.NamedTupleType((tf.int32, tf.int32))
    comp = computation_building_blocks.Data('data', type_signature)
    with self.assertRaises(ValueError):
      computation_constructing_utils.create_named_tuple(comp, ('a',))

  def test_creates_named_tuple_from_unamed_tuple(self):
    type_signature = computation_types.NamedTupleType((tf.int32, tf.int32))
    comp = computation_building_blocks.Data('data', type_signature)
    named_comp = computation_constructing_utils.create_named_tuple(
        comp, ('a', 'b'))
    expected_type_signature = computation_types.NamedTupleType(
        (('a', tf.int32), ('b', tf.int32)))
    self.assertEqual(named_comp.type_signature, expected_type_signature)

  def test_creates_named_tuple_from_named_tuple(self):
    type_signature = computation_types.NamedTupleType(
        (('a', tf.int32), ('b', tf.int32)))
    comp = computation_building_blocks.Data('data', type_signature)
    named_comp = computation_constructing_utils.create_named_tuple(
        comp, ('c', 'd'))
    expected_type_signature = computation_types.NamedTupleType(
        (('c', tf.int32), ('d', tf.int32)))
    self.assertEqual(named_comp.type_signature, expected_type_signature)


class CreateZipTest(absltest.TestCase):

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_zip(None)

  def test_zips_tuple_unnamed(self):
    data_1 = computation_building_blocks.Data('a', tf.int32)
    data_2 = computation_building_blocks.Data('b', tf.float32)
    data_3 = computation_building_blocks.Data('c', tf.bool)
    tup_1 = computation_building_blocks.Tuple((data_1, data_2, data_3))
    tup_2 = computation_building_blocks.Tuple((tup_1, tup_1))
    comp = tup_2
    new_comp = computation_constructing_utils.create_zip(comp)
    self.assertEqual(comp.compact_representation(), '<<a,b,c>,<a,b,c>>')
    # pyformat: disable
    self.assertEqual(
        new_comp.formatted_representation(),
        '(let\n'
        '  _var1=<\n'
        '    <\n'
        '      a,\n'
        '      b,\n'
        '      c\n'
        '    >,\n'
        '    <\n'
        '      a,\n'
        '      b,\n'
        '      c\n'
        '    >\n'
        '  >\n'
        ' in <\n'
        '  <\n'
        '    _var1[0][0],\n'
        '    _var1[1][0]\n'
        '  >,\n'
        '  <\n'
        '    _var1[0][1],\n'
        '    _var1[1][1]\n'
        '  >,\n'
        '  <\n'
        '    _var1[0][2],\n'
        '    _var1[1][2]\n'
        '  >\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(
        str(comp.type_signature), '<<int32,float32,bool>,<int32,float32,bool>>')
    self.assertEqual(
        str(new_comp.type_signature),
        '<<int32,int32>,<float32,float32>,<bool,bool>>')

  def test_zips_tuple_named(self):
    data_1 = computation_building_blocks.Data('a', tf.int32)
    data_2 = computation_building_blocks.Data('b', tf.float32)
    data_3 = computation_building_blocks.Data('c', tf.bool)
    tup_1 = computation_building_blocks.Tuple(
        (('d', data_1), ('e', data_2), ('f', data_3)))
    tup_2 = computation_building_blocks.Tuple((('g', tup_1), ('h', tup_1)))
    comp = tup_2
    new_comp = computation_constructing_utils.create_zip(comp)
    self.assertEqual(comp.compact_representation(),
                     '<g=<d=a,e=b,f=c>,h=<d=a,e=b,f=c>>')
    # pyformat: disable
    self.assertEqual(
        new_comp.formatted_representation(),
        '(let\n'
        '  _var1=<\n'
        '    g=<\n'
        '      d=a,\n'
        '      e=b,\n'
        '      f=c\n'
        '    >,\n'
        '    h=<\n'
        '      d=a,\n'
        '      e=b,\n'
        '      f=c\n'
        '    >\n'
        '  >\n'
        ' in <\n'
        '  <\n'
        '    _var1[0][0],\n'
        '    _var1[1][0]\n'
        '  >,\n'
        '  <\n'
        '    _var1[0][1],\n'
        '    _var1[1][1]\n'
        '  >,\n'
        '  <\n'
        '    _var1[0][2],\n'
        '    _var1[1][2]\n'
        '  >\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(
        str(comp.type_signature),
        '<g=<d=int32,e=float32,f=bool>,h=<d=int32,e=float32,f=bool>>')
    self.assertEqual(
        str(new_comp.type_signature),
        '<<int32,int32>,<float32,float32>,<bool,bool>>')

  def test_zips_reference(self):
    type_signature_1 = computation_types.NamedTupleType(
        [tf.int32, tf.float32, tf.bool])
    type_signature_2 = computation_types.NamedTupleType(
        [type_signature_1, type_signature_1])
    ref = computation_building_blocks.Reference('a', type_signature_2)
    comp = ref
    new_comp = computation_constructing_utils.create_zip(comp)
    self.assertEqual(comp.compact_representation(), 'a')
    # pyformat: disable
    self.assertEqual(
        new_comp.formatted_representation(),
        '<\n'
        '  <\n'
        '    a[0][0],\n'
        '    a[1][0]\n'
        '  >,\n'
        '  <\n'
        '    a[0][1],\n'
        '    a[1][1]\n'
        '  >,\n'
        '  <\n'
        '    a[0][2],\n'
        '    a[1][2]\n'
        '  >\n'
        '>'
    )
    # pyformat: enable
    self.assertEqual(
        str(comp.type_signature), '<<int32,float32,bool>,<int32,float32,bool>>')
    self.assertEqual(
        str(new_comp.type_signature),
        '<<int32,int32>,<float32,float32>,<bool,bool>>')


class CreateTensorFlowBroadcastFunctionTest(absltest.TestCase):

  def test_raises_python_type(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_tensorflow_to_broadcast_scalar(
          int, tf.TensorShape([]))

  def test_raises_list_for_shape(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_tensorflow_to_broadcast_scalar(
          tf.int32, [1, 1])

  def test_raises_partially_defined(self):
    with self.assertRaises(ValueError):
      computation_constructing_utils.create_tensorflow_to_broadcast_scalar(
          tf.int32, tf.TensorShape([None, 1]))

  def test_constructs_identity_scalar_function(self):
    int_identity = computation_constructing_utils.create_tensorflow_to_broadcast_scalar(
        tf.int32, tf.TensorShape([]))
    executable_int_identity = computation_wrapper_instances.building_block_to_computation(
        int_identity)
    for k in range(5):
      self.assertEqual(executable_int_identity(k), k)

  def test_broadcasts_ints_to_nonempty_shape(self):
    int_broadcast = computation_constructing_utils.create_tensorflow_to_broadcast_scalar(
        tf.int32, tf.TensorShape([2, 2]))
    executable_int_broadcast = computation_wrapper_instances.building_block_to_computation(
        int_broadcast)
    for k in range(5):
      self.assertTrue(
          np.array_equal(
              executable_int_broadcast(k), np.array([[k, k], [k, k]])))

  def test_broadcasts_bools_to_nonempty_shape(self):
    int_broadcast = computation_constructing_utils.create_tensorflow_to_broadcast_scalar(
        tf.bool, tf.TensorShape([2, 2]))
    executable_int_broadcast = computation_wrapper_instances.building_block_to_computation(
        int_broadcast)
    self.assertTrue(
        np.array_equal(
            executable_int_broadcast(True),
            np.array([[True, True], [True, True]])))


class CreateTensorFlowBinaryOpTest(absltest.TestCase):

  def test_raises_on_none_type(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_tensorflow_binary_operator(
          None, tf.add)

  def test_raises_non_callable_op(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_tensorflow_binary_operator(
          tf.int32, 1)

  def test_raises_on_federated_type(self):
    fed_type = computation_types.FederatedType(tf.int32,
                                               placement_literals.SERVER)
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_tensorflow_binary_operator(
          fed_type, tf.add)

  def test_raises_on_nested_sequence_type(self):
    hiding_sequence_type = computation_types.NamedTupleType(
        [computation_types.SequenceType(tf.int32)])
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_tensorflow_binary_operator(
          hiding_sequence_type, tf.add)

  def test_divide_integers(self):
    integer_division_func = computation_constructing_utils.create_tensorflow_binary_operator(
        tf.int32, tf.divide)
    self.assertEqual(
        integer_division_func.type_signature,
        computation_types.FunctionType([tf.int32, tf.int32], tf.float64))
    callable_division = computation_wrapper_instances.building_block_to_computation(
        integer_division_func)
    self.assertEqual(callable_division(1, 1), 1)
    self.assertEqual(callable_division(1, 2), 0.5)
    self.assertEqual(callable_division(2, 1), 2)
    self.assertEqual(callable_division(1, 0), np.inf)

  def test_divide_unnamed_tuple(self):
    division_func = computation_constructing_utils.create_tensorflow_binary_operator(
        [tf.int32, tf.float32], tf.divide)
    self.assertEqual(
        division_func.type_signature,
        computation_types.FunctionType(
            [[tf.int32, tf.float32], [tf.int32, tf.float32]],
            [tf.float64, tf.float32]))
    callable_division = computation_wrapper_instances.building_block_to_computation(
        division_func)
    self.assertEqual(callable_division([1, 0.], [1, 1.])[0], 1)
    self.assertEqual(callable_division([1, 0.], [1, 1.])[1], 0.)

  def test_divide_named_tuple(self):
    integer_division_func = computation_constructing_utils.create_tensorflow_binary_operator(
        [('a', tf.int32), ('b', tf.float32)], tf.divide)
    callable_division = computation_wrapper_instances.building_block_to_computation(
        integer_division_func)
    self.assertDictEqual(
        anonymous_tuple.to_odict(callable_division([1, 0.], [1, 1.])), {
            'a': 1,
            'b': 0.
        })

  def test_multiply_integers(self):
    integer_multiplication_func = computation_constructing_utils.create_tensorflow_binary_operator(
        tf.int32, tf.multiply)
    callable_multiplication = computation_wrapper_instances.building_block_to_computation(
        integer_multiplication_func)
    self.assertEqual(callable_multiplication(1, 1), 1)
    self.assertEqual(callable_multiplication(1, 2), 2)
    self.assertEqual(callable_multiplication(2, 1), 2)

  def test_multiply_named_tuple(self):
    integer_multiplication_func = computation_constructing_utils.create_tensorflow_binary_operator(
        [('a', tf.int32), ('b', tf.float32)], tf.multiply)
    callable_multiplication = computation_wrapper_instances.building_block_to_computation(
        integer_multiplication_func)
    self.assertDictEqual(
        anonymous_tuple.to_odict(callable_multiplication([1, 0.], [1, 1.])), {
            'a': 1,
            'b': 0.
        })
    self.assertDictEqual(
        anonymous_tuple.to_odict(callable_multiplication([2, 2.], [1, 1.])), {
            'a': 2,
            'b': 2.
        })

  def test_add_integers(self):
    integer_add = computation_constructing_utils.create_tensorflow_binary_operator(
        tf.int32, tf.add)
    callable_add = computation_wrapper_instances.building_block_to_computation(
        integer_add)
    self.assertEqual(callable_add(0, 0), 0)
    self.assertEqual(callable_add(1, 0), 1)
    self.assertEqual(callable_add(0, 1), 1)
    self.assertEqual(callable_add(1, 1), 2)


class TensorFlowConstantTest(absltest.TestCase):

  def test_raises_on_none_type_spec(self):
    with self.assertRaises(TypeError):
      computation_constructing_utils.create_tensorflow_constant(None, 0)

  def test_raises_type_spec_federated_int(self):
    federated_int = computation_types.FederatedType(tf.int32,
                                                    placement_literals.SERVER)
    with self.assertRaisesRegex(TypeError, 'only nested tuples and tensors'):
      computation_constructing_utils.create_tensorflow_constant(
          federated_int, 0)

  def test_raises_non_scalar_value(self):
    non_scalar_value = np.zeros([1])
    with self.assertRaisesRegex(TypeError, 'Must pass a scalar'):
      computation_constructing_utils.create_tensorflow_constant(
          tf.int32, non_scalar_value)

  def test_raises_float_passed_for_int(self):
    with self.assertRaisesRegex(TypeError, 'Only integers'):
      computation_constructing_utils.create_tensorflow_constant(tf.int32, 1.)

  def test_constructs_integer_tensor_zero(self):
    tensor_zero = computation_constructing_utils.create_tensorflow_constant(
        computation_types.TensorType(tf.int32, [2, 2]), 0)
    self.assertIsInstance(tensor_zero, computation_building_blocks.Call)
    executable_noarg_zero = computation_wrapper_instances.building_block_to_computation(
        tensor_zero.function)
    self.assertTrue(
        np.array_equal(executable_noarg_zero(), np.zeros([2, 2],
                                                         dtype=np.int32)))

  def test_constructs_float_tensor_one(self):
    tensor_one = computation_constructing_utils.create_tensorflow_constant(
        computation_types.TensorType(tf.float32, [2, 2]), 1.)
    self.assertIsInstance(tensor_one, computation_building_blocks.Call)
    executable_noarg_one = computation_wrapper_instances.building_block_to_computation(
        tensor_one.function)
    self.assertTrue(
        np.array_equal(executable_noarg_one(), np.ones([2, 2],
                                                       dtype=np.float32)))

  def test_constructs_unnamed_tuple_of_float_tensor_ones(self):
    tuple_type = computation_types.NamedTupleType(
        [computation_types.TensorType(tf.float32, [2, 2])] * 2)
    tuple_of_ones = computation_constructing_utils.create_tensorflow_constant(
        tuple_type, 1.)
    self.assertEqual(tuple_of_ones.type_signature, tuple_type)
    self.assertIsInstance(tuple_of_ones, computation_building_blocks.Call)
    executable_noarg_one = computation_wrapper_instances.building_block_to_computation(
        tuple_of_ones.function)
    self.assertTrue(
        np.array_equal(executable_noarg_one()[0],
                       np.ones([2, 2], dtype=np.float32)))
    self.assertTrue(
        np.array_equal(executable_noarg_one()[1],
                       np.ones([2, 2], dtype=np.float32)))

  def test_constructs_named_tuple_of_float_tensor_ones(self):
    tuple_type = computation_types.NamedTupleType([
        ('a', computation_types.TensorType(tf.float32, [2, 2])),
        ('b', computation_types.TensorType(tf.float32, [2, 2]))
    ])
    tuple_of_ones = computation_constructing_utils.create_tensorflow_constant(
        tuple_type, 1.)
    self.assertEqual(tuple_of_ones.type_signature, tuple_type)
    self.assertIsInstance(tuple_of_ones, computation_building_blocks.Call)
    executable_noarg_one = computation_wrapper_instances.building_block_to_computation(
        tuple_of_ones.function)
    self.assertTrue(
        np.array_equal(executable_noarg_one().a,
                       np.ones([2, 2], dtype=np.float32)))
    self.assertTrue(
        np.array_equal(executable_noarg_one().b,
                       np.ones([2, 2], dtype=np.float32)))

  def test_constructs_nested_named_tuple_of_float_tensor_ones(self):
    tuple_type = computation_types.NamedTupleType([[
        ('a', computation_types.TensorType(tf.float32, [2, 2])),
        ('b', computation_types.TensorType(tf.float32, [2, 2]))
    ]])
    tuple_of_ones = computation_constructing_utils.create_tensorflow_constant(
        tuple_type, 1.)
    self.assertEqual(tuple_of_ones.type_signature, tuple_type)
    self.assertIsInstance(tuple_of_ones, computation_building_blocks.Call)
    executable_noarg_one = computation_wrapper_instances.building_block_to_computation(
        tuple_of_ones.function)
    self.assertTrue(
        np.array_equal(executable_noarg_one()[0].a,
                       np.ones([2, 2], dtype=np.float32)))
    self.assertTrue(
        np.array_equal(executable_noarg_one()[0].b,
                       np.ones([2, 2], dtype=np.float32)))

  def test_constructs_nested_named_tuple_of_int_and_float_tensor_ones(self):
    tuple_type = computation_types.NamedTupleType([[
        ('a', computation_types.TensorType(tf.int32, [2, 2])),
        ('b', computation_types.TensorType(tf.float32, [2, 2]))
    ]])
    tuple_of_ones = computation_constructing_utils.create_tensorflow_constant(
        tuple_type, 1)
    self.assertEqual(tuple_of_ones.type_signature, tuple_type)
    self.assertIsInstance(tuple_of_ones, computation_building_blocks.Call)
    executable_noarg_zero = computation_wrapper_instances.building_block_to_computation(
        tuple_of_ones.function)
    self.assertTrue(
        np.array_equal(executable_noarg_zero()[0].a,
                       np.ones([2, 2], dtype=np.int32)))
    self.assertTrue(
        np.array_equal(executable_noarg_zero()[0].b,
                       np.ones([2, 2], dtype=np.float32)))


if __name__ == '__main__':
  absltest.main()
