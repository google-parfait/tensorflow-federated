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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import anonymous_tuple
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import placement_literals
from tensorflow_federated.python.core.impl.compiler import test_utils
from tensorflow_federated.python.core.impl.compiler import type_factory


class UniqueNameGeneratorTest(absltest.TestCase):

  def test_does_not_raise_type_error_with_none_comp(self):
    try:
      building_block_factory.unique_name_generator(None)
    except TypeError:
      self.fail('Raised TypeError unexpectedly.')

  def test_returns_unique_names_with_none_comp_and_none_prefix(self):
    name_generator = building_block_factory.unique_name_generator(
        None, prefix=None)
    names = set(next(name_generator) for _ in range(10))
    first_name = list(names)[0]
    prefix = first_name[:3]
    self.assertLen(names, 10)
    self.assertTrue(all(n.startswith(prefix) for n in names))

  def test_returns_unique_names_with_none_comp_and_unset_prefix(self):
    name_generator = building_block_factory.unique_name_generator(None)
    names = set(next(name_generator) for _ in range(10))
    self.assertLen(names, 10)
    self.assertTrue(all(n.startswith('_var') for n in names))

  def test_returns_unique_names_with_none_comp_and_prefix(self):
    name_generator = building_block_factory.unique_name_generator(
        None, prefix='_test')
    names = set(next(name_generator) for _ in range(10))
    self.assertLen(names, 10)
    self.assertTrue(all(n.startswith('_test') for n in names))

  def test_returns_unique_names_with_comp_and_none_prefix(self):
    ref = building_blocks.Reference('a', tf.int32)
    comp = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    name_generator = building_block_factory.unique_name_generator(
        comp, prefix=None)
    names = set(next(name_generator) for _ in range(10))
    first_name = list(names)[0]
    prefix = first_name[:3]
    self.assertLen(names, 10)
    self.assertTrue(all(n.startswith(prefix) for n in names))

  def test_returns_unique_names_with_comp_and_unset_prefix(self):
    ref = building_blocks.Reference('a', tf.int32)
    comp = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    name_generator = building_block_factory.unique_name_generator(comp)
    names = set(next(name_generator) for _ in range(10))
    self.assertLen(names, 10)
    self.assertTrue(all(n.startswith('_var') for n in names))

  def test_returns_unique_names_with_comp_and_prefix(self):
    ref = building_blocks.Reference('a', tf.int32)
    comp = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    name_generator = building_block_factory.unique_name_generator(
        comp, prefix='_test')
    names = set(next(name_generator) for _ in range(10))
    self.assertLen(names, 10)
    self.assertTrue(all(n.startswith('_test') for n in names))

  def test_returns_unique_names_with_conflicting_prefix(self):
    ref = building_blocks.Reference('_test', tf.int32)
    comp = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    name_generator = building_block_factory.unique_name_generator(
        comp, prefix='_test')
    names = set(next(name_generator) for _ in range(10))
    first_name = list(names)[0]
    prefix = first_name[:3]
    self.assertNotEqual(prefix, '_test')
    self.assertTrue(all(n.startswith(prefix) for n in names))


class CreateCompiledEmptyTupleTest(absltest.TestCase):

  def test_constructs_correct_type(self):
    empty_tuple = building_block_factory.create_compiled_empty_tuple()
    self.assertEqual(empty_tuple.type_signature,
                     building_blocks.Tuple([]).type_signature)

  def test_constructs_called_graph(self):
    empty_tuple = building_block_factory.create_compiled_empty_tuple()
    self.assertIsInstance(empty_tuple, building_blocks.Call)
    self.assertIsNone(empty_tuple.argument)
    self.assertIsInstance(empty_tuple.function,
                          building_blocks.CompiledComputation)


class CreateCompiledIdentityTest(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_compiled_identity(None)

  def test_raises_on_federated_type(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_compiled_identity(
          computation_types.FederatedType(tf.int32, placement_literals.SERVER))

  def test_raises_on_federated_type_under_tuple(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_compiled_identity([
          computation_types.FederatedType(tf.int32, placement_literals.SERVER)
      ])

  def test_integer_identity_type_signature(self):
    int_identity = building_block_factory.create_compiled_identity(tf.int32)
    self.assertIsInstance(int_identity, building_blocks.CompiledComputation)
    expected_type_signature = computation_types.FunctionType(tf.int32, tf.int32)
    self.assertEqual(int_identity.type_signature, expected_type_signature)

  def test_integer_identity_acts_as_identity(self):
    int_identity = building_block_factory.create_compiled_identity(tf.int32)
    for k in range(10):
      result = test_utils.run_tensorflow(int_identity.proto, k)
      self.assertEqual(result, k)

  def test_unnamed_tuple_identity_type_signature(self):
    tuple_type = [tf.int32, tf.float32]
    tuple_identity = building_block_factory.create_compiled_identity(tuple_type)
    self.assertIsInstance(tuple_identity, building_blocks.CompiledComputation)
    expected_type_signature = computation_types.FunctionType(
        tuple_type, tuple_type)
    self.assertEqual(tuple_identity.type_signature, expected_type_signature)

  def test_unnamed_tuple_identity_acts_as_identity(self):
    tuple_type = [tf.int32, tf.float32]
    tuple_identity = building_block_factory.create_compiled_identity(tuple_type)
    for k in range(10):
      result = test_utils.run_tensorflow(tuple_identity.proto, [k, 10. - k])
      self.assertLen(result, 2)
      self.assertEqual(result[0], k)
      self.assertEqual(result[1], 10. - k)

  def test_named_tuple_identity_type_signature(self):
    tuple_type = [('a', tf.int32), ('b', tf.float32)]
    tuple_identity = building_block_factory.create_compiled_identity(tuple_type)
    self.assertIsInstance(tuple_identity, building_blocks.CompiledComputation)
    expected_type_signature = computation_types.FunctionType(
        tuple_type, tuple_type)
    self.assertEqual(tuple_identity.type_signature, expected_type_signature)

  def test_named_tuple_identity_acts_as_identity(self):
    tuple_type = [('a', tf.int32), ('b', tf.float32)]
    tuple_identity = building_block_factory.create_compiled_identity(tuple_type)
    for k in range(10):
      result = test_utils.run_tensorflow(tuple_identity.proto, {
          'a': k,
          'b': 10. - k
      })
      self.assertLen(result, 2)
      self.assertEqual(result.a, k)
      self.assertEqual(result.b, 10. - k)

  def test_sequence_identity_type_signature(self):
    sequence_type = computation_types.SequenceType(tf.int32)
    sequence_identity = building_block_factory.create_compiled_identity(
        sequence_type)
    self.assertIsInstance(sequence_identity,
                          building_blocks.CompiledComputation)
    expected_type_signature = computation_types.FunctionType(
        sequence_type, sequence_type)
    self.assertEqual(sequence_identity.type_signature, expected_type_signature)

  def test_sequence_identity_acts_as_identity(self):
    sequence_type = computation_types.SequenceType(tf.int32)
    sequence_identity = building_block_factory.create_compiled_identity(
        sequence_type)
    seq = list(range(10))
    result = test_utils.run_tensorflow(sequence_identity.proto, seq)
    self.assertEqual(result, seq)


class CreateCompiledInputReplicationTest(absltest.TestCase):

  def test_raises_on_none_type(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_compiled_input_replication(None, 2)

  def test_raises_on_none_int(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_compiled_input_replication(tf.int32, None)

  def test_raises_on_federated_type(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_compiled_input_replication(
          computation_types.FederatedType(tf.int32, placement_literals.SERVER),
          2)

  def test_raises_on_federated_type_under_tuple(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_compiled_input_replication([
          computation_types.FederatedType(tf.int32, placement_literals.SERVER)
      ])

  def test_integer_input_duplicate_type_signature(self):
    int_duplicate_input = building_block_factory.create_compiled_input_replication(
        tf.int32, 2)
    self.assertIsInstance(int_duplicate_input,
                          building_blocks.CompiledComputation)
    expected_type_signature = computation_types.FunctionType(
        tf.int32, [tf.int32, tf.int32])
    self.assertEqual(int_duplicate_input.type_signature,
                     expected_type_signature)

  def test_integer_input_duplicate_duplicates_input(self):
    int_duplicate_input = building_block_factory.create_compiled_input_replication(
        tf.int32, 2)
    for k in range(10):
      result = test_utils.run_tensorflow(int_duplicate_input.proto, k)
      self.assertLen(result, 2)
      self.assertEqual(result[0], k)
      self.assertEqual(result[1], k)

  def test_integer_input_triplicate_type_signature(self):
    int_duplicate_input = building_block_factory.create_compiled_input_replication(
        tf.int32, 3)
    self.assertIsInstance(int_duplicate_input,
                          building_blocks.CompiledComputation)
    expected_type_signature = computation_types.FunctionType(
        tf.int32, [tf.int32, tf.int32, tf.int32])
    self.assertEqual(int_duplicate_input.type_signature,
                     expected_type_signature)

  def test_integer_input_triplicate_triplicates_input(self):
    int_duplicate_input = building_block_factory.create_compiled_input_replication(
        tf.int32, 3)
    for k in range(10):
      result = test_utils.run_tensorflow(int_duplicate_input.proto, k)
      self.assertLen(result, 3)
      self.assertEqual(result[0], k)
      self.assertEqual(result[1], k)
      self.assertEqual(result[2], k)

  def test_unnamed_tuple_input_duplicate_type_signature(self):
    tuple_type = [tf.int32, tf.float32]
    tuple_duplicate_input = building_block_factory.create_compiled_input_replication(
        tuple_type, 2)
    self.assertIsInstance(tuple_duplicate_input,
                          building_blocks.CompiledComputation)
    expected_type_signature = computation_types.FunctionType(
        tuple_type, [tuple_type, tuple_type])
    self.assertEqual(tuple_duplicate_input.type_signature,
                     expected_type_signature)

  def test_unnamed_tuple_input_duplicate_duplicates_input(self):
    tuple_type = [tf.int32, tf.float32]
    tuple_duplicate_input = building_block_factory.create_compiled_input_replication(
        tuple_type, 2)
    for k in range(10):
      result = test_utils.run_tensorflow(tuple_duplicate_input.proto,
                                         [k, 10. - k])
      self.assertLen(result, 2)
      self.assertEqual(result[0][0], k)
      self.assertEqual(result[1][0], k)
      self.assertEqual(result[0][1], 10. - k)
      self.assertEqual(result[1][1], 10. - k)

  def test_named_tuple_input_duplicate_type_signature(self):
    tuple_type = [('a', tf.int32), ('b', tf.float32)]
    tuple_duplicate_input = building_block_factory.create_compiled_input_replication(
        tuple_type, 2)
    self.assertIsInstance(tuple_duplicate_input,
                          building_blocks.CompiledComputation)
    expected_type_signature = computation_types.FunctionType(
        tuple_type, [tuple_type, tuple_type])
    self.assertEqual(tuple_duplicate_input.type_signature,
                     expected_type_signature)

  def test_named_tuple_input_duplicate_duplicates_input(self):
    tuple_type = [('a', tf.int32), ('b', tf.float32)]
    tuple_duplicate_input = building_block_factory.create_compiled_input_replication(
        tuple_type, 2)
    for k in range(10):
      result = test_utils.run_tensorflow(tuple_duplicate_input.proto,
                                         [k, 10. - k])
      self.assertLen(result, 2)
      self.assertEqual(result[0].a, k)
      self.assertEqual(result[1].a, k)
      self.assertEqual(result[0].b, 10. - k)
      self.assertEqual(result[1].b, 10. - k)

  def test_sequence_input_duplicate_type_signature(self):
    sequence_type = computation_types.SequenceType(tf.int32)
    sequence_duplicate_input = building_block_factory.create_compiled_input_replication(
        sequence_type, 2)
    self.assertIsInstance(sequence_duplicate_input,
                          building_blocks.CompiledComputation)
    expected_type_signature = computation_types.FunctionType(
        sequence_type, [sequence_type, sequence_type])
    self.assertEqual(sequence_duplicate_input.type_signature,
                     expected_type_signature)

  def test_sequence_input_duplicate_duplicates_input(self):
    sequence_type = computation_types.SequenceType(tf.int32)
    sequence_duplicate_input = building_block_factory.create_compiled_input_replication(
        sequence_type, 2)
    seq = list(range(10))
    result = test_utils.run_tensorflow(sequence_duplicate_input.proto, seq)
    self.assertLen(result, 2)
    self.assertEqual(result[0], seq)
    self.assertEqual(result[1], seq)


class CreateFederatedGetitemCompTest(parameterized.TestCase):

  def test_raises_type_error_on_none(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_getitem_comp(None, 0)

  @parameterized.named_parameters(('clients', placement_literals.CLIENTS),
                                  ('server', placement_literals.SERVER))
  def test_returns_comp(self, placement):
    federated_value = building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32), ('b', tf.bool)],
                                        placement))
    get_0_comp = building_block_factory.create_federated_getitem_comp(
        federated_value, 0)
    self.assertEqual(str(get_0_comp), '(x -> x[0])')
    get_slice_comp = building_block_factory.create_federated_getitem_comp(
        federated_value, slice(None, None, -1))
    self.assertEqual(str(get_slice_comp), '(x -> <b=x[1],a=x[0]>)')


class CreateFederatedGetattrCompTest(parameterized.TestCase):

  def test_raises_type_error_on_none(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_getattr_comp(None, 'x')

  @parameterized.named_parameters(('clients', placement_literals.CLIENTS),
                                  ('server', placement_literals.SERVER))
  def test_returns_comp(self, placement):
    federated_value = building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32), ('b', tf.bool)],
                                        placement))
    get_a_comp = building_block_factory.create_federated_getattr_comp(
        federated_value, 'a')
    self.assertEqual(str(get_a_comp), '(x -> x.a)')
    get_b_comp = building_block_factory.create_federated_getattr_comp(
        federated_value, 'b')
    self.assertEqual(str(get_b_comp), '(x -> x.b)')
    non_federated_arg = building_blocks.Reference(
        'test',
        computation_types.NamedTupleType([('a', tf.int32), ('b', tf.bool)]))
    with self.assertRaises(TypeError):
      _ = building_block_factory.create_federated_getattr_comp(
          non_federated_arg, 'a')
    with self.assertRaisesRegex(ValueError, 'has no element of name `c`'):
      _ = building_block_factory.create_federated_getattr_comp(
          federated_value, 'c')


class CreateFederatedGetattrCallTest(parameterized.TestCase):

  def test_raises_type_error_on_none(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_getattr_call(None, 'x')

  @parameterized.named_parameters(
      ('clients', placement_literals.CLIENTS),
      ('server', placement_literals.SERVER),
  )
  def test_returns_named(self, placement):
    federated_comp_named = building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32),
                                         ('b', tf.bool), tf.int32], placement))
    self.assertEqual(
        str(federated_comp_named.type_signature.member),
        '<a=int32,b=bool,int32>')
    name_a = building_block_factory.create_federated_getattr_call(
        federated_comp_named, 'a')
    name_b = building_block_factory.create_federated_getattr_call(
        federated_comp_named, 'b')
    self.assertIsInstance(name_a.type_signature,
                          computation_types.FederatedType)
    self.assertIsInstance(name_b.type_signature,
                          computation_types.FederatedType)
    self.assertEqual(str(name_a.type_signature.member), 'int32')
    self.assertEqual(str(name_b.type_signature.member), 'bool')
    try:
      type_utils.check_federated_type(
          name_a.type_signature, placement=placement)
    except TypeError:
      self.fail(
          'Function \'check_federated_type\' raised TypeError unexpectedly.')
    try:
      type_utils.check_federated_type(
          name_b.type_signature, placement=placement)
    except TypeError:
      self.fail(
          'Function \'check_federated_type\' raised TypeError unexpectedly.')
    with self.assertRaisesRegex(ValueError, 'has no element of name `c`'):
      _ = building_block_factory.create_federated_getattr_call(
          federated_comp_named, 'c')


class CreateFederatedGetitemCallTest(parameterized.TestCase):

  def test_fails_value(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_getitem_call(None, 0)

  @parameterized.named_parameters(
      ('clients', placement_literals.CLIENTS),
      ('server', placement_literals.SERVER),
  )
  def test_returns_named(self, placement):
    federated_comp_named = building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32), ('b', tf.bool)],
                                        placement))
    self.assertEqual(
        str(federated_comp_named.type_signature.member), '<a=int32,b=bool>')
    idx_0 = building_block_factory.create_federated_getitem_call(
        federated_comp_named, 0)
    idx_1 = building_block_factory.create_federated_getitem_call(
        federated_comp_named, 1)
    self.assertIsInstance(idx_0.type_signature, computation_types.FederatedType)
    self.assertIsInstance(idx_1.type_signature, computation_types.FederatedType)
    self.assertEqual(str(idx_0.type_signature.member), 'int32')
    self.assertEqual(str(idx_1.type_signature.member), 'bool')
    try:
      type_utils.check_federated_type(idx_0.type_signature, placement=placement)
    except TypeError:
      self.fail(
          'Function \'check_federated_type\' raised TypeError unexpectedly.')
    try:
      type_utils.check_federated_type(idx_1.type_signature, placement=placement)
    except TypeError:
      self.fail(
          'Function \'check_federated_type\' raised TypeError unexpectedly.')
    flipped = building_block_factory.create_federated_getitem_call(
        federated_comp_named, slice(None, None, -1))
    self.assertIsInstance(flipped.type_signature,
                          computation_types.FederatedType)
    self.assertEqual(str(flipped.type_signature.member), '<b=bool,a=int32>')
    try:
      type_utils.check_federated_type(
          flipped.type_signature, placement=placement)
    except TypeError:
      self.fail(
          'Function \'check_federated_type\' raised TypeError unexpectedly.')

  @parameterized.named_parameters(
      ('clients', placement_literals.CLIENTS),
      ('server', placement_literals.SERVER),
  )
  def test_returns_unnamed(self, placement):
    federated_comp_unnamed = building_blocks.Reference(
        'test', computation_types.FederatedType([tf.int32, tf.bool], placement))
    self.assertEqual(
        str(federated_comp_unnamed.type_signature.member), '<int32,bool>')
    unnamed_idx_0 = building_block_factory.create_federated_getitem_call(
        federated_comp_unnamed, 0)
    unnamed_idx_1 = building_block_factory.create_federated_getitem_call(
        federated_comp_unnamed, 1)
    self.assertIsInstance(unnamed_idx_0.type_signature,
                          computation_types.FederatedType)
    self.assertIsInstance(unnamed_idx_1.type_signature,
                          computation_types.FederatedType)
    self.assertEqual(str(unnamed_idx_0.type_signature.member), 'int32')
    self.assertEqual(str(unnamed_idx_1.type_signature.member), 'bool')
    try:
      type_utils.check_federated_type(
          unnamed_idx_0.type_signature, placement=placement)
    except TypeError:
      self.fail(
          'Function \'check_federated_type\' raised TypeError unexpectedly.')
    try:
      type_utils.check_federated_type(
          unnamed_idx_1.type_signature, placement=placement)
    except TypeError:
      self.fail(
          'Function \'check_federated_type\' raised TypeError unexpectedly.')
    unnamed_flipped = building_block_factory.create_federated_getitem_call(
        federated_comp_unnamed, slice(None, None, -1))
    self.assertIsInstance(unnamed_flipped.type_signature,
                          computation_types.FederatedType)
    self.assertEqual(str(unnamed_flipped.type_signature.member), '<bool,int32>')
    try:
      type_utils.check_federated_type(
          unnamed_flipped.type_signature, placement=placement)
    except TypeError:
      self.fail(
          'Function \'check_federated_type\' raised TypeError unexpectedly.')


class CreateFederatedSetitemLambdaTest(parameterized.TestCase):

  def test_fails_on_bad_type(self):
    bad_type = computation_types.FederatedType([('a', tf.int32)],
                                               placement_literals.CLIENTS)
    value_comp = building_blocks.Data('x', tf.int32)
    with self.assertRaises(TypeError):
      _ = building_block_factory.create_named_tuple_setattr_lambda(
          bad_type, 'a', value_comp)

  def test_fails_on_none_name(self):
    good_type = computation_types.NamedTupleType([('a', tf.int32)])
    value_comp = building_blocks.Data('x', tf.int32)
    with self.assertRaises(TypeError):
      _ = building_block_factory.create_named_tuple_setattr_lambda(
          good_type, None, value_comp)

  def test_fails_on_none_value(self):
    good_type = computation_types.NamedTupleType([('a', tf.int32)])
    with self.assertRaises(TypeError):
      _ = building_block_factory.create_named_tuple_setattr_lambda(
          good_type, 'a', None)

  def test_fails_implicit_type_conversion(self):
    good_type = computation_types.NamedTupleType([('a', tf.int32),
                                                  ('b', tf.bool)])
    value_comp = building_blocks.Data('x', tf.int32)
    with self.assertRaisesRegex(TypeError, 'incompatible type'):
      _ = building_block_factory.create_named_tuple_setattr_lambda(
          good_type, 'b', value_comp)

  def test_fails_unknown_name(self):
    good_type = computation_types.NamedTupleType([('a', tf.int32),
                                                  ('b', tf.bool)])
    value_comp = building_blocks.Data('x', tf.int32)
    with self.assertRaises(AttributeError):
      _ = building_block_factory.create_named_tuple_setattr_lambda(
          good_type, 'c', value_comp)

  def test_replaces_single_element(self):
    good_type = computation_types.NamedTupleType([('a', tf.int32),
                                                  ('b', tf.bool)])
    value_comp = building_blocks.Data('x', tf.int32)
    lam = building_block_factory.create_named_tuple_setattr_lambda(
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
    value_comp = building_blocks.Data('x', tf.int32)
    lam = building_block_factory.create_named_tuple_setattr_lambda(
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
    value_comp = building_blocks.Data('x', tf.int32)
    lam = building_block_factory.create_named_tuple_setattr_lambda(
        good_type, 'a', value_comp)
    self.assertTrue(
        type_utils.are_equivalent_types(lam.type_signature.parameter,
                                        lam.type_signature.result))


class CreateFederatedSetatterCallTest(parameterized.TestCase):

  def test_fails_on_none_federated_comp(self):
    value_comp = building_blocks.Data('x', tf.int32)
    with self.assertRaises(TypeError):
      _ = building_block_factory.create_federated_setattr_call(
          None, 'a', value_comp)

  def test_fails_non_federated_type(self):
    bad_type = computation_types.NamedTupleType([('a', tf.int32),
                                                 (None, tf.float32),
                                                 ('b', tf.bool)])
    bad_comp = building_blocks.Data('data', bad_type)
    value_comp = building_blocks.Data('x', tf.int32)

    with self.assertRaises(TypeError):
      _ = building_block_factory.create_federated_setattr_call(
          bad_comp, 'a', value_comp)

  def test_fails_on_none_name(self):
    named_tuple_type = computation_types.NamedTupleType([('a', tf.int32),
                                                         (None, tf.float32),
                                                         ('b', tf.bool)])
    good_type = computation_types.FederatedType(named_tuple_type,
                                                placement_literals.CLIENTS)
    acceptable_comp = building_blocks.Data('data', good_type)
    value_comp = building_blocks.Data('x', tf.int32)

    with self.assertRaises(TypeError):
      _ = building_block_factory.create_federated_setattr_call(
          acceptable_comp, None, value_comp)

  def test_fails_on_none_value(self):
    named_tuple_type = computation_types.NamedTupleType([('a', tf.int32),
                                                         (None, tf.float32),
                                                         ('b', tf.bool)])
    good_type = computation_types.FederatedType(named_tuple_type,
                                                placement_literals.CLIENTS)
    acceptable_comp = building_blocks.Data('data', good_type)

    with self.assertRaises(TypeError):
      _ = building_block_factory.create_federated_setattr_call(
          acceptable_comp, 'a', None)

  def test_constructs_correct_intrinsic_clients(self):
    named_tuple_type = computation_types.NamedTupleType([('a', tf.int32),
                                                         (None, tf.float32),
                                                         ('b', tf.bool)])
    good_type = computation_types.FederatedType(named_tuple_type,
                                                placement_literals.CLIENTS)
    federated_comp = building_blocks.Data('federated_comp', good_type)
    value_comp = building_blocks.Data('x', tf.int32)

    federated_setattr = building_block_factory.create_federated_setattr_call(
        federated_comp, 'a', value_comp)
    self.assertEqual(federated_setattr.function.uri,
                     intrinsic_defs.FEDERATED_MAP.uri)

  def test_constructs_correct_intrinsic_server(self):
    named_tuple_type = computation_types.NamedTupleType([('a', tf.int32),
                                                         (None, tf.float32),
                                                         ('b', tf.bool)])
    good_type = computation_types.FederatedType(named_tuple_type,
                                                placement_literals.SERVER)
    federated_comp = building_blocks.Data('federated_comp', good_type)
    value_comp = building_blocks.Data('x', tf.int32)

    federated_setattr = building_block_factory.create_federated_setattr_call(
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
    federated_comp = building_blocks.Data('federated_comp', good_type)
    value_comp = building_blocks.Data('x', tf.int32)

    federated_setattr = building_block_factory.create_federated_setattr_call(
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
    federated_comp = building_blocks.Data('federated_comp', good_type)
    value_comp = building_blocks.Data('x', tf.int32)

    federated_setattr = building_block_factory.create_federated_setattr_call(
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
    federated_comp = building_blocks.Data('federated_comp', good_type)
    value_comp = building_blocks.Data('x', tf.int32)

    federated_setattr = building_block_factory.create_federated_setattr_call(
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
    comp2 = building_blocks.Data('y', tf.int32)
    with self.assertRaises(TypeError):
      building_block_factory.create_computation_appending(None, comp2)

  def test_raises_type_error_with_none_comp2(self):
    value = building_blocks.Data('x', tf.int32)
    comp1 = building_blocks.Tuple([value, value])
    with self.assertRaises(TypeError):
      building_block_factory.create_computation_appending(comp1, None)

  def test_raises_type_error_with_comp1_bad_type(self):
    comp1 = building_blocks.Data('x', tf.int32)
    comp2 = building_blocks.Data('y', tf.int32)
    with self.assertRaises(TypeError):
      building_block_factory.create_computation_appending(comp1, comp2)

  def test_returns_comp_unnamed(self):
    value = building_blocks.Data('x', tf.int32)
    comp1 = building_blocks.Tuple([value, value])
    comp2 = building_blocks.Data('y', tf.int32)
    comp = building_block_factory.create_computation_appending(comp1, comp2)
    self.assertEqual(
        comp.compact_representation(),
        '(let comps=<<x,x>,y> in <comps[0][0],comps[0][1],comps[1]>)')
    self.assertEqual(str(comp.type_signature), '<int32,int32,int32>')

  def test_returns_comp_named(self):
    value = building_blocks.Data('x', tf.int32)
    comp1 = building_blocks.Tuple((
        ('a', value),
        ('b', value),
    ))
    comp2 = building_blocks.Data('y', tf.int32)
    comp = building_block_factory.create_computation_appending(
        comp1, ('c', comp2))
    self.assertEqual(
        comp.compact_representation(),
        '(let comps=<<a=x,b=x>,y> in <a=comps[0][0],b=comps[0][1],c=comps[1]>)')
    self.assertEqual(str(comp.type_signature), '<a=int32,b=int32,c=int32>')


class CreateFederatedAggregateTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    zero = building_blocks.Data('z', tf.int32)
    accumulate_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    accumulate_result = building_blocks.Data('a', tf.int32)
    accumulate = building_blocks.Lambda('x', accumulate_type, accumulate_result)
    merge_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    merge_result = building_blocks.Data('m', tf.int32)
    merge = building_blocks.Lambda('x', merge_type, merge_result)
    report_ref = building_blocks.Reference('r', tf.int32)
    report = building_blocks.Lambda(report_ref.name, report_ref.type_signature,
                                    report_ref)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_aggregate(None, zero, accumulate,
                                                        merge, report)

  def test_raises_type_error_with_none_zero(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    accumulate_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    accumulate_result = building_blocks.Data('a', tf.int32)
    accumulate = building_blocks.Lambda('x', accumulate_type, accumulate_result)
    merge_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    merge_result = building_blocks.Data('m', tf.int32)
    merge = building_blocks.Lambda('x', merge_type, merge_result)
    report_ref = building_blocks.Reference('r', tf.int32)
    report = building_blocks.Lambda(report_ref.name, report_ref.type_signature,
                                    report_ref)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_aggregate(value, None, accumulate,
                                                        merge, report)

  def test_raises_type_error_with_none_accumulate(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    zero = building_blocks.Data('z', tf.int32)
    merge_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    merge_result = building_blocks.Data('m', tf.int32)
    merge = building_blocks.Lambda('x', merge_type, merge_result)
    report_ref = building_blocks.Reference('r', tf.int32)
    report = building_blocks.Lambda(report_ref.name, report_ref.type_signature,
                                    report_ref)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_aggregate(value, zero, None,
                                                        merge, report)

  def test_raises_type_error_with_none_merge(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    zero = building_blocks.Data('z', tf.int32)
    accumulate_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    accumulate_result = building_blocks.Data('a', tf.int32)
    accumulate = building_blocks.Lambda('x', accumulate_type, accumulate_result)
    report_ref = building_blocks.Reference('r', tf.int32)
    report = building_blocks.Lambda(report_ref.name, report_ref.type_signature,
                                    report_ref)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_aggregate(value, zero, accumulate,
                                                        None, report)

  def test_raises_type_error_with_none_report(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    zero = building_blocks.Data('z', tf.int32)
    accumulate_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    accumulate_result = building_blocks.Data('a', tf.int32)
    accumulate = building_blocks.Lambda('x', accumulate_type, accumulate_result)
    merge_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    merge_result = building_blocks.Data('m', tf.int32)
    merge = building_blocks.Lambda('x', merge_type, merge_result)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_aggregate(value, zero, accumulate,
                                                        merge, None)

  def test_returns_federated_aggregate(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    zero = building_blocks.Data('z', tf.int32)
    accumulate_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    accumulate_result = building_blocks.Data('a', tf.int32)
    accumulate = building_blocks.Lambda('x', accumulate_type, accumulate_result)
    merge_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    merge_result = building_blocks.Data('m', tf.int32)
    merge = building_blocks.Lambda('x', merge_type, merge_result)
    report_ref = building_blocks.Reference('r', tf.int32)
    report = building_blocks.Lambda(report_ref.name, report_ref.type_signature,
                                    report_ref)
    comp = building_block_factory.create_federated_aggregate(
        value, zero, accumulate, merge, report)
    self.assertEqual(comp.compact_representation(),
                     'federated_aggregate(<v,z,(x -> a),(x -> m),(r -> r)>)')
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')


class CreateFederatedApplyTest(absltest.TestCase):

  def test_raises_type_error_with_none_fn(self):
    arg = building_blocks.Data('y', tf.int32)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_apply(None, arg)

  def test_raises_type_error_with_none_arg(self):
    ref = building_blocks.Reference('x', tf.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_apply(fn, None)

  def test_returns_federated_apply(self):
    ref = building_blocks.Reference('x', tf.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg_type = computation_types.FederatedType(tf.int32, placements.SERVER)
    arg = building_blocks.Data('y', arg_type)
    comp = building_block_factory.create_federated_apply(fn, arg)
    self.assertEqual(comp.compact_representation(),
                     'federated_apply(<(x -> x),y>)')
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')


class CreateFederatedBroadcastTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_broadcast(None)

  def test_returns_federated_broadcast(self):
    value_type = computation_types.FederatedType(tf.int32, placements.SERVER)
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_broadcast(value)
    self.assertEqual(comp.compact_representation(), 'federated_broadcast(v)')
    self.assertEqual(str(comp.type_signature), 'int32@CLIENTS')


class CreateFederatedCollectTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_collect(None)

  def test_returns_federated_collect(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_collect(value)
    self.assertEqual(comp.compact_representation(), 'federated_collect(v)')
    self.assertEqual(str(comp.type_signature), 'int32*@SERVER')

  def test_constructs_federated_collect_with_all_equal_argument(self):
    value_type = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=True)
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_collect(value)
    self.assertEqual(comp.compact_representation(), 'federated_collect(v)')
    self.assertEqual(str(comp.type_signature), 'int32*@SERVER')


class CreateFederatedMapTest(absltest.TestCase):

  def test_raises_type_error_with_none_fn(self):
    arg = building_blocks.Data('y', tf.int32)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_map(None, arg)

  def test_raises_type_error_with_none_arg(self):
    ref = building_blocks.Reference('x', tf.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_map(fn, None)

  def test_returns_federated_map(self):
    ref = building_blocks.Reference('x', tf.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = building_blocks.Data('y', arg_type)
    comp = building_block_factory.create_federated_map(fn, arg)
    self.assertEqual(comp.compact_representation(),
                     'federated_map(<(x -> x),y>)')
    self.assertEqual(str(comp.type_signature), '{int32}@CLIENTS')


class CreateFederatedMapAllEqualTest(absltest.TestCase):

  def test_raises_type_error_with_none_fn(self):
    arg = building_blocks.Data('y', tf.int32)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_map_all_equal(None, arg)

  def test_raises_type_error_with_none_arg(self):
    ref = building_blocks.Reference('x', tf.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_map_all_equal(fn, None)

  def test_returns_federated_map_all_equal(self):
    ref = building_blocks.Reference('x', tf.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg_type = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=True)
    arg = building_blocks.Data('y', arg_type)
    comp = building_block_factory.create_federated_map_all_equal(fn, arg)
    self.assertEqual(comp.compact_representation(),
                     'federated_map_all_equal(<(x -> x),y>)')
    self.assertEqual(str(comp.type_signature), 'int32@CLIENTS')


class CreateFederatedMapOrApplyTest(absltest.TestCase):

  def test_raises_type_error_with_none_fn(self):
    arg = building_blocks.Data('y', tf.int32)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_map_or_apply(None, arg)

  def test_raises_type_error_with_none_arg(self):
    ref = building_blocks.Reference('x', tf.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_map_or_apply(fn, None)

  def test_returns_federated_apply(self):
    ref = building_blocks.Reference('x', tf.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg_type = computation_types.FederatedType(tf.int32, placements.SERVER)
    arg = building_blocks.Data('y', arg_type)
    comp = building_block_factory.create_federated_map_or_apply(fn, arg)
    self.assertEqual(comp.compact_representation(),
                     'federated_apply(<(x -> x),y>)')
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')

  def test_returns_federated_map(self):
    ref = building_blocks.Reference('x', tf.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = building_blocks.Data('y', arg_type)
    comp = building_block_factory.create_federated_map_or_apply(fn, arg)
    self.assertEqual(comp.compact_representation(),
                     'federated_map(<(x -> x),y>)')
    self.assertEqual(str(comp.type_signature), '{int32}@CLIENTS')


class CreateFederatedMeanTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_mean(None, None)

  def test_returns_federated_mean(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_mean(value, None)
    self.assertEqual(comp.compact_representation(), 'federated_mean(v)')
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')

  def test_returns_federated_weighted_mean(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    weight_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    weight = building_blocks.Data('w', weight_type)
    comp = building_block_factory.create_federated_mean(value, weight)
    self.assertEqual(comp.compact_representation(),
                     'federated_weighted_mean(<v,w>)')
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')


class CreateFederatedReduceTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    zero = building_blocks.Data('z', tf.int32)
    op_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    op_result = building_blocks.Data('o', tf.int32)
    op = building_blocks.Lambda('x', op_type, op_result)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_reduce(None, zero, op)

  def test_raises_type_error_with_none_zero(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    op_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    op_result = building_blocks.Data('o', tf.int32)
    op = building_blocks.Lambda('x', op_type, op_result)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_reduce(value, None, op)

  def test_raises_type_error_with_none_op(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    zero = building_blocks.Data('z', tf.int32)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_reduce(value, zero, None)

  def test_returns_federated_reduce(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    zero = building_blocks.Data('z', tf.int32)
    op_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    op_result = building_blocks.Data('o', tf.int32)
    op = building_blocks.Lambda('x', op_type, op_result)
    comp = building_block_factory.create_federated_reduce(value, zero, op)
    self.assertEqual(comp.compact_representation(),
                     'federated_reduce(<v,z,(x -> o)>)')
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')


class CreateFederatedSumTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_sum(None)

  def test_returns_federated_sum(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_sum(value)
    self.assertEqual(comp.compact_representation(), 'federated_sum(v)')
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')


class CreateFederatedUnzipTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_unzip(None)

  def test_returns_tuple_federated_map_with_empty_value(self):
    value_type = computation_types.FederatedType([], placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    with self.assertRaises(ValueError):
      building_block_factory.create_federated_unzip(value)

  def test_returns_tuple_federated_map_with_one_value_unnamed(self):
    value_type = computation_types.FederatedType((tf.int32,),
                                                 placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
    self.assertEqual(
        comp.compact_representation(),
        '(let value=v in <federated_map(<(arg -> arg[0]),value>)>)')
    self.assertEqual(str(comp.type_signature), '<{int32}@CLIENTS>')

  def test_returns_tuple_federated_map_with_one_value_named(self):
    type_signature = computation_types.NamedTupleType((('a', tf.int32),))
    value_type = computation_types.FederatedType(type_signature,
                                                 placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
    self.assertEqual(
        comp.compact_representation(),
        '(let value=v in <a=federated_map(<(arg -> arg[0]),value>)>)')
    self.assertEqual(str(comp.type_signature), '<a={int32}@CLIENTS>')

  def test_returns_tuple_federated_map_with_two_values_unnamed(self):
    value_type = computation_types.FederatedType((tf.int32, tf.int32),
                                                 placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
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
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
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
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
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
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
    self.assertEqual(
        comp.compact_representation(),
        '(let value=v in <federated_apply(<(arg -> arg[0]),value>)>)')
    self.assertEqual(str(comp.type_signature), '<int32@SERVER>')

  def test_returns_tuple_federated_apply_with_one_value_named(self):
    type_signature = computation_types.NamedTupleType((('a', tf.int32),))
    value_type = computation_types.FederatedType(type_signature,
                                                 placements.SERVER)
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
    self.assertEqual(
        comp.compact_representation(),
        '(let value=v in <a=federated_apply(<(arg -> arg[0]),value>)>)')
    self.assertEqual(str(comp.type_signature), '<a=int32@SERVER>')

  def test_returns_tuple_federated_apply_with_two_values_unnamed(self):
    value_type = computation_types.FederatedType((tf.int32, tf.int32),
                                                 placements.SERVER)
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
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
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
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
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
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
      building_block_factory.create_federated_value(None,
                                                    placement_literals.CLIENTS)

  def test_raises_type_error_with_none_placement(self):
    value = building_blocks.Data('v', tf.int32)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_value(value, None)

  def test_raises_type_error_with_unknown_placement(self):
    value = building_blocks.Data('v', tf.int32)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_value(value, 'unknown')

  def test_returns_federated_value_at_clients(self):
    value = building_blocks.Data('v', tf.int32)
    comp = building_block_factory.create_federated_value(
        value, placement_literals.CLIENTS)
    self.assertEqual(comp.compact_representation(),
                     'federated_value_at_clients(v)')
    self.assertEqual(str(comp.type_signature), 'int32@CLIENTS')

  def test_returns_federated_value_at_server(self):
    value = building_blocks.Data('v', tf.int32)
    comp = building_block_factory.create_federated_value(
        value, placement_literals.SERVER)
    self.assertEqual(comp.compact_representation(),
                     'federated_value_at_server(v)')
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')


class CreateFederatedZipTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_zip(None)

  def test_raises_value_error_with_empty_value(self):
    value_type = computation_types.NamedTupleType([])
    value = building_blocks.Data('v', value_type)
    with self.assertRaises(ValueError):
      building_block_factory.create_federated_zip(value)

  def test_returns_federated_map_with_one_value_unnamed(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    value_type = computation_types.NamedTupleType((type_signature,))
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
    self.assertEqual(comp.compact_representation(),
                     'federated_map(<(arg -> <arg>),v[0]>)')
    self.assertEqual(str(comp.type_signature), '{<int32>}@CLIENTS')

  def test_returns_federated_map_with_one_value_unnamed_tuple(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    tup = building_blocks.Tuple((value,))
    comp = building_block_factory.create_federated_zip(tup)
    self.assertEqual(comp.compact_representation(),
                     'federated_map(<(arg -> <arg>),<v>[0]>)')
    self.assertEqual(str(comp.type_signature), '{<int32>}@CLIENTS')

  def test_returns_federated_map_with_one_value_named(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    value_type = computation_types.NamedTupleType((('a', type_signature),))
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
    self.assertEqual(comp.compact_representation(),
                     'federated_map(<(arg -> <a=arg>),v[0]>)')
    self.assertEqual(str(comp.type_signature), '{<a=int32>}@CLIENTS')

  def test_returns_federated_map_with_one_value_named_tuple(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    tup = building_blocks.Tuple((('a', value),))
    comp = building_block_factory.create_federated_zip(tup)
    self.assertEqual(comp.compact_representation(),
                     'federated_map(<(arg -> <a=arg>),<a=v>[0]>)')
    self.assertEqual(str(comp.type_signature), '{<a=int32>}@CLIENTS')

  def test_returns_federated_zip_at_clients_with_two_values_unnamed(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    value_type = computation_types.NamedTupleType(
        (type_signature, type_signature))
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
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
    value = building_blocks.Data('v', value_type)
    tup = building_blocks.Tuple((value, value))
    comp = building_block_factory.create_federated_zip(tup)
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
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
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
    value = building_blocks.Data('v', value_type)
    tup = building_blocks.Tuple((('a', value), ('b', value)))
    comp = building_block_factory.create_federated_zip(tup)
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
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
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
    value = building_blocks.Data('v', value_type)
    tup = building_blocks.Tuple((value, value, value))
    comp = building_block_factory.create_federated_zip(tup)
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
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
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
    value = building_blocks.Data('v', value_type)
    tup = building_blocks.Tuple((
        ('a', value),
        ('b', value),
        ('c', value),
    ))
    comp = building_block_factory.create_federated_zip(tup)
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
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
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
    value1 = building_blocks.Data('v1', value_type1)
    value_type2 = computation_types.FederatedType(tf.float32,
                                                  placements.CLIENTS)
    value2 = building_blocks.Data('v2', value_type2)
    value_type3 = computation_types.FederatedType(tf.bool, placements.CLIENTS)
    value3 = building_blocks.Data('v3', value_type3)
    tup = building_blocks.Tuple((value1, value2, value3))
    comp = building_block_factory.create_federated_zip(tup)
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
    value = building_blocks.Data('v', value_type)
    tup = building_blocks.Tuple((value,))
    comp = building_block_factory.create_federated_zip(tup)
    self.assertEqual(comp.compact_representation(),
                     'federated_apply(<(arg -> <arg>),<v>[0]>)')
    self.assertEqual(str(comp.type_signature), '<int32>@SERVER')

  def test_returns_federated_apply_with_one_value_named(self):
    value_type = computation_types.FederatedType(tf.int32, placements.SERVER)
    value = building_blocks.Data('v', value_type)
    tup = building_blocks.Tuple((('a', value),))
    comp = building_block_factory.create_federated_zip(tup)
    self.assertEqual(comp.compact_representation(),
                     'federated_apply(<(arg -> <a=arg>),<a=v>[0]>)')
    self.assertEqual(str(comp.type_signature), '<a=int32>@SERVER')

  def test_returns_federated_zip_at_server_with_two_values_unnamed(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.SERVER)
    value_type = computation_types.NamedTupleType(
        (type_signature, type_signature))
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
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
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
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
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
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
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
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
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
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
      building_block_factory.create_generic_constant(None, 0)

  def test_raises_non_scalar(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_generic_constant([tf.int32], [0])

  def test_constructs_tensor_zero(self):
    tensor_type = computation_types.TensorType(tf.float32, [2, 2])
    tensor_zero = building_block_factory.create_generic_constant(tensor_type, 0)
    self.assertEqual(tensor_zero.type_signature, tensor_type)
    self.assertIsInstance(tensor_zero, building_blocks.Call)
    self.assertTrue(
        np.array_equal(
            test_utils.run_tensorflow(tensor_zero.function.proto),
            np.zeros([2, 2])))

  def test_create_unnamed_tuple_zero(self):
    tuple_type = [computation_types.TensorType(tf.float32, [2, 2])] * 2
    tuple_zero = building_block_factory.create_generic_constant(tuple_type, 0)
    self.assertEqual(tuple_zero.type_signature,
                     computation_types.to_type(tuple_type))
    self.assertIsInstance(tuple_zero, building_blocks.Call)
    result = test_utils.run_tensorflow(tuple_zero.function.proto)
    self.assertLen(result, 2)
    self.assertTrue(np.array_equal(result[0], np.zeros([2, 2])))
    self.assertTrue(np.array_equal(result[1], np.zeros([2, 2])))

  def test_create_named_tuple_one(self):
    tuple_type = [('a', computation_types.TensorType(tf.float32, [2, 2])),
                  ('b', computation_types.TensorType(tf.float32, [2, 2]))]
    tuple_zero = building_block_factory.create_generic_constant(tuple_type, 1)
    self.assertEqual(tuple_zero.type_signature,
                     computation_types.to_type(tuple_type))
    self.assertIsInstance(tuple_zero, building_blocks.Call)
    result = test_utils.run_tensorflow(tuple_zero.function.proto)
    self.assertLen(result, 2)
    self.assertTrue(np.array_equal(result.a, np.ones([2, 2])))
    self.assertTrue(np.array_equal(result.b, np.ones([2, 2])))

  def test_create_federated_tensor_one(self):
    fed_type = computation_types.FederatedType(
        computation_types.TensorType(tf.float32, [2, 2]),
        placement_literals.CLIENTS)
    fed_zero = building_block_factory.create_generic_constant(fed_type, 1)
    self.assertEqual(fed_zero.type_signature.member, fed_type.member)
    self.assertEqual(fed_zero.type_signature.placement, fed_type.placement)
    self.assertTrue(fed_zero.type_signature.all_equal)
    self.assertIsInstance(fed_zero, building_blocks.Call)
    self.assertIsInstance(fed_zero.function, building_blocks.Intrinsic)
    self.assertEqual(fed_zero.function.uri,
                     intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri)
    self.assertIsInstance(fed_zero.argument, building_blocks.Call)
    self.assertTrue(
        np.array_equal(
            test_utils.run_tensorflow(fed_zero.argument.function.proto),
            np.ones([2, 2])))

  def test_create_federated_named_tuple_one(self):
    tuple_type = [('a', computation_types.TensorType(tf.float32, [2, 2])),
                  ('b', computation_types.TensorType(tf.float32, [2, 2]))]
    fed_type = computation_types.FederatedType(tuple_type,
                                               placement_literals.SERVER)
    fed_zero = building_block_factory.create_generic_constant(fed_type, 1)
    self.assertEqual(fed_zero.type_signature.member, fed_type.member)
    self.assertEqual(fed_zero.type_signature.placement, fed_type.placement)
    self.assertTrue(fed_zero.type_signature.all_equal)
    self.assertIsInstance(fed_zero, building_blocks.Call)
    self.assertIsInstance(fed_zero.function, building_blocks.Intrinsic)
    self.assertEqual(fed_zero.function.uri,
                     intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri)
    self.assertIsInstance(fed_zero.argument, building_blocks.Call)
    result = test_utils.run_tensorflow(fed_zero.argument.function.proto)
    self.assertLen(result, 2)
    self.assertTrue(np.array_equal(result.a, np.ones([2, 2])))
    self.assertTrue(np.array_equal(result.b, np.ones([2, 2])))

  def test_create_named_tuple_of_federated_tensors_zero(self):
    fed_type = computation_types.FederatedType(
        computation_types.TensorType(tf.float32, [2, 2]),
        placement_literals.CLIENTS,
        all_equal=True)
    tuple_type = [('a', fed_type), ('b', fed_type)]
    zero = building_block_factory.create_generic_constant(tuple_type, 0)
    fed_zero = zero.argument[0]
    self.assertEqual(zero.type_signature, computation_types.to_type(tuple_type))
    self.assertIsInstance(fed_zero.function, building_blocks.Intrinsic)
    self.assertEqual(fed_zero.function.uri,
                     intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri)
    self.assertIsInstance(fed_zero.argument, building_blocks.Call)
    self.assertTrue(
        np.array_equal(
            test_utils.run_tensorflow(fed_zero.argument.function.proto),
            np.zeros([2, 2])))


class CreateSequenceMapTest(absltest.TestCase):

  def test_raises_type_error_with_none_fn(self):
    arg_type = computation_types.SequenceType(tf.int32)
    arg = building_blocks.Data('y', arg_type)
    with self.assertRaises(TypeError):
      building_block_factory.create_sequence_map(None, arg)

  def test_raises_type_error_with_none_arg(self):
    ref = building_blocks.Reference('x', tf.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    with self.assertRaises(TypeError):
      building_block_factory.create_sequence_map(fn, None)

  def test_returns_sequence_map(self):
    ref = building_blocks.Reference('x', tf.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    arg_type = computation_types.SequenceType(tf.int32)
    arg = building_blocks.Data('y', arg_type)
    comp = building_block_factory.create_sequence_map(fn, arg)
    self.assertEqual(comp.compact_representation(),
                     'sequence_map(<(x -> x),y>)')
    self.assertEqual(str(comp.type_signature), 'int32*')


class CreateSequenceReduceTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    zero = building_blocks.Data('z', tf.int32)
    op_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    op_result = building_blocks.Data('o', tf.int32)
    op = building_blocks.Lambda('x', op_type, op_result)
    with self.assertRaises(TypeError):
      building_block_factory.create_sequence_reduce(None, zero, op)

  def test_raises_type_error_with_none_zero(self):
    value_type = computation_types.SequenceType(tf.int32)
    value = building_blocks.Data('v', value_type)
    op_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    op_result = building_blocks.Data('o', tf.int32)
    op = building_blocks.Lambda('x', op_type, op_result)
    with self.assertRaises(TypeError):
      building_block_factory.create_sequence_reduce(value, None, op)

  def test_raises_type_error_with_none_op(self):
    value_type = computation_types.SequenceType(tf.int32)
    value = building_blocks.Data('v', value_type)
    zero = building_blocks.Data('z', tf.int32)
    with self.assertRaises(TypeError):
      building_block_factory.create_sequence_reduce(value, zero, None)

  def test_returns_sequence_reduce(self):
    value_type = computation_types.SequenceType(tf.int32)
    value = building_blocks.Data('v', value_type)
    zero = building_blocks.Data('z', tf.int32)
    op_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    op_result = building_blocks.Data('o', tf.int32)
    op = building_blocks.Lambda('x', op_type, op_result)
    comp = building_block_factory.create_sequence_reduce(value, zero, op)
    self.assertEqual(comp.compact_representation(),
                     'sequence_reduce(<v,z,(x -> o)>)')
    self.assertEqual(str(comp.type_signature), 'int32')


class CreateSequenceSumTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_sequence_sum(None)

  def test_returns_federated_sum(self):
    value_type = computation_types.SequenceType(tf.int32)
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_sequence_sum(value)
    self.assertEqual(comp.compact_representation(), 'sequence_sum(v)')
    self.assertEqual(str(comp.type_signature), 'int32')


class CreateNamedFederatedTupleTest(parameterized.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_named_federated_tuple(None, ['a'])

  def test_raises_non_federated_type(self):
    bad_comp = building_blocks.Data('x', computation_types.to_type(tf.int32))
    with self.assertRaises(TypeError):
      building_block_factory.create_named_federated_tuple(bad_comp, ['a'])

  def test_raises_federated_non_tuple(self):
    bad_comp = building_blocks.Data(
        'x',
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    with self.assertRaises(TypeError):
      building_block_factory.create_named_federated_tuple(bad_comp, ['a'])

  def test_raises_on_naked_string(self):
    data_tuple = building_blocks.Data(
        'x',
        computation_types.FederatedType([tf.int32], placement_literals.CLIENTS))
    with self.assertRaises(TypeError):
      building_block_factory.create_named_federated_tuple(data_tuple, 'a')

  def test_raises_list_of_ints(self):
    data_tuple = building_blocks.Data(
        'x',
        computation_types.FederatedType([tf.int32], placement_literals.SERVER))
    with self.assertRaises(TypeError):
      building_block_factory.create_named_federated_tuple(data_tuple, [1])

  def test_raises_wrong_list_length(self):
    data_tuple = building_blocks.Data(
        'x',
        computation_types.FederatedType([tf.int32], placement_literals.SERVER))
    with self.assertRaises(ValueError):
      building_block_factory.create_named_federated_tuple(
          data_tuple, ['a', 'b'])

  @parameterized.named_parameters(
      ('server', placement_literals.SERVER),
      ('clients', placement_literals.CLIENTS),
  )
  def test_constructs_correct_type_from_unnamed_tuple(self, placement):
    fed_type = computation_types.FederatedType([tf.int32, tf.float32],
                                               placement)
    data_tuple = building_blocks.Data('x', fed_type)
    named_tuple = building_block_factory.create_named_federated_tuple(
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
    data_tuple = building_blocks.Data('x', fed_type)
    named_tuple = building_block_factory.create_named_federated_tuple(
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
    data_tuple = building_blocks.Data('data', ntt)
    named_tuple = building_block_factory.create_named_federated_tuple(
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
    data_tuple = building_blocks.Data('data', fed_type)
    named_tuple = building_block_factory.create_named_federated_tuple(
        data_tuple, ['a', 'b'])
    self.assertRegexMatch(
        named_tuple.compact_representation(),
        [r'federated_(map|apply)\(<\(x -> <a=x\[0\],b=x\[1\]>\),data>\)'])


class CreateNamedTupleTest(absltest.TestCase):

  def test_raises_type_error_with_none_comp(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_named_tuple(None, ('a',))

  def test_raises_type_error_with_wrong_comp_type(self):
    comp = building_blocks.Data('data', tf.int32)
    with self.assertRaises(TypeError):
      building_block_factory.create_named_tuple(comp, ('a',))

  def test_raises_type_error_with_wrong_names_type_string(self):
    type_signature = computation_types.NamedTupleType((tf.int32, tf.int32))
    comp = building_blocks.Data('data', type_signature)
    with self.assertRaises(TypeError):
      building_block_factory.create_named_tuple(comp, 'a')

  def test_raises_type_error_with_wrong_names_type_ints(self):
    type_signature = computation_types.NamedTupleType((tf.int32, tf.int32))
    comp = building_blocks.Data('data', type_signature)
    with self.assertRaises(TypeError):
      building_block_factory.create_named_tuple(comp, 'a')

  def test_raises_value_error_with_wrong_lengths(self):
    type_signature = computation_types.NamedTupleType((tf.int32, tf.int32))
    comp = building_blocks.Data('data', type_signature)
    with self.assertRaises(ValueError):
      building_block_factory.create_named_tuple(comp, ('a',))

  def test_creates_named_tuple_from_unamed_tuple(self):
    type_signature = computation_types.NamedTupleType((tf.int32, tf.int32))
    comp = building_blocks.Data('data', type_signature)
    named_comp = building_block_factory.create_named_tuple(comp, ('a', 'b'))
    expected_type_signature = computation_types.NamedTupleType(
        (('a', tf.int32), ('b', tf.int32)))
    self.assertEqual(named_comp.type_signature, expected_type_signature)

  def test_creates_named_tuple_from_named_tuple(self):
    type_signature = computation_types.NamedTupleType(
        (('a', tf.int32), ('b', tf.int32)))
    comp = building_blocks.Data('data', type_signature)
    named_comp = building_block_factory.create_named_tuple(comp, ('c', 'd'))
    expected_type_signature = computation_types.NamedTupleType(
        (('c', tf.int32), ('d', tf.int32)))
    self.assertEqual(named_comp.type_signature, expected_type_signature)


class CreateZipTest(absltest.TestCase):

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_zip(None)

  def test_zips_tuple_unnamed(self):
    data_1 = building_blocks.Data('a', tf.int32)
    data_2 = building_blocks.Data('b', tf.float32)
    data_3 = building_blocks.Data('c', tf.bool)
    tup_1 = building_blocks.Tuple((data_1, data_2, data_3))
    tup_2 = building_blocks.Tuple((tup_1, tup_1))
    comp = tup_2
    new_comp = building_block_factory.create_zip(comp)
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
    data_1 = building_blocks.Data('a', tf.int32)
    data_2 = building_blocks.Data('b', tf.float32)
    data_3 = building_blocks.Data('c', tf.bool)
    tup_1 = building_blocks.Tuple((('d', data_1), ('e', data_2), ('f', data_3)))
    tup_2 = building_blocks.Tuple((('g', tup_1), ('h', tup_1)))
    comp = tup_2
    new_comp = building_block_factory.create_zip(comp)
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
    ref = building_blocks.Reference('a', type_signature_2)
    comp = ref
    new_comp = building_block_factory.create_zip(comp)
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
      building_block_factory.create_tensorflow_to_broadcast_scalar(
          int, tf.TensorShape([]))

  def test_raises_list_for_shape(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_tensorflow_to_broadcast_scalar(
          tf.int32, [1, 1])

  def test_raises_partially_defined(self):
    with self.assertRaises(ValueError):
      building_block_factory.create_tensorflow_to_broadcast_scalar(
          tf.int32, tf.TensorShape([None, 1]))

  def test_constructs_identity_scalar_function(self):
    int_identity = building_block_factory.create_tensorflow_to_broadcast_scalar(
        tf.int32, tf.TensorShape([]))
    for k in range(5):
      result = test_utils.run_tensorflow(int_identity.proto, k)
      self.assertEqual(result, k)

  def test_broadcasts_ints_to_nonempty_shape(self):
    int_broadcast = building_block_factory.create_tensorflow_to_broadcast_scalar(
        tf.int32, tf.TensorShape([2, 2]))
    for k in range(5):
      self.assertTrue(
          np.array_equal(
              test_utils.run_tensorflow(int_broadcast.proto, k),
              np.array([[k, k], [k, k]])))

  def test_broadcasts_bools_to_nonempty_shape(self):
    int_broadcast = building_block_factory.create_tensorflow_to_broadcast_scalar(
        tf.bool, tf.TensorShape([2, 2]))
    self.assertTrue(
        np.array_equal(
            test_utils.run_tensorflow(int_broadcast.proto, True),
            np.array([[True, True], [True, True]])))


class CreateTensorFlowBinaryOpTest(absltest.TestCase):

  def test_raises_on_none_type(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_tensorflow_binary_operator(None, tf.add)

  def test_raises_non_callable_op(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_tensorflow_binary_operator(tf.int32, 1)

  def test_raises_on_federated_type(self):
    fed_type = computation_types.FederatedType(tf.int32,
                                               placement_literals.SERVER)
    with self.assertRaises(TypeError):
      building_block_factory.create_tensorflow_binary_operator(fed_type, tf.add)

  def test_raises_on_nested_sequence_type(self):
    hiding_sequence_type = computation_types.NamedTupleType(
        [computation_types.SequenceType(tf.int32)])
    with self.assertRaises(TypeError):
      building_block_factory.create_tensorflow_binary_operator(
          hiding_sequence_type, tf.add)

  def test_divide_integers(self):
    integer_division_func = building_block_factory.create_tensorflow_binary_operator(
        tf.int32, tf.divide)
    self.assertEqual(
        integer_division_func.type_signature,
        computation_types.FunctionType([tf.int32, tf.int32], tf.float64))
    result_1 = test_utils.run_tensorflow(integer_division_func.proto, [1, 1])
    self.assertEqual(result_1, 1)
    result_2 = test_utils.run_tensorflow(integer_division_func.proto, [1, 2])
    self.assertEqual(result_2, 0.5)
    result_3 = test_utils.run_tensorflow(integer_division_func.proto, [2, 1])
    self.assertEqual(result_3, 2)
    result_4 = test_utils.run_tensorflow(integer_division_func.proto, [1, 0])
    self.assertEqual(result_4, np.inf)

  def test_divide_unnamed_tuple(self):
    division_func = building_block_factory.create_tensorflow_binary_operator(
        [tf.int32, tf.float32], tf.divide)
    self.assertEqual(
        division_func.type_signature,
        computation_types.FunctionType(
            [[tf.int32, tf.float32], [tf.int32, tf.float32]],
            [tf.float64, tf.float32]))
    self.assertEqual(
        test_utils.run_tensorflow(division_func.proto, [[1, 0.], [1, 1.]])[0],
        1)
    self.assertEqual(
        test_utils.run_tensorflow(division_func.proto, [[1, 0.], [1, 1.]])[1],
        0.)

  def test_divide_named_tuple(self):
    integer_division_func = building_block_factory.create_tensorflow_binary_operator(
        [('a', tf.int32), ('b', tf.float32)], tf.divide)
    self.assertDictEqual(
        anonymous_tuple.to_odict(
            test_utils.run_tensorflow(integer_division_func.proto,
                                      [[1, 0.], [1, 1.]])), {
                                          'a': 1,
                                          'b': 0.
                                      })

  def test_multiply_integers(self):
    integer_multiplication_func = building_block_factory.create_tensorflow_binary_operator(
        tf.int32, tf.multiply)
    self.assertEqual(
        test_utils.run_tensorflow(integer_multiplication_func.proto, [1, 1]), 1)
    self.assertEqual(
        test_utils.run_tensorflow(integer_multiplication_func.proto, [1, 2]), 2)
    self.assertEqual(
        test_utils.run_tensorflow(integer_multiplication_func.proto, [2, 1]), 2)

  def test_multiply_named_tuple(self):
    integer_multiplication_func = building_block_factory.create_tensorflow_binary_operator(
        [('a', tf.int32), ('b', tf.float32)], tf.multiply)
    self.assertDictEqual(
        anonymous_tuple.to_odict(
            test_utils.run_tensorflow(integer_multiplication_func.proto,
                                      [[1, 0.], [1, 1.]])), {
                                          'a': 1,
                                          'b': 0.
                                      })
    self.assertDictEqual(
        anonymous_tuple.to_odict(
            test_utils.run_tensorflow(integer_multiplication_func.proto,
                                      [[2, 2.], [1, 1.]])), {
                                          'a': 2,
                                          'b': 2.
                                      })

  def test_add_integers(self):
    integer_add = building_block_factory.create_tensorflow_binary_operator(
        tf.int32, tf.add)
    result_1 = test_utils.run_tensorflow(integer_add.proto, [0, 0])
    self.assertEqual(result_1, 0)
    result_2 = test_utils.run_tensorflow(integer_add.proto, [1, 0])
    self.assertEqual(result_2, 1)
    result_3 = test_utils.run_tensorflow(integer_add.proto, [0, 1])
    self.assertEqual(result_3, 1)
    result_4 = test_utils.run_tensorflow(integer_add.proto, [1, 1])
    self.assertEqual(result_4, 2)


class CreateTensorFlowConstantTest(absltest.TestCase):

  def test_raises_on_none_type_spec(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_tensorflow_constant(None, 0)

  def test_raises_type_spec_federated_int(self):
    federated_int = computation_types.FederatedType(tf.int32,
                                                    placement_literals.SERVER)
    with self.assertRaisesRegex(TypeError, 'only nested tuples and tensors'):
      building_block_factory.create_tensorflow_constant(federated_int, 0)

  def test_raises_non_scalar_value(self):
    non_scalar_value = np.zeros([1])
    with self.assertRaisesRegex(TypeError, 'Must pass a scalar'):
      building_block_factory.create_tensorflow_constant(tf.int32,
                                                        non_scalar_value)

  def test_raises_float_passed_for_int(self):
    with self.assertRaisesRegex(TypeError, 'Only integers'):
      building_block_factory.create_tensorflow_constant(tf.int32, 1.)

  def test_constructs_integer_tensor_zero(self):
    tensor_zero = building_block_factory.create_tensorflow_constant(
        computation_types.TensorType(tf.int32, [2, 2]), 0)
    self.assertIsInstance(tensor_zero, building_blocks.Call)
    self.assertTrue(
        np.array_equal(
            test_utils.run_tensorflow(tensor_zero.function.proto),
            np.zeros([2, 2], dtype=np.int32)))

  def test_constructs_tensor_zero_with_unknown_shape(self):
    tensor_zero = building_block_factory.create_tensorflow_constant(
        computation_types.TensorType(tf.int32, [None, 2]), 0)
    self.assertIsInstance(tensor_zero, building_blocks.Call)
    self.assertEqual(str(tensor_zero.type_signature), 'int32[?,2]')

  def test_constructs_float_tensor_one(self):
    tensor_one = building_block_factory.create_tensorflow_constant(
        computation_types.TensorType(tf.float32, [2, 2]), 1.)
    self.assertIsInstance(tensor_one, building_blocks.Call)
    self.assertTrue(
        np.array_equal(
            test_utils.run_tensorflow(tensor_one.function.proto),
            np.ones([2, 2], dtype=np.float32)))

  def test_constructs_unnamed_tuple_of_float_tensor_ones(self):
    tuple_type = computation_types.NamedTupleType(
        [computation_types.TensorType(tf.float32, [2, 2])] * 2)
    tuple_of_ones = building_block_factory.create_tensorflow_constant(
        tuple_type, 1.)
    self.assertEqual(tuple_of_ones.type_signature, tuple_type)
    self.assertIsInstance(tuple_of_ones, building_blocks.Call)
    result = test_utils.run_tensorflow(tuple_of_ones.function.proto)
    self.assertLen(result, 2)
    self.assertTrue(
        np.array_equal(result[0], np.ones([2, 2], dtype=np.float32)))
    self.assertTrue(
        np.array_equal(result[1], np.ones([2, 2], dtype=np.float32)))

  def test_constructs_named_tuple_of_float_tensor_ones(self):
    tuple_type = computation_types.NamedTupleType([
        ('a', computation_types.TensorType(tf.float32, [2, 2])),
        ('b', computation_types.TensorType(tf.float32, [2, 2]))
    ])
    tuple_of_ones = building_block_factory.create_tensorflow_constant(
        tuple_type, 1.)
    self.assertEqual(tuple_of_ones.type_signature, tuple_type)
    self.assertIsInstance(tuple_of_ones, building_blocks.Call)
    result = test_utils.run_tensorflow(tuple_of_ones.function.proto)
    self.assertLen(result, 2)
    self.assertTrue(np.array_equal(result.a, np.ones([2, 2], dtype=np.float32)))
    self.assertTrue(np.array_equal(result.b, np.ones([2, 2], dtype=np.float32)))

  def test_constructs_nested_named_tuple_of_float_tensor_ones(self):
    tuple_type = computation_types.NamedTupleType([[
        ('a', computation_types.TensorType(tf.float32, [2, 2])),
        ('b', computation_types.TensorType(tf.float32, [2, 2]))
    ]])
    tuple_of_ones = building_block_factory.create_tensorflow_constant(
        tuple_type, 1.)
    self.assertEqual(tuple_of_ones.type_signature, tuple_type)
    self.assertIsInstance(tuple_of_ones, building_blocks.Call)
    result = test_utils.run_tensorflow(tuple_of_ones.function.proto)
    self.assertLen(result, 1)
    self.assertTrue(
        np.array_equal(result[0].a, np.ones([2, 2], dtype=np.float32)))
    self.assertTrue(
        np.array_equal(result[0].b, np.ones([2, 2], dtype=np.float32)))

  def test_constructs_nested_named_tuple_of_int_and_float_tensor_ones(self):
    tuple_type = computation_types.NamedTupleType([[
        ('a', computation_types.TensorType(tf.int32, [2, 2])),
        ('b', computation_types.TensorType(tf.float32, [2, 2]))
    ]])
    tuple_of_ones = building_block_factory.create_tensorflow_constant(
        tuple_type, 1)
    self.assertEqual(tuple_of_ones.type_signature, tuple_type)
    self.assertIsInstance(tuple_of_ones, building_blocks.Call)
    result = test_utils.run_tensorflow(tuple_of_ones.function.proto)
    self.assertLen(result, 1)
    self.assertTrue(
        np.array_equal(result[0].a, np.ones([2, 2], dtype=np.int32)))
    self.assertTrue(
        np.array_equal(result[0].b, np.ones([2, 2], dtype=np.float32)))


class BinaryOperatorTest(absltest.TestCase):

  def test_apply_op_raises_on_none(self):
    with self.assertRaisesRegex(TypeError, 'ComputationBuildingBlock'):
      building_block_factory.apply_binary_operator_with_upcast(
          None, tf.multiply)

  def test_construct_op_raises_on_none_operator(self):
    with self.assertRaisesRegex(TypeError, 'found non-callable'):
      building_block_factory.create_binary_operator_with_upcast(tf.int32, None)

  def test_raises_incompatible_tuple_and_tensor(self):
    bad_type_ref = building_blocks.Reference(
        'x',
        computation_types.FederatedType([[tf.int32, tf.int32], tf.float32],
                                        placement_literals.CLIENTS))
    with self.assertRaisesRegex(TypeError, 'incompatible with upcasted'):
      building_block_factory.apply_binary_operator_with_upcast(
          bad_type_ref, tf.multiply)
    with self.assertRaisesRegex(TypeError, 'incompatible with upcasted'):
      building_block_factory.create_binary_operator_with_upcast(
          bad_type_ref.type_signature.member, tf.multiply)

  def test_raises_non_callable_op(self):
    bad_type_ref = building_blocks.Reference('x', [tf.float32, tf.float32])
    with self.assertRaisesRegex(TypeError, 'non-callable'):
      building_block_factory.apply_binary_operator_with_upcast(
          bad_type_ref, tf.constant(0))
    with self.assertRaisesRegex(TypeError, 'non-callable'):
      building_block_factory.create_binary_operator_with_upcast(
          bad_type_ref, tf.constant(0))

  def test_raises_tuple_and_nonscalar_tensor(self):
    bad_type_ref = building_blocks.Reference(
        'x',
        computation_types.FederatedType(
            [[tf.int32, tf.int32],
             computation_types.TensorType(tf.float32, [2])],
            placement_literals.CLIENTS))
    with self.assertRaisesRegex(TypeError, 'incompatible with upcasted'):
      building_block_factory.apply_binary_operator_with_upcast(
          bad_type_ref, tf.multiply)
    with self.assertRaisesRegex(TypeError, 'incompatible with upcasted'):
      building_block_factory.create_binary_operator_with_upcast(
          bad_type_ref.type_signature.member, tf.multiply)

  def test_raises_tuple_scalar_multiplied_by_nonscalar(self):
    bad_type_ref = building_blocks.Reference(
        'x', [tf.int32, computation_types.TensorType(tf.float32, [2])])
    with self.assertRaisesRegex(TypeError, 'incompatible with upcasted'):
      building_block_factory.apply_binary_operator_with_upcast(
          bad_type_ref, tf.multiply)
    with self.assertRaisesRegex(TypeError, 'incompatible with upcasted'):
      building_block_factory.create_binary_operator_with_upcast(
          bad_type_ref.type_signature, tf.multiply)

  def test_construct_generic_raises_federated_type(self):
    bad_type = computation_types.FederatedType(
        [[tf.int32, tf.int32],
         computation_types.TensorType(tf.float32, [2])],
        placement_literals.CLIENTS)
    with self.assertRaisesRegex(TypeError, 'argument that is not a two-tuple'):
      building_block_factory.create_binary_operator_with_upcast(
          bad_type, tf.multiply)

  def test_apply_integer_type_signature(self):
    ref = building_blocks.Reference('x', [tf.int32, tf.int32])
    multiplied = building_block_factory.apply_binary_operator_with_upcast(
        ref, tf.multiply)
    self.assertEqual(multiplied.type_signature,
                     computation_types.to_type(tf.int32))

  def test_construct_integer_type_signature(self):
    ref = building_blocks.Reference('x', [tf.int32, tf.int32])
    multiplier = building_block_factory.create_binary_operator_with_upcast(
        ref.type_signature, tf.multiply)
    self.assertEqual(
        multiplier.type_signature,
        type_factory.binary_op(computation_types.to_type(tf.int32)))

  def test_multiply_federated_integer_type_signature(self):
    ref = building_blocks.Reference(
        'x',
        computation_types.FederatedType([tf.int32, tf.int32],
                                        placement_literals.CLIENTS))
    multiplied = building_block_factory.apply_binary_operator_with_upcast(
        ref, tf.multiply)
    self.assertEqual(
        multiplied.type_signature,
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))

  def test_divide_federated_float_type_signature(self):
    ref = building_blocks.Reference(
        'x',
        computation_types.FederatedType([tf.float32, tf.float32],
                                        placement_literals.CLIENTS))
    multiplied = building_block_factory.apply_binary_operator_with_upcast(
        ref, tf.multiply)
    self.assertEqual(
        multiplied.type_signature,
        computation_types.FederatedType(tf.float32, placement_literals.CLIENTS))

  def test_multiply_federated_unnamed_tuple_type_signature(self):
    ref = building_blocks.Reference(
        'x',
        computation_types.FederatedType(
            [[tf.int32, tf.float32], [tf.int32, tf.float32]],
            placement_literals.CLIENTS))
    multiplied = building_block_factory.apply_binary_operator_with_upcast(
        ref, tf.multiply)
    self.assertEqual(
        multiplied.type_signature,
        computation_types.FederatedType([tf.int32, tf.float32],
                                        placement_literals.CLIENTS))

  def test_multiply_federated_named_tuple_type_signature(self):
    ref = building_blocks.Reference(
        'x',
        computation_types.FederatedType(
            [[('a', tf.int32),
              ('b', tf.float32)], [('a', tf.int32), ('b', tf.float32)]],
            placement_literals.CLIENTS))
    multiplied = building_block_factory.apply_binary_operator_with_upcast(
        ref, tf.multiply)
    self.assertEqual(
        multiplied.type_signature,
        computation_types.FederatedType([('a', tf.int32), ('b', tf.float32)],
                                        placement_literals.CLIENTS))

  def test_divide_federated_named_tuple_type_signature(self):
    ref = building_blocks.Reference(
        'x',
        computation_types.FederatedType(
            [[('a', tf.int32),
              ('b', tf.float32)], [('a', tf.int32), ('b', tf.float32)]],
            placement_literals.CLIENTS))
    multiplied = building_block_factory.apply_binary_operator_with_upcast(
        ref, tf.divide)
    self.assertEqual(
        multiplied.type_signature,
        computation_types.FederatedType([('a', tf.float64), ('b', tf.float32)],
                                        placement_literals.CLIENTS))

  def test_multiply_federated_named_tuple_with_scalar_type_signature(self):
    ref = building_blocks.Reference(
        'x',
        computation_types.FederatedType([[('a', tf.float32),
                                          ('b', tf.float32)], tf.float32],
                                        placement_literals.CLIENTS))
    multiplied = building_block_factory.apply_binary_operator_with_upcast(
        ref, tf.multiply)
    self.assertEqual(
        multiplied.type_signature,
        computation_types.FederatedType([('a', tf.float32), ('b', tf.float32)],
                                        placement_literals.CLIENTS))

  def test_multiply_named_tuple_with_scalar_type_signature(self):
    ref = building_blocks.Reference('x', [[('a', tf.float32),
                                           ('b', tf.float32)], tf.float32])
    multiplied = building_block_factory.apply_binary_operator_with_upcast(
        ref, tf.multiply)
    self.assertEqual(
        multiplied.type_signature,
        computation_types.NamedTupleType([('a', tf.float32),
                                          ('b', tf.float32)]))

  def test_construct_multiply_op_named_tuple_with_scalar_type_signature(self):
    type_spec = computation_types.to_type([[('a', tf.float32),
                                            ('b', tf.float32)], tf.float32])
    multiplier = building_block_factory.create_binary_operator_with_upcast(
        type_spec, tf.multiply)
    expected_function_type = computation_types.FunctionType(
        type_spec, type_spec[0])
    self.assertEqual(multiplier.type_signature, expected_function_type)

  def test_construct_divide_op_named_tuple_with_scalar_type_signature(self):
    type_spec = computation_types.to_type([[('a', tf.float32),
                                            ('b', tf.float32)], tf.float32])
    multiplier = building_block_factory.create_binary_operator_with_upcast(
        type_spec, tf.divide)
    expected_function_type = computation_types.FunctionType(
        type_spec, type_spec[0])
    self.assertEqual(multiplier.type_signature, expected_function_type)

  def test_divide_federated_named_tuple_with_scalar_type_signature(self):
    ref = building_blocks.Reference(
        'x',
        computation_types.FederatedType([[('a', tf.float32),
                                          ('b', tf.float32)], tf.float32],
                                        placement_literals.CLIENTS))
    multiplied = building_block_factory.apply_binary_operator_with_upcast(
        ref, tf.divide)
    self.assertEqual(
        multiplied.type_signature,
        computation_types.FederatedType([('a', tf.float32), ('b', tf.float32)],
                                        placement_literals.CLIENTS))


if __name__ == '__main__':
  absltest.main()
