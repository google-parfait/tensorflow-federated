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

from tensorflow_federated.python.common_libs import golden
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import test_utils
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis


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


class CreateFederatedGetitemCompTest(parameterized.TestCase):

  def test_raises_type_error_on_none(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_getitem_comp(None, 0)

  @parameterized.named_parameters(('clients', placements.CLIENTS),
                                  ('server', placements.SERVER))
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

  @parameterized.named_parameters(('clients', placements.CLIENTS),
                                  ('server', placements.SERVER))
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
        'test', computation_types.StructType([('a', tf.int32), ('b', tf.bool)]))
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
      ('clients', placements.CLIENTS),
      ('server', placements.SERVER),
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
      type_analysis.check_federated_type(
          name_a.type_signature, placement=placement)
    except TypeError:
      self.fail(
          'Function \'check_federated_type\' raised TypeError unexpectedly.')
    try:
      type_analysis.check_federated_type(
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
      ('clients', placements.CLIENTS),
      ('server', placements.SERVER),
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
      type_analysis.check_federated_type(
          idx_0.type_signature, placement=placement)
    except TypeError:
      self.fail(
          'Function \'check_federated_type\' raised TypeError unexpectedly.')
    try:
      type_analysis.check_federated_type(
          idx_1.type_signature, placement=placement)
    except TypeError:
      self.fail(
          'Function \'check_federated_type\' raised TypeError unexpectedly.')
    flipped = building_block_factory.create_federated_getitem_call(
        federated_comp_named, slice(None, None, -1))
    self.assertIsInstance(flipped.type_signature,
                          computation_types.FederatedType)
    self.assertEqual(str(flipped.type_signature.member), '<b=bool,a=int32>')
    try:
      type_analysis.check_federated_type(
          flipped.type_signature, placement=placement)
    except TypeError:
      self.fail(
          'Function \'check_federated_type\' raised TypeError unexpectedly.')

  @parameterized.named_parameters(
      ('clients', placements.CLIENTS),
      ('server', placements.SERVER),
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
      type_analysis.check_federated_type(
          unnamed_idx_0.type_signature, placement=placement)
    except TypeError:
      self.fail(
          'Function \'check_federated_type\' raised TypeError unexpectedly.')
    try:
      type_analysis.check_federated_type(
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
      type_analysis.check_federated_type(
          unnamed_flipped.type_signature, placement=placement)
    except TypeError:
      self.fail(
          'Function \'check_federated_type\' raised TypeError unexpectedly.')


class CreateComputationAppendingTest(absltest.TestCase):

  def test_raises_type_error_with_none_comp1(self):
    comp2 = building_blocks.Data('y', tf.int32)
    with self.assertRaises(TypeError):
      building_block_factory.create_computation_appending(None, comp2)

  def test_raises_type_error_with_none_comp2(self):
    value = building_blocks.Data('x', tf.int32)
    comp1 = building_blocks.Struct([value, value])
    with self.assertRaises(TypeError):
      building_block_factory.create_computation_appending(comp1, None)

  def test_raises_type_error_with_comp1_bad_type(self):
    comp1 = building_blocks.Data('x', tf.int32)
    comp2 = building_blocks.Data('y', tf.int32)
    with self.assertRaises(TypeError):
      building_block_factory.create_computation_appending(comp1, comp2)

  def test_returns_comp_unnamed(self):
    value = building_blocks.Data('x', tf.int32)
    comp1 = building_blocks.Struct([value, value])
    comp2 = building_blocks.Data('y', tf.int32)
    comp = building_block_factory.create_computation_appending(comp1, comp2)
    self.assertEqual(
        comp.compact_representation(),
        '(let comps=<<x,x>,y> in <comps[0][0],comps[0][1],comps[1]>)')
    self.assertEqual(str(comp.type_signature), '<int32,int32,int32>')

  def test_returns_comp_named(self):
    value = building_blocks.Data('x', tf.int32)
    comp1 = building_blocks.Struct((
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
    accumulate_type = computation_types.StructType((tf.int32, tf.int32))
    accumulate_result = building_blocks.Data('a', tf.int32)
    accumulate = building_blocks.Lambda('x', accumulate_type, accumulate_result)
    merge_type = computation_types.StructType((tf.int32, tf.int32))
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
    accumulate_type = computation_types.StructType((tf.int32, tf.int32))
    accumulate_result = building_blocks.Data('a', tf.int32)
    accumulate = building_blocks.Lambda('x', accumulate_type, accumulate_result)
    merge_type = computation_types.StructType((tf.int32, tf.int32))
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
    merge_type = computation_types.StructType((tf.int32, tf.int32))
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
    accumulate_type = computation_types.StructType((tf.int32, tf.int32))
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
    accumulate_type = computation_types.StructType((tf.int32, tf.int32))
    accumulate_result = building_blocks.Data('a', tf.int32)
    accumulate = building_blocks.Lambda('x', accumulate_type, accumulate_result)
    merge_type = computation_types.StructType((tf.int32, tf.int32))
    merge_result = building_blocks.Data('m', tf.int32)
    merge = building_blocks.Lambda('x', merge_type, merge_result)
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_aggregate(value, zero, accumulate,
                                                        merge, None)

  def test_returns_federated_aggregate(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    zero = building_blocks.Data('z', tf.int32)
    accumulate_type = computation_types.StructType((tf.int32, tf.int32))
    accumulate_result = building_blocks.Data('a', tf.int32)
    accumulate = building_blocks.Lambda('x', accumulate_type, accumulate_result)
    merge_type = computation_types.StructType((tf.int32, tf.int32))
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


class CreateFederatedEvalTest(absltest.TestCase):

  def assert_type_error(self, fn, placement):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_eval(fn, placement)

  def test_raises_type_error_with_none_fn(self):
    self.assert_type_error(None, placements.CLIENTS)

  def test_raises_type_error_with_nonfunctional_fn(self):
    fn = building_blocks.Data('y', tf.int32)
    self.assert_type_error(fn, placements.CLIENTS)

  def test_returns_federated_eval(self):
    fn = building_blocks.Data('y',
                              computation_types.FunctionType(None, tf.int32))
    comp = building_block_factory.create_federated_eval(fn, placements.CLIENTS)
    self.assertEqual(comp.compact_representation(),
                     'federated_eval_at_clients(y)')
    self.assertEqual(str(comp.type_signature), '{int32}@CLIENTS')


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


class CreateFederatedSecureSumTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    bitwidth_type = computation_types.TensorType(tf.int32)
    bitwidth = building_block_factory.create_compiled_identity(
        bitwidth_type, name='b')

    with self.assertRaises(TypeError):
      building_block_factory.create_federated_secure_sum(None, bitwidth)

  def test_raises_type_error_with_none_bitwidth(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)

    with self.assertRaises(TypeError):
      building_block_factory.create_federated_secure_sum(value, None)

  def test_returns_federated_sum(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    bitwidth_type = computation_types.TensorType(tf.int32)
    bitwidth = building_block_factory.create_tensorflow_constant(
        bitwidth_type, 8, 'b')
    comp = building_block_factory.create_federated_secure_sum(value, bitwidth)
    self.assertEqual(comp.compact_representation(),
                     'federated_secure_sum(<v,comp#b()>)')
    self.assertEqual(comp.type_signature.compact_representation(),
                     'int32@SERVER')


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
    type_signature = computation_types.StructType((('a', tf.int32),))
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
    golden.check_string('tuple_federated_map_with_two_values_unnamed.expected',
                        comp.formatted_representation())
    self.assertEqual(
        str(comp.type_signature), '<{int32}@CLIENTS,{int32}@CLIENTS>')

  def test_returns_tuple_federated_map_with_two_values_named(self):
    type_signature = computation_types.StructType(
        (('a', tf.int32), ('b', tf.int32)))
    value_type = computation_types.FederatedType(type_signature,
                                                 placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
    golden.check_string('tuple_federated_map_with_two_values_named.expected',
                        comp.formatted_representation())
    self.assertEqual(
        str(comp.type_signature), '<a={int32}@CLIENTS,b={int32}@CLIENTS>')

  def test_returns_tuple_federated_map_with_two_values_different_typed(self):
    value_type = computation_types.FederatedType((tf.int32, tf.bool),
                                                 placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
    golden.check_string(
        'tuple_federated_map_with_two_values_different_typed.expected',
        comp.formatted_representation())
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
    type_signature = computation_types.StructType((('a', tf.int32),))
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
    golden.check_string(
        'tuple_federated_apply_with_two_values_unnamed.expected',
        comp.formatted_representation())
    self.assertEqual(str(comp.type_signature), '<int32@SERVER,int32@SERVER>')

  def test_returns_tuple_federated_apply_with_two_values_named(self):
    type_signature = computation_types.StructType(
        (('a', tf.int32), ('b', tf.int32)))
    value_type = computation_types.FederatedType(type_signature,
                                                 placements.SERVER)
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
    golden.check_string('tuple_federated_apply_with_two_values_named.expected',
                        comp.formatted_representation())
    self.assertEqual(
        str(comp.type_signature), '<a=int32@SERVER,b=int32@SERVER>')

  def test_returns_tuple_federated_apply_with_two_values_different_typed(self):
    value_type = computation_types.FederatedType((tf.int32, tf.bool),
                                                 placements.SERVER)
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_unzip(value)
    golden.check_string(
        'tuple_federated_apply_with_two_values_different_typed.expected',
        comp.formatted_representation())
    self.assertEqual(str(comp.type_signature), '<int32@SERVER,bool@SERVER>')


class CreateFederatedValueTest(absltest.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_value(None, placements.CLIENTS)

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
        value, placements.CLIENTS)
    self.assertEqual(comp.compact_representation(),
                     'federated_value_at_clients(v)')
    self.assertEqual(str(comp.type_signature), 'int32@CLIENTS')

  def test_returns_federated_value_at_server(self):
    value = building_blocks.Data('v', tf.int32)
    comp = building_block_factory.create_federated_value(
        value, placements.SERVER)
    self.assertEqual(comp.compact_representation(),
                     'federated_value_at_server(v)')
    self.assertEqual(str(comp.type_signature), 'int32@SERVER')


class CreateFederatedZipTest(parameterized.TestCase):

  def test_raises_type_error_with_none_value(self):
    with self.assertRaisesRegex(TypeError, 'found NoneType'):
      building_block_factory.create_federated_zip(None)

  def test_raises_type_error_with_empty_value(self):
    value_type = computation_types.StructType([])
    value = building_blocks.Data('v', value_type)
    with self.assertRaisesRegex(TypeError, 'at least one FederatedType'):
      building_block_factory.create_federated_zip(value)

  def test_returns_federated_map_with_one_value_unnamed(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    value_type = computation_types.StructType((type_signature,))
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
    self.assertEqual(comp.compact_representation(),
                     'federated_map(<(arg -> <arg>),v[0]>)')
    self.assertEqual(str(comp.type_signature), '{<int32>}@CLIENTS')

  def test_returns_federated_map_with_one_value_unnamed_tuple(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    tup = building_blocks.Struct((value,))
    comp = building_block_factory.create_federated_zip(tup)
    self.assertEqual(comp.compact_representation(),
                     'federated_map(<(arg -> <arg>),<v>[0]>)')
    self.assertEqual(str(comp.type_signature), '{<int32>}@CLIENTS')

  def test_returns_federated_map_with_one_value_named(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    value_type = computation_types.StructType((('a', type_signature),))
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
    self.assertEqual(comp.compact_representation(),
                     'federated_map(<(arg -> <a=arg>),v[0]>)')
    self.assertEqual(str(comp.type_signature), '{<a=int32>}@CLIENTS')

  def test_returns_federated_map_with_one_value_named_tuple(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    tup = building_blocks.Struct((('a', value),))
    comp = building_block_factory.create_federated_zip(tup)
    self.assertEqual(comp.compact_representation(),
                     'federated_map(<(arg -> <a=arg>),<a=v>[0]>)')
    self.assertEqual(str(comp.type_signature), '{<a=int32>}@CLIENTS')

  def test_returns_federated_zip_at_clients_with_two_values_unnamed(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    value_type = computation_types.StructType((type_signature, type_signature))
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
    self.assertEqual(
        comp.formatted_representation(),
        'federated_zip_at_clients(v)'
    )
    self.assertEqual(str(comp.type_signature), '{<int32,int32>}@CLIENTS')

  def test_returns_federated_zip_at_clients_with_two_values_unnamed_tuple(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    tup = building_blocks.Struct((value, value))
    comp = building_block_factory.create_federated_zip(tup)
    golden.check_string(
        'federated_zip_at_clients_with_two_values_unnamed_tuple.expected',
        comp.formatted_representation())
    self.assertEqual(str(comp.type_signature), '{<int32,int32>}@CLIENTS')

  def test_returns_federated_zip_at_clients_with_two_values_named(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    value_type = computation_types.StructType(
        (('a', type_signature), ('b', type_signature)))
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
    golden.check_string(
        'federated_zip_at_clients_with_two_values_named.expected',
        comp.formatted_representation())
    self.assertEqual(str(comp.type_signature), '{<a=int32,b=int32>}@CLIENTS')

  def test_returns_federated_zip_at_clients_with_two_values_named_tuple(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    tup = building_blocks.Struct((('a', value), ('b', value)))
    comp = building_block_factory.create_federated_zip(tup)
    golden.check_string(
        'federated_zip_at_clients_with_two_values_named_tuple.expected',
        comp.formatted_representation())
    self.assertEqual(str(comp.type_signature), '{<a=int32,b=int32>}@CLIENTS')

  def test_returns_federated_zip_at_clients_with_three_values_unnamed(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    value_type = computation_types.StructType(
        (type_signature, type_signature, type_signature))
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
    golden.check_string(
        'federated_zip_at_clients_with_three_values_unnamed.expected',
        comp.formatted_representation())
    self.assertEqual(str(comp.type_signature), '{<int32,int32,int32>}@CLIENTS')

  def test_returns_federated_zip_at_clients_with_three_values_unnamed_tuple(
      self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    tup = building_blocks.Struct((value, value, value))
    comp = building_block_factory.create_federated_zip(tup)
    golden.check_string(
        'federated_zip_at_clients_with_three_values_unnamed_tuple.expected',
        comp.formatted_representation())
    self.assertEqual(str(comp.type_signature), '{<int32,int32,int32>}@CLIENTS')

  def test_returns_federated_zip_at_clients_with_three_values_named(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.CLIENTS)
    value_type = computation_types.StructType((
        ('a', type_signature),
        ('b', type_signature),
        ('c', type_signature),
    ))
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
    golden.check_string(
        'federated_zip_at_clients_with_three_values_named.expected',
        comp.formatted_representation())
    self.assertEqual(
        str(comp.type_signature), '{<a=int32,b=int32,c=int32>}@CLIENTS')

  def test_returns_federated_zip_at_clients_with_three_values_named_tuple(self):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    tup = building_blocks.Struct((
        ('a', value),
        ('b', value),
        ('c', value),
    ))
    comp = building_block_factory.create_federated_zip(tup)
    golden.check_string(
        'federated_zip_at_clients_with_three_values_named_tuple.expected',
        comp.formatted_representation())
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
    value_type = computation_types.StructType(
        (type_signature1, type_signature2, type_signature3))
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
    golden.check_string(
        'federated_zip_at_clients_with_three_values_different_typed.expected',
        comp.formatted_representation())
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
    tup = building_blocks.Struct((value1, value2, value3))
    comp = building_block_factory.create_federated_zip(tup)
    golden.check_string(
        'federated_zip_at_clients_with_three_values_different_typed_tuple.expected',
        comp.formatted_representation())
    self.assertEqual(str(comp.type_signature), '{<int32,float32,bool>}@CLIENTS')

  @parameterized.named_parameters(('one_element', 1), ('two_elements', 2),
                                  ('three_elements', 3))
  def test_returns_federated_zip_with_container_type(self, num_elements):
    value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value = building_blocks.Data('v', value_type)
    tup = building_blocks.Struct([value] * num_elements, list)
    comp = building_block_factory.create_federated_zip(tup)
    self.assertEqual(
        repr(comp.type_signature),
        repr(
            computation_types.FederatedType(
                computation_types.StructWithPythonType(
                    [tf.int32] * num_elements, list), placements.CLIENTS)))

  def test_returns_federated_apply_with_one_value_unnamed(self):
    value_type = computation_types.FederatedType(tf.int32, placements.SERVER)
    value = building_blocks.Data('v', value_type)
    tup = building_blocks.Struct((value,))
    comp = building_block_factory.create_federated_zip(tup)
    self.assertEqual(comp.compact_representation(),
                     'federated_apply(<(arg -> <arg>),<v>[0]>)')
    self.assertEqual(str(comp.type_signature), '<int32>@SERVER')

  def test_returns_federated_apply_with_one_value_named(self):
    value_type = computation_types.FederatedType(tf.int32, placements.SERVER)
    value = building_blocks.Data('v', value_type)
    tup = building_blocks.Struct((('a', value),))
    comp = building_block_factory.create_federated_zip(tup)
    self.assertEqual(comp.compact_representation(),
                     'federated_apply(<(arg -> <a=arg>),<a=v>[0]>)')
    self.assertEqual(str(comp.type_signature), '<a=int32>@SERVER')

  def test_returns_federated_zip_at_server_with_two_values_unnamed(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.SERVER)
    value_type = computation_types.StructType((type_signature, type_signature))
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
    self.assertEqual(
        comp.formatted_representation(),
        'federated_zip_at_server(v)'
    )
    self.assertEqual(str(comp.type_signature), '<int32,int32>@SERVER')

  def test_returns_federated_zip_at_server_with_two_values_named(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.SERVER)
    value_type = computation_types.StructType(
        (('a', type_signature), ('b', type_signature)))
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
    golden.check_string(
        'federated_zip_at_server_with_two_values_named.expected',
        comp.formatted_representation())
    self.assertEqual(str(comp.type_signature), '<a=int32,b=int32>@SERVER')

  def test_returns_federated_zip_at_server_with_three_values_unnamed(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.SERVER)
    value_type = computation_types.StructType(
        (type_signature, type_signature, type_signature))
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
    golden.check_string(
        'federated_zip_at_server_with_three_values_unnamed.expected',
        comp.formatted_representation())
    self.assertEqual(str(comp.type_signature), '<int32,int32,int32>@SERVER')

  def test_returns_federated_zip_at_server_with_three_values_named(self):
    type_signature = computation_types.FederatedType(tf.int32,
                                                     placements.SERVER)
    value_type = computation_types.StructType((
        ('a', type_signature),
        ('b', type_signature),
        ('c', type_signature),
    ))
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
    golden.check_string(
        'federated_zip_at_server_with_three_values_named.expected',
        comp.formatted_representation())
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
    value_type = computation_types.StructType(
        (type_signature1, type_signature2, type_signature3))
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
    golden.check_string(
        'federated_zip_at_server_with_three_values_different_typed.expected',
        comp.formatted_representation())
    self.assertEqual(str(comp.type_signature), '<int32,float32,bool>@SERVER')

  def test_wide_zip_creates_minimum_depth_binary_tree(self):
    server_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    value_type = computation_types.StructType([server_type for _ in range(8)])
    value = building_blocks.Data('v', value_type)
    comp = building_block_factory.create_federated_zip(value)
    golden.check_string('wide_zip_creates_minimum_depth_binary_tree.expected',
                        comp.formatted_representation())

  def test_flat_raises_type_error_with_inconsistent_placement(self):
    client_type = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=True)
    server_type = computation_types.FederatedType(
        tf.int32, placements.SERVER, all_equal=True)
    value_type = computation_types.StructType([('a', client_type),
                                               ('b', server_type)])
    value = building_blocks.Data('v', value_type)
    self.assertEqual(value.type_signature.compact_representation(),
                     '<a=int32@CLIENTS,b=int32@SERVER>')
    with self.assertRaisesRegex(TypeError, 'same placement'):
      building_block_factory.create_federated_zip(value)

  def test_nested_raises_type_error_with_inconsistent_placement(self):
    client_type = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=True)
    server_type = computation_types.FederatedType(
        tf.int32, placements.SERVER, all_equal=True)
    tuple_type = computation_types.StructType([('c', server_type),
                                               ('d', server_type)])
    value_type = computation_types.StructType([('a', client_type),
                                               ('b', tuple_type)])
    value = building_blocks.Data('v', value_type)
    self.assertEqual(value.type_signature.compact_representation(),
                     '<a=int32@CLIENTS,b=<c=int32@SERVER,d=int32@SERVER>>')
    with self.assertRaisesRegex(TypeError, 'same placement'):
      building_block_factory.create_federated_zip(value)

  def test_flat_raises_type_error_with_unplaced(self):
    client_type = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=True)
    value_type = computation_types.StructType([('a', client_type),
                                               ('b', tf.int32)])
    value = building_blocks.Data('v', value_type)
    self.assertEqual(value.type_signature.compact_representation(),
                     '<a=int32@CLIENTS,b=int32>')
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_zip(value)

  def test_nested_raises_type_error_with_unplaced(self):
    client_type = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=True)
    tuple_type = computation_types.StructType([('c', tf.int32),
                                               ('d', tf.int32)])
    value_type = computation_types.StructType([('a', client_type),
                                               ('b', tuple_type)])
    value = building_blocks.Data('v', value_type)
    self.assertEqual(value.type_signature.compact_representation(),
                     '<a=int32@CLIENTS,b=<c=int32,d=int32>>')
    with self.assertRaises(TypeError):
      building_block_factory.create_federated_zip(value)

  def test_nested_returns_federated_zip_at_clients(self):
    int_type = computation_types.FederatedType(
        tf.int32, placements.CLIENTS, all_equal=True)
    tuple_type = computation_types.StructType([('c', int_type),
                                               ('d', int_type)])
    value_type = computation_types.StructType([('a', int_type),
                                               ('b', tuple_type)])
    value = building_blocks.Data('v', value_type)
    self.assertEqual(value.type_signature.compact_representation(),
                     '<a=int32@CLIENTS,b=<c=int32@CLIENTS,d=int32@CLIENTS>>')

    comp = building_block_factory.create_federated_zip(value)

    self.assertEqual(
        str(comp.type_signature), '{<a=int32,b=<c=int32,d=int32>>}@CLIENTS')
    golden.check_string('nested_returns_federated_zip_at_clients.expected',
                        comp.formatted_representation())

  def test_nested_returns_federated_zip_at_server(self):
    value_type = computation_types.StructType([
        ('a',
         computation_types.StructType([
             ('b',
              computation_types.FederatedType(
                  computation_types.StructType([('c', tf.int32)]),
                  placements.SERVER,
                  all_equal=True))
         ]))
    ])
    value = building_blocks.Data('v', value_type)
    self.assertEqual(value.type_signature.compact_representation(),
                     '<a=<b=<c=int32>@SERVER>>')

    comp = building_block_factory.create_federated_zip(value)

    self.assertEqual(str(comp.type_signature), '<a=<b=<c=int32>>>@SERVER')
    golden.check_string('nested_returns_federated_zip_at_server.expected',
                        comp.formatted_representation())


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
    tensor_type = computation_types.TensorType(tf.float32, [2, 2])
    tuple_type = computation_types.StructType((tensor_type, tensor_type))
    tuple_zero = building_block_factory.create_generic_constant(tuple_type, 0)
    self.assertEqual(tuple_zero.type_signature, tuple_type)
    self.assertIsInstance(tuple_zero, building_blocks.Call)
    result = test_utils.run_tensorflow(tuple_zero.function.proto)
    self.assertLen(result, 2)
    self.assertTrue(np.array_equal(result[0], np.zeros([2, 2])))
    self.assertTrue(np.array_equal(result[1], np.zeros([2, 2])))

  def test_create_named_tuple_one(self):
    tensor_type = computation_types.TensorType(tf.float32, [2, 2])
    tuple_type = computation_types.StructType([('a', tensor_type),
                                               ('b', tensor_type)])

    tuple_zero = building_block_factory.create_generic_constant(tuple_type, 1)

    self.assertEqual(tuple_zero.type_signature, tuple_type)
    self.assertIsInstance(tuple_zero, building_blocks.Call)
    result = test_utils.run_tensorflow(tuple_zero.function.proto)
    self.assertLen(result, 2)
    self.assertTrue(np.array_equal(result.a, np.ones([2, 2])))
    self.assertTrue(np.array_equal(result.b, np.ones([2, 2])))

  def test_create_federated_tensor_one(self):
    fed_type = computation_types.FederatedType(
        computation_types.TensorType(tf.float32, [2, 2]), placements.CLIENTS)
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
    fed_type = computation_types.FederatedType(tuple_type, placements.SERVER)
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
        placements.CLIENTS,
        all_equal=True)
    tuple_type = computation_types.StructType([('a', fed_type),
                                               ('b', fed_type)])

    zero = building_block_factory.create_generic_constant(tuple_type, 0)

    fed_zero = zero.argument[0]
    self.assertEqual(zero.type_signature, tuple_type)
    self.assertIsInstance(fed_zero.function, building_blocks.Intrinsic)
    self.assertEqual(fed_zero.function.uri,
                     intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri)
    self.assertIsInstance(fed_zero.argument, building_blocks.Call)
    actual_result = test_utils.run_tensorflow(fed_zero.argument.function.proto)
    self.assertTrue(np.array_equal(actual_result, np.zeros([2, 2])))


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
    op_type = computation_types.StructType((tf.int32, tf.int32))
    op_result = building_blocks.Data('o', tf.int32)
    op = building_blocks.Lambda('x', op_type, op_result)
    with self.assertRaises(TypeError):
      building_block_factory.create_sequence_reduce(None, zero, op)

  def test_raises_type_error_with_none_zero(self):
    value_type = computation_types.SequenceType(tf.int32)
    value = building_blocks.Data('v', value_type)
    op_type = computation_types.StructType((tf.int32, tf.int32))
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
    op_type = computation_types.StructType((tf.int32, tf.int32))
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
    tensor_type = computation_types.TensorType(tf.int32)
    bad_comp = building_blocks.Data('x', tensor_type)
    with self.assertRaises(TypeError):
      building_block_factory.create_named_federated_tuple(bad_comp, ['a'])

  def test_raises_federated_non_tuple(self):
    bad_comp = building_blocks.Data(
        'x', computation_types.FederatedType(tf.int32, placements.CLIENTS))
    with self.assertRaises(TypeError):
      building_block_factory.create_named_federated_tuple(bad_comp, ['a'])

  def test_raises_on_naked_string(self):
    data_tuple = building_blocks.Data(
        'x', computation_types.FederatedType([tf.int32], placements.CLIENTS))
    with self.assertRaises(TypeError):
      building_block_factory.create_named_federated_tuple(data_tuple, 'a')

  def test_raises_list_of_ints(self):
    data_tuple = building_blocks.Data(
        'x', computation_types.FederatedType([tf.int32], placements.SERVER))
    with self.assertRaises(TypeError):
      building_block_factory.create_named_federated_tuple(data_tuple, [1])

  def test_raises_wrong_list_length(self):
    data_tuple = building_blocks.Data(
        'x', computation_types.FederatedType([tf.int32], placements.SERVER))
    with self.assertRaises(ValueError):
      building_block_factory.create_named_federated_tuple(
          data_tuple, ['a', 'b'])

  @parameterized.named_parameters(
      ('server', placements.SERVER),
      ('clients', placements.CLIENTS),
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
      ('server', placements.SERVER),
      ('clients', placements.CLIENTS),
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
      ('server', placements.SERVER),
      ('clients', placements.CLIENTS),
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
      ('server', placements.SERVER),
      ('clients', placements.CLIENTS),
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
    type_signature = computation_types.StructType((tf.int32, tf.int32))
    comp = building_blocks.Data('data', type_signature)
    with self.assertRaises(TypeError):
      building_block_factory.create_named_tuple(comp, 'a')

  def test_raises_type_error_with_wrong_names_type_ints(self):
    type_signature = computation_types.StructType((tf.int32, tf.int32))
    comp = building_blocks.Data('data', type_signature)
    with self.assertRaises(TypeError):
      building_block_factory.create_named_tuple(comp, 'a')

  def test_raises_value_error_with_wrong_lengths(self):
    type_signature = computation_types.StructType((tf.int32, tf.int32))
    comp = building_blocks.Data('data', type_signature)
    with self.assertRaises(ValueError):
      building_block_factory.create_named_tuple(comp, ('a',))

  def test_creates_named_tuple_from_unamed_tuple(self):
    type_signature = computation_types.StructType((tf.int32, tf.int32))
    comp = building_blocks.Data('data', type_signature)
    named_comp = building_block_factory.create_named_tuple(comp, ('a', 'b'))
    expected_type_signature = computation_types.StructType(
        (('a', tf.int32), ('b', tf.int32)))
    self.assertEqual(named_comp.type_signature, expected_type_signature)

  def test_creates_named_tuple_from_named_tuple(self):
    type_signature = computation_types.StructType(
        (('a', tf.int32), ('b', tf.int32)))
    comp = building_blocks.Data('data', type_signature)
    named_comp = building_block_factory.create_named_tuple(comp, ('c', 'd'))
    expected_type_signature = computation_types.StructType(
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
    tup_1 = building_blocks.Struct((data_1, data_2, data_3))
    tup_2 = building_blocks.Struct((tup_1, tup_1))
    comp = tup_2
    new_comp = building_block_factory.create_zip(comp)
    self.assertEqual(comp.compact_representation(), '<<a,b,c>,<a,b,c>>')
    golden.check_string('zips_tuple_unnamed.expected',
                        comp.formatted_representation())
    self.assertEqual(
        str(comp.type_signature), '<<int32,float32,bool>,<int32,float32,bool>>')
    self.assertEqual(
        str(new_comp.type_signature),
        '<<int32,int32>,<float32,float32>,<bool,bool>>')

  def test_zips_tuple_named(self):
    data_1 = building_blocks.Data('a', tf.int32)
    data_2 = building_blocks.Data('b', tf.float32)
    data_3 = building_blocks.Data('c', tf.bool)
    tup_1 = building_blocks.Struct(
        (('d', data_1), ('e', data_2), ('f', data_3)))
    tup_2 = building_blocks.Struct((('g', tup_1), ('h', tup_1)))
    comp = tup_2
    new_comp = building_block_factory.create_zip(comp)
    self.assertEqual(comp.compact_representation(),
                     '<g=<d=a,e=b,f=c>,h=<d=a,e=b,f=c>>')
    golden.check_string('zips_tuple_named.expected',
                        comp.formatted_representation())
    self.assertEqual(
        str(comp.type_signature),
        '<g=<d=int32,e=float32,f=bool>,h=<d=int32,e=float32,f=bool>>')
    self.assertEqual(
        str(new_comp.type_signature),
        '<<int32,int32>,<float32,float32>,<bool,bool>>')

  def test_zips_reference(self):
    type_signature_1 = computation_types.StructType(
        [tf.int32, tf.float32, tf.bool])
    type_signature_2 = computation_types.StructType(
        [type_signature_1, type_signature_1])
    ref = building_blocks.Reference('a', type_signature_2)
    comp = ref
    new_comp = building_block_factory.create_zip(comp)
    self.assertEqual(comp.compact_representation(), 'a')
    golden.check_string('zips_reference.expected',
                        comp.formatted_representation())
    self.assertEqual(
        str(comp.type_signature), '<<int32,float32,bool>,<int32,float32,bool>>')
    self.assertEqual(
        str(new_comp.type_signature),
        '<<int32,int32>,<float32,float32>,<bool,bool>>')


class ConstructTensorFlowSelectingOutputsTest(absltest.TestCase):

  def test_raises_non_named_tuple_type(self):
    parameter_type = computation_types.TensorType(tf.int32)
    selection_spec = building_block_factory.SelectionSpec(
        tuple_index=0, selection_sequence=[0])
    with self.assertRaises(TypeError):
      building_block_factory.construct_tensorflow_selecting_and_packing_outputs(
          parameter_type, structure.Struct([(None, selection_spec)]))

  def test_raises_non_structure(self):
    parameter_type = computation_types.StructType([tf.int32])
    selection_spec = building_block_factory.SelectionSpec(
        tuple_index=0, selection_sequence=[0])
    with self.assertRaises(TypeError):
      building_block_factory.construct_tensorflow_selecting_and_packing_outputs(
          parameter_type, [selection_spec])

  def test_raises_nested_non_structure(self):
    parameter_type = computation_types.StructType([tf.int32])
    selection_spec = building_block_factory.SelectionSpec(
        tuple_index=0, selection_sequence=[0])
    with self.assertRaises(TypeError):
      building_block_factory.construct_tensorflow_selecting_and_packing_outputs(
          parameter_type,
          structure.from_container([[selection_spec]], recursive=False))

  def test_construct_selection_from_tuple_with_empty_list_type_signature(self):
    ntt = computation_types.StructType
    parameter_type = ntt([tf.int32, tf.float32])
    constructed_tf = building_block_factory.construct_tensorflow_selecting_and_packing_outputs(
        parameter_type, structure.from_container([]))
    self.assertIsInstance(constructed_tf, building_blocks.CompiledComputation)
    self.assertEqual(
        constructed_tf.type_signature,
        computation_types.FunctionType(ntt((tf.int32, tf.float32)), ntt(())))

  def test_construct_selection_from_two_tuple_correct_type_signature(self):
    ntt = computation_types.StructType
    parameter_type = ntt([tf.int32, tf.float32])
    selection_spec_1 = building_block_factory.SelectionSpec(
        tuple_index=0, selection_sequence=[])
    selection_spec_2 = building_block_factory.SelectionSpec(
        tuple_index=0, selection_sequence=[])
    output_structure = structure.from_container(
        [selection_spec_1, selection_spec_2])
    constructed_tf = building_block_factory.construct_tensorflow_selecting_and_packing_outputs(
        parameter_type, output_structure=output_structure)
    self.assertIsInstance(constructed_tf, building_blocks.CompiledComputation)
    self.assertEqual(
        constructed_tf.type_signature,
        computation_types.FunctionType(
            ntt((tf.int32, tf.float32)), ntt((tf.int32, tf.int32))))

  def test_construct_selection_from_two_tuple_correct_singleton_type_signature(
      self):
    parameter_type = computation_types.StructType([tf.int32, tf.float32])
    selection_spec = building_block_factory.SelectionSpec(
        tuple_index=0, selection_sequence=[])
    output_structure = structure.from_container([selection_spec])
    ntt = computation_types.StructType
    constructed_tf = building_block_factory.construct_tensorflow_selecting_and_packing_outputs(
        parameter_type, output_structure=output_structure)
    self.assertIsInstance(constructed_tf, building_blocks.CompiledComputation)
    self.assertEqual(
        constructed_tf.type_signature,
        computation_types.FunctionType(
            ntt((tf.int32, tf.float32)), ntt((tf.int32,))))

  def test_construct_selection_from_two_tuple_executes_correctly(self):
    parameter_type = computation_types.StructType([tf.int32, tf.float32])
    selection_spec_1 = building_block_factory.SelectionSpec(
        tuple_index=0, selection_sequence=[])
    selection_spec_2 = building_block_factory.SelectionSpec(
        tuple_index=0, selection_sequence=[])
    output_structure = structure.from_container(
        [selection_spec_1, selection_spec_2])
    constructed_tf = building_block_factory.construct_tensorflow_selecting_and_packing_outputs(
        parameter_type, output_structure=output_structure)
    result = test_utils.run_tensorflow(constructed_tf.proto, [0, 1.])
    self.assertLen(result, 2)
    self.assertEqual(result[0], 0)
    self.assertEqual(result[1], 0)
    result = test_utils.run_tensorflow(constructed_tf.proto, [1, 0.])
    self.assertLen(result, 2)
    self.assertEqual(result[0], 1)
    self.assertEqual(result[1], 1)

  def test_construct_selection_with_names(self):
    parameter_type = computation_types.StructType([('a', tf.int32),
                                                   ('b', tf.float32)])
    selection_spec_1 = building_block_factory.SelectionSpec(
        tuple_index=0, selection_sequence=[])
    selection_spec_2 = building_block_factory.SelectionSpec(
        tuple_index=1, selection_sequence=[])
    output_structure = structure.Struct([('a', selection_spec_1),
                                         ('b', selection_spec_2)])
    constructed_tf = building_block_factory.construct_tensorflow_selecting_and_packing_outputs(
        parameter_type, output_structure=output_structure)
    self.assertEqual(
        constructed_tf.type_signature,
        computation_types.FunctionType([('a', tf.int32), ('b', tf.float32)],
                                       [('a', tf.int32), ('b', tf.float32)]))

  def test_construct_tuple_packed_selection_with_name(self):
    parameter_type = computation_types.StructType([('a', tf.int32),
                                                   ('b', tf.float32)])
    selection_spec_1 = building_block_factory.SelectionSpec(
        tuple_index=0, selection_sequence=[])
    selection_spec_2 = building_block_factory.SelectionSpec(
        tuple_index=1, selection_sequence=[])
    output_structure = structure.Struct([
        ('c',
         structure.from_container([selection_spec_1, selection_spec_2],
                                  recursive=True))
    ])
    ntt = computation_types.StructType
    constructed_tf = building_block_factory.construct_tensorflow_selecting_and_packing_outputs(
        parameter_type, output_structure=output_structure)
    self.assertEqual(
        constructed_tf.type_signature,
        computation_types.FunctionType(
            ntt((('a', tf.int32), ('b', tf.float32))),
            ntt((('c', ntt((tf.int32, tf.float32)))))))

  def test_construct_selection_from_nested_tuple_executes_correctly(self):
    parameter_type = computation_types.StructType([[[tf.int32]], tf.float32])
    selection_spec = building_block_factory.SelectionSpec(
        tuple_index=0, selection_sequence=[0, 0])
    output_structure = structure.from_container([selection_spec],
                                                recursive=True)
    constructed_tf = building_block_factory.construct_tensorflow_selecting_and_packing_outputs(
        parameter_type, output_structure=output_structure)
    result = test_utils.run_tensorflow(constructed_tf.proto, [[[0]], 1.])
    self.assertEqual(result[0], 0)

  def test_construct_selection_from_nested_tuple_repack_into_tuple_executes_correctly(
      self):
    parameter_type = computation_types.StructType([[[tf.int32]], tf.float32])
    selection_spec = building_block_factory.SelectionSpec(
        tuple_index=0, selection_sequence=[0, 0])
    output_structure = structure.from_container([[[selection_spec]]],
                                                recursive=True)
    constructed_tf = building_block_factory.construct_tensorflow_selecting_and_packing_outputs(
        parameter_type, output_structure=output_structure)
    result = test_utils.run_tensorflow(constructed_tf.proto, [[[0]], 1.])
    self.assertEqual(result[0][0][0], 0)

  def test_construct_selection_from_two_tuple_repack_named_lower_level_type_signature(
      self):
    parameter_type = computation_types.StructType([tf.int32, tf.float32])
    selection_spec_1 = building_block_factory.SelectionSpec(
        tuple_index=0, selection_sequence=[])
    selection_spec_2 = building_block_factory.SelectionSpec(
        tuple_index=0, selection_sequence=[])
    output_structure = structure.from_container(
        [structure.Struct([('a', selection_spec_1)]), selection_spec_2],
        recursive=True)
    ntt = computation_types.StructType
    constructed_tf = building_block_factory.construct_tensorflow_selecting_and_packing_outputs(
        parameter_type, output_structure=output_structure)
    self.assertEqual(
        constructed_tf.type_signature,
        computation_types.FunctionType(
            ntt((tf.int32, tf.float32)), ntt((ntt(('a', tf.int32)), tf.int32))))

  def test_construct_selection_from_two_tuple_repack_lower_level_output_executes_correctly(
      self):
    parameter_type = computation_types.StructType([tf.int32, tf.float32])
    selection_spec_1 = building_block_factory.SelectionSpec(
        tuple_index=0, selection_sequence=[])
    selection_spec_2 = building_block_factory.SelectionSpec(
        tuple_index=0, selection_sequence=[])
    output_structure = structure.from_container(
        [[selection_spec_1], selection_spec_2], recursive=True)
    constructed_tf = building_block_factory.construct_tensorflow_selecting_and_packing_outputs(
        parameter_type, output_structure=output_structure)
    result = test_utils.run_tensorflow(constructed_tf.proto, [0, 1.])
    self.assertLen(result, 2)
    self.assertLen(result[0], 1)
    self.assertEqual(result[0][0], 0)
    self.assertEqual(result[1], 0)
    result = test_utils.run_tensorflow(constructed_tf.proto, [1, 0.])
    self.assertLen(result, 2)
    self.assertLen(result[0], 1)
    self.assertEqual(result[0][0], 1)
    self.assertEqual(result[1], 1)
    flipped_parameter_type = computation_types.StructType(
        [tf.int32, tf.float32])
    flipped_output_structure = structure.from_container(
        [selection_spec_1, [selection_spec_2]], recursive=True)
    flipped_packing_tf = building_block_factory.construct_tensorflow_selecting_and_packing_outputs(
        flipped_parameter_type, output_structure=flipped_output_structure)
    result = test_utils.run_tensorflow(flipped_packing_tf.proto, [0, 1.])
    self.assertLen(result, 2)
    self.assertEqual(result[0], 0)
    self.assertEqual(result[1][0], 0)
    result = test_utils.run_tensorflow(flipped_packing_tf.proto, [1, 0.])
    self.assertLen(result, 2)
    self.assertEqual(result[0], 1)
    self.assertEqual(result[1][0], 1)


if __name__ == '__main__':
  absltest.main()
