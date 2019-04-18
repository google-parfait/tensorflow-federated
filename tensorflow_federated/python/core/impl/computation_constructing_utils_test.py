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
import tensorflow as tf

from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_constructing_utils
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import placement_literals
from tensorflow_federated.python.core.impl import type_utils
from tensorflow_federated.python.core.impl import value_impl


class ComputationConstructionUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(('clients', placement_literals.CLIENTS),
                                  ('server', placement_literals.SERVER))
  def test_getitem_comp_construction(self, placement):
    federated_value = computation_building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32), ('b', tf.bool)],
                                        placement, True))
    get_0_comp = computation_constructing_utils.construct_federated_getitem_comp(
        federated_value, 0)
    self.assertEqual(str(get_0_comp), '(x -> x[0])')
    get_slice_comp = computation_constructing_utils.construct_federated_getitem_comp(
        federated_value, slice(None, None, -1))
    self.assertEqual(str(get_slice_comp), '(x -> <b=x[1],a=x[0]>)')

  @parameterized.named_parameters(('clients', placement_literals.CLIENTS),
                                  ('server', placement_literals.SERVER))
  def test_getattr_comp_construction(self, placement):
    federated_value = computation_building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32), ('b', tf.bool)],
                                        placement, True))
    get_a_comp = computation_constructing_utils.construct_federated_getattr_comp(
        federated_value, 'a')
    self.assertEqual(str(get_a_comp), '(x -> x.a)')
    get_b_comp = computation_constructing_utils.construct_federated_getattr_comp(
        federated_value, 'b')
    self.assertEqual(str(get_b_comp), '(x -> x.b)')
    non_federated_arg = computation_building_blocks.Reference(
        'test',
        computation_types.NamedTupleType([('a', tf.int32), ('b', tf.bool)]))
    with self.assertRaises(TypeError):
      _ = computation_constructing_utils.construct_federated_getattr_comp(
          non_federated_arg, 'a')
    with self.assertRaisesRegex(ValueError, 'has no element of name c'):
      _ = computation_constructing_utils.construct_federated_getattr_comp(
          federated_value, 'c')

  def test_intrinsic_construction_server(self):
    federated_comp = computation_building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32), ('b', tf.bool)],
                                        placement_literals.SERVER, True))
    arg_ref = computation_building_blocks.Reference('x', [('a', tf.int32),
                                                          ('b', tf.bool)])
    return_val = computation_building_blocks.Selection(arg_ref, name='a')
    non_federated_fn = computation_building_blocks.Lambda(
        'x', arg_ref.type_signature, return_val)
    intrinsic = computation_constructing_utils.construct_map_or_apply(
        non_federated_fn, federated_comp)
    self.assertEqual(str(intrinsic), 'federated_apply')

  def test_intrinsic_construction_clients(self):
    federated_comp = computation_building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32), ('b', tf.bool)],
                                        placement_literals.CLIENTS, True))
    arg_ref = computation_building_blocks.Reference('x', [('a', tf.int32),
                                                          ('b', tf.bool)])
    return_val = computation_building_blocks.Selection(arg_ref, name='a')
    non_federated_fn = computation_building_blocks.Lambda(
        'x', arg_ref.type_signature, return_val)
    intrinsic = computation_constructing_utils.construct_map_or_apply(
        non_federated_fn, federated_comp)
    self.assertEqual(str(intrinsic), 'federated_map')

  def test_intrinsic_construction_fails_bad_type(self):
    x = computation_building_blocks.Reference('x', tf.int32)
    bad_lambda = computation_building_blocks.Lambda('x', tf.int32, x)
    x = computation_building_blocks.Reference('x', tf.int32)
    clients_ref = computation_building_blocks.Reference(
        'y',
        computation_types.FederatedType(tf.float32, placement_literals.CLIENTS))
    with self.assertRaises(TypeError):
      computation_constructing_utils.construct_map_or_apply(
          bad_lambda, clients_ref)

  def test_federated_getitem_call_fails_value(self):
    x = computation_building_blocks.Reference(
        'x', computation_types.to_type([tf.int32]))
    with self.assertRaises(TypeError):
      computation_constructing_utils.construct_federated_getitem_call(
          value_impl.to_value(x), 0)

  def test_federated_getattr_call_fails_value(self):
    x = computation_building_blocks.Reference(
        'x', computation_types.to_type([('x', tf.int32)]))
    with self.assertRaises(TypeError):
      computation_constructing_utils.construct_federated_getattr_call(
          value_impl.to_value(x), 'x')

  def test_federated_getitem_comp_fails_value(self):
    x = computation_building_blocks.Reference(
        'x', computation_types.to_type([tf.int32]))
    with self.assertRaises(TypeError):
      computation_constructing_utils.construct_federated_getitem_comp(
          value_impl.to_value(x), 0)

  def test_federated_getattr_comp_fails_value(self):
    x = computation_building_blocks.Reference(
        'x', computation_types.to_type([('x', tf.int32)]))
    with self.assertRaises(TypeError):
      computation_constructing_utils.construct_federated_getattr_comp(
          value_impl.to_value(x), 'x')

  @parameterized.named_parameters(('clients', placement_literals.CLIENTS),
                                  ('server', placement_literals.SERVER))
  def test_getitem_call_named(self, placement):
    federated_comp_named = computation_building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32), ('b', tf.bool)],
                                        placement, True))
    self.assertEqual(
        str(federated_comp_named.type_signature.member), '<a=int32,b=bool>')
    idx_0 = computation_constructing_utils.construct_federated_getitem_call(
        federated_comp_named, 0)
    idx_1 = computation_constructing_utils.construct_federated_getitem_call(
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
    flipped = computation_constructing_utils.construct_federated_getitem_call(
        federated_comp_named, slice(None, None, -1))
    self.assertIsInstance(flipped.type_signature,
                          computation_types.FederatedType)
    self.assertEqual(str(flipped.type_signature.member), '<b=bool,a=int32>')
    type_utils.check_federated_value_placement(
        value_impl.to_value(flipped, None, context_stack_impl.context_stack),
        placement)

  @parameterized.named_parameters(('clients', placement_literals.CLIENTS),
                                  ('server', placement_literals.SERVER))
  def test_getitem_call_unnamed(self, placement):
    federated_comp_unnamed = computation_building_blocks.Reference(
        'test',
        computation_types.FederatedType([tf.int32, tf.bool], placement, True))
    self.assertEqual(
        str(federated_comp_unnamed.type_signature.member), '<int32,bool>')
    unnamed_idx_0 = computation_constructing_utils.construct_federated_getitem_call(
        federated_comp_unnamed, 0)
    unnamed_idx_1 = computation_constructing_utils.construct_federated_getitem_call(
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
    unnamed_flipped = computation_constructing_utils.construct_federated_getitem_call(
        federated_comp_unnamed, slice(None, None, -1))
    self.assertIsInstance(unnamed_flipped.type_signature,
                          computation_types.FederatedType)
    self.assertEqual(str(unnamed_flipped.type_signature.member), '<bool,int32>')
    type_utils.check_federated_value_placement(
        value_impl.to_value(unnamed_flipped, None,
                            context_stack_impl.context_stack), placement)

  @parameterized.named_parameters(('clients', placement_literals.CLIENTS),
                                  ('server', placement_literals.SERVER))
  def test_getattr_call_named(self, placement):
    federated_comp_named = computation_building_blocks.Reference(
        'test',
        computation_types.FederatedType([('a', tf.int32),
                                         ('b', tf.bool), tf.int32], placement,
                                        True))
    self.assertEqual(
        str(federated_comp_named.type_signature.member),
        '<a=int32,b=bool,int32>')
    name_a = computation_constructing_utils.construct_federated_getattr_call(
        federated_comp_named, 'a')
    name_b = computation_constructing_utils.construct_federated_getattr_call(
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
      _ = computation_constructing_utils.construct_federated_getattr_call(
          federated_comp_named, 'c')


if __name__ == '__main__':
  absltest.main()
