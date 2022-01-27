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

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_federated.python.common_libs import golden
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.compiler import building_block_analysis
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import test_utils as compiler_test_utils
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements


class TransformTestBase(test_case.TestCase):

  def assert_transforms(self, comp, file, changes_type=False, unmodified=False):
    # NOTE: A `transform` method must be present on inheritors.
    after, modified = self.transform(comp)
    golden.check_string(
        file, f'Before transformation:\n\n{comp.formatted_representation()}\n\n'
        f'After transformation:\n\n{after.formatted_representation()}')
    if not changes_type:
      self.assert_types_identical(comp.type_signature, after.type_signature)
    if unmodified:
      self.assertFalse(modified)
    else:
      self.assertTrue(modified)


def _create_chained_whimsy_federated_applys(functions, arg):
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  for fn in functions:
    py_typecheck.check_type(fn, building_blocks.ComputationBuildingBlock)
    if not fn.parameter_type.is_assignable_from(arg.type_signature.member):
      raise TypeError(
          'The parameter of the function is of type {}, and the argument is of '
          'an incompatible type {}.'.format(
              str(fn.parameter_type), str(arg.type_signature.member)))
    call = building_block_factory.create_federated_apply(fn, arg)
    arg = call
  return call


def _create_chained_whimsy_federated_maps(functions, arg):
  py_typecheck.check_type(arg, building_blocks.ComputationBuildingBlock)
  for fn in functions:
    py_typecheck.check_type(fn, building_blocks.ComputationBuildingBlock)
    if not fn.parameter_type.is_assignable_from(arg.type_signature.member):
      raise TypeError(
          'The parameter of the function is of type {}, and the argument is of '
          'an incompatible type {}.'.format(
              str(fn.parameter_type), str(arg.type_signature.member)))
    call = building_block_factory.create_federated_map(fn, arg)
    arg = call
  return call


def _create_lambda_to_whimsy_cast(parameter_name, parameter_type, result_type):
  py_typecheck.check_type(parameter_type, tf.dtypes.DType)
  py_typecheck.check_type(result_type, tf.dtypes.DType)
  arg = building_blocks.Data('data', result_type)
  return building_blocks.Lambda(parameter_name, parameter_type, arg)


def _count_called_intrinsics(comp, uri=None):

  def _predicate(comp):
    return building_block_analysis.is_called_intrinsic(comp, uri)

  return tree_analysis.count(comp, _predicate)


def _create_complex_computation():
  tensor_type = computation_types.TensorType(tf.int32)
  compiled = building_block_factory.create_compiled_identity(tensor_type, 'a')
  federated_type = computation_types.FederatedType(tf.int32, placements.SERVER)
  arg_ref = building_blocks.Reference('arg', federated_type)
  bindings = []
  results = []

  def _bind(name, value):
    bindings.append((name, value))
    return building_blocks.Reference(name, value.type_signature)

  for i in range(2):
    called_federated_broadcast = building_block_factory.create_federated_broadcast(
        arg_ref)
    called_federated_map = building_block_factory.create_federated_map(
        compiled, _bind(f'broadcast_{i}', called_federated_broadcast))
    called_federated_mean = building_block_factory.create_federated_mean(
        _bind(f'map_{i}', called_federated_map), None)
    results.append(_bind(f'mean_{i}', called_federated_mean))
  result = building_blocks.Struct(results)
  block = building_blocks.Block(bindings, result)
  return building_blocks.Lambda('arg', tf.int32, block)


class InlineBlockLocalsTest(test_case.TestCase):

  def test_raises_type_error_with_none_comp(self):
    with self.assertRaises(TypeError):
      tree_transformations.inline_block_locals(None)

  def test_raises_type_error_with_wrong_type_variable_names(self):
    block = compiler_test_utils.create_identity_block_with_whimsy_data(
        variable_name='a')
    comp = block
    with self.assertRaises(TypeError):
      tree_transformations.inline_block_locals(comp, 1)

  def test_raises_value_error_with_non_unique_variable_names(self):
    data = building_blocks.Data('data', tf.int32)
    block = building_blocks.Block([('a', data), ('a', data)], data)
    with self.assertRaises(ValueError):
      tree_transformations.inline_block_locals(block)

  def test_noops_with_unbound_reference(self):
    ref = building_blocks.Reference('x', tf.int32)
    lambda_binding_y = building_blocks.Lambda('y', tf.float32, ref)

    transformed_comp, modified = tree_transformations.inline_block_locals(
        lambda_binding_y)

    self.assertEqual(lambda_binding_y.compact_representation(), '(y -> x)')
    self.assertEqual(transformed_comp.compact_representation(), '(y -> x)')
    self.assertEqual(transformed_comp.type_signature,
                     lambda_binding_y.type_signature)
    self.assertFalse(modified)

  def test_inlines_one_block_variable(self):
    block = compiler_test_utils.create_identity_block_with_whimsy_data(
        variable_name='a')
    comp = block

    transformed_comp, modified = tree_transformations.inline_block_locals(comp)

    self.assertEqual(comp.compact_representation(), '(let a=data in a)')
    self.assertEqual(transformed_comp.compact_representation(), 'data')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_inlines_two_block_variables(self):
    data = building_blocks.Data('data', tf.int32)
    ref = building_blocks.Reference('a', tf.int32)
    tup = building_blocks.Struct((ref, ref))
    block = building_blocks.Block((('a', data),), tup)
    comp = block

    transformed_comp, modified = tree_transformations.inline_block_locals(comp)

    self.assertEqual(comp.compact_representation(), '(let a=data in <a,a>)')
    self.assertEqual(transformed_comp.compact_representation(), '<data,data>')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_inlines_block_variables(self):
    data = building_blocks.Data('data', tf.int32)
    ref_1 = building_blocks.Reference('a', tf.int32)
    ref_2 = building_blocks.Reference('b', tf.int32)
    tup = building_blocks.Struct((ref_1, ref_2))
    block = building_blocks.Block((('a', data), ('b', data)), tup)
    comp = block

    transformed_comp, modified = tree_transformations.inline_block_locals(
        comp, variable_names=('a',))

    self.assertEqual(comp.compact_representation(),
                     '(let a=data,b=data in <a,b>)')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let b=data in <data,b>)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_inlines_variables_in_block_variables(self):
    block_1 = compiler_test_utils.create_identity_block_with_whimsy_data(
        variable_name='a')
    ref = building_blocks.Reference('b', block_1.type_signature)
    block_2 = building_blocks.Block((('b', block_1),), ref)
    comp = block_2

    transformed_comp, modified = tree_transformations.inline_block_locals(comp)

    self.assertEqual(comp.compact_representation(),
                     '(let b=(let a=data in a) in b)')
    self.assertEqual(transformed_comp.compact_representation(), 'data')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_inlines_variables_in_block_results(self):
    ref_1 = building_blocks.Reference('a', tf.int32)
    data = building_blocks.Data('data', tf.int32)
    ref_2 = building_blocks.Reference('b', tf.int32)
    block_1 = building_blocks.Block([('b', ref_1)], ref_2)
    block_2 = building_blocks.Block([('a', data)], block_1)
    comp = block_2

    transformed_comp, modified = tree_transformations.inline_block_locals(comp)

    self.assertEqual(comp.compact_representation(),
                     '(let a=data in (let b=a in b))')
    self.assertEqual(transformed_comp.compact_representation(), 'data')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_inlines_variables_bound_sequentially(self):
    data = building_blocks.Data('data', tf.int32)
    ref_1 = building_blocks.Reference('a', tf.int32)
    ref_2 = building_blocks.Reference('b', tf.int32)
    ref_3 = building_blocks.Reference('c', tf.int32)
    block = building_blocks.Block((('b', data), ('c', ref_2), ('a', ref_3)),
                                  ref_1)
    comp = block

    transformed_comp, modified = tree_transformations.inline_block_locals(comp)

    self.assertEqual(comp.compact_representation(), '(let b=data,c=b,a=c in a)')
    self.assertEqual(transformed_comp.compact_representation(), 'data')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_does_not_inline_lambda_parameter(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    comp = fn

    transformed_comp, modified = tree_transformations.inline_block_locals(comp)

    self.assertEqual(transformed_comp.compact_representation(),
                     comp.compact_representation())
    self.assertEqual(transformed_comp.compact_representation(), '(a -> a)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_does_not_inline_block_variables(self):
    block = compiler_test_utils.create_identity_block_with_whimsy_data(
        variable_name='a')
    comp = block

    transformed_comp, modified = tree_transformations.inline_block_locals(
        comp, variable_names=('b',))

    self.assertEqual(comp.compact_representation(), '(let a=data in a)')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let a=data in a)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)


class RemoveDuplicateBlockLocals(TransformTestBase):

  def transform(self, comp):
    intermediate_comp, _ = tree_transformations.uniquify_reference_names(comp)
    transformed_comp, modified = tree_transformations.remove_duplicate_block_locals(
        intermediate_comp)
    transformed_comp, _ = tree_transformations.uniquify_reference_names(
        transformed_comp)
    return transformed_comp, modified

  def test_raises_type_error_with_none(self):
    with self.assertRaises(TypeError):
      tree_transformations.remove_duplicate_block_locals(None)

  def test_raises_value_error_with_non_unique_variable_names(self):
    data = building_blocks.Data('data', tf.int32)
    block = building_blocks.Block([('a', data), ('a', data)], data)
    with self.assertRaises(ValueError):
      tree_transformations.remove_duplicate_block_locals(block)

  def test_removes_federated_aggregate(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    block = building_blocks.Block(
        [
            ('agg_0', called_intrinsic),
            ('agg_1', called_intrinsic),
        ],
        building_blocks.Struct([
            building_blocks.Reference('agg_0', called_intrinsic.type_signature),
            building_blocks.Reference('agg_1', called_intrinsic.type_signature),
        ]))
    self.assert_transforms(block,
                           'remove_duplicate_federated_aggregate.expected')

  def test_removes_federated_broadcast(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_federated_broadcast(
    )
    block = building_blocks.Block(
        [
            ('broadcast_0', called_intrinsic),
            ('broadcast_1', called_intrinsic),
        ],
        building_blocks.Struct([
            building_blocks.Reference('broadcast_0',
                                      called_intrinsic.type_signature),
            building_blocks.Reference('broadcast_1',
                                      called_intrinsic.type_signature),
        ]))
    self.assert_transforms(block,
                           'remove_duplicate_federated_broadcast.expected')

  def test_removes_complex_comp(self):
    complex_comp = _create_complex_computation()
    self.assert_transforms(complex_comp,
                           'remove_duplicate_complex_comp.expected')

  def test_removes_chained_references_bound_by_lambda(self):
    ref_1 = building_blocks.Reference('a', tf.int32)
    ref_2 = building_blocks.Reference('b', tf.int32)
    tup = building_blocks.Struct([ref_2])
    block = building_blocks.Block([(ref_2.name, ref_1)], tup)
    fn = building_blocks.Lambda(ref_1.name, ref_1.type_signature, block)
    comp = fn

    transformed_comp, modified = tree_transformations.remove_duplicate_block_locals(
        comp)

    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        '(a -> (let\n'
        '  b=a\n'
        ' in <\n'
        '  b\n'
        '>))'
    )
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(a -> <a>)'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_leaves_unbound_ref_alone(self):
    ref_1 = building_blocks.Reference('a', tf.int32)
    ref_2 = building_blocks.Reference('b', tf.int32)
    tup = building_blocks.Struct([ref_2])
    block = building_blocks.Block([(ref_2.name, ref_1)], tup)
    comp = block

    transformed_comp, modified = tree_transformations.remove_duplicate_block_locals(
        comp)

    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        '(let\n'
        '  b=a\n'
        ' in <\n'
        '  b\n'
        '>)'
    )
    self.assertEqual(
        transformed_comp.compact_representation(),
        '<a>'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_strips_reference_chain_culminating_in_unbound_ref(self):
    ref_1 = building_blocks.Reference('a', tf.int32)
    ref_2 = building_blocks.Reference('b', tf.int32)
    ref_3 = building_blocks.Reference('c', tf.int32)
    tup = building_blocks.Struct([ref_3])
    comp = building_blocks.Block([(ref_2.name, ref_1), (ref_3.name, ref_2)],
                                 tup)
    transformed_comp, modified = tree_transformations.remove_duplicate_block_locals(
        comp)

    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        '(let\n'
        '  b=a,\n'
        '  c=b\n'
        ' in <\n'
        '  c\n'
        '>)'
    )
    self.assertEqual(
        transformed_comp.compact_representation(),
        '<a>'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)


class RemoveMappedOrAppliedIdentityTest(parameterized.TestCase):

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      tree_transformations.remove_mapped_or_applied_identity(None)

  # pyformat: disable
  @parameterized.named_parameters(
      ('federated_apply',
       intrinsic_defs.FEDERATED_APPLY.uri,
       compiler_test_utils.create_whimsy_called_federated_apply),
      ('federated_map',
       intrinsic_defs.FEDERATED_MAP.uri,
       compiler_test_utils.create_whimsy_called_federated_map),
      ('federated_map_all_equal',
       intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri,
       compiler_test_utils.create_whimsy_called_federated_map_all_equal),
      ('sequence_map',
       intrinsic_defs.SEQUENCE_MAP.uri,
       compiler_test_utils.create_whimsy_called_sequence_map),
  )
  # pyformat: enable
  def test_removes_intrinsic(self, uri, factory):
    call = factory(parameter_name='a')
    comp = call

    transformed_comp, modified = tree_transformations.remove_mapped_or_applied_identity(
        comp)

    self.assertEqual(comp.compact_representation(),
                     '{}(<(a -> a),data>)'.format(uri))
    self.assertEqual(transformed_comp.compact_representation(), 'data')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_removes_federated_map_with_named_result(self):
    parameter_type = [('a', tf.int32), ('b', tf.int32)]
    fn = compiler_test_utils.create_identity_function('c', parameter_type)
    arg_type = computation_types.FederatedType(parameter_type,
                                               placements.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    call = building_block_factory.create_federated_map(fn, arg)
    comp = call

    transformed_comp, modified = tree_transformations.remove_mapped_or_applied_identity(
        comp)

    self.assertEqual(comp.compact_representation(),
                     'federated_map(<(c -> c),data>)')
    self.assertEqual(transformed_comp.compact_representation(), 'data')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_removes_nested_federated_map(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_federated_map(
        parameter_name='a')
    block = compiler_test_utils.create_whimsy_block(
        called_intrinsic, variable_name='b')
    comp = block

    transformed_comp, modified = tree_transformations.remove_mapped_or_applied_identity(
        comp)

    self.assertEqual(comp.compact_representation(),
                     '(let b=data in federated_map(<(a -> a),data>))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let b=data in data)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_removes_chained_federated_maps(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    call = _create_chained_whimsy_federated_maps([fn, fn], arg)
    comp = call

    transformed_comp, modified = tree_transformations.remove_mapped_or_applied_identity(
        comp)

    self.assertEqual(
        comp.compact_representation(),
        'federated_map(<(a -> a),federated_map(<(a -> a),data>)>)')
    self.assertEqual(transformed_comp.compact_representation(), 'data')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_does_not_remove_whimsy_intrinsic(self):
    comp = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')

    transformed_comp, modified = tree_transformations.remove_mapped_or_applied_identity(
        comp)

    self.assertEqual(transformed_comp.compact_representation(),
                     comp.compact_representation())
    self.assertEqual(transformed_comp.compact_representation(), 'intrinsic(a)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_does_not_remove_called_lambda(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    arg = building_blocks.Data('data', tf.int32)
    call = building_blocks.Call(fn, arg)
    comp = call

    transformed_comp, modified = tree_transformations.remove_mapped_or_applied_identity(
        comp)

    self.assertEqual(transformed_comp.compact_representation(),
                     comp.compact_representation())
    self.assertEqual(transformed_comp.compact_representation(),
                     '(a -> a)(data)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)


class RemoveUnusedBlockLocalsTest(test_case.TestCase):

  def setUp(self):
    super().setUp()
    self._unused_block_remover = tree_transformations.RemoveUnusedBlockLocals()

  def test_should_transform_block(self):
    blk = building_blocks.Block([('x', building_blocks.Data('a', tf.int32))],
                                building_blocks.Data('b', tf.int32))
    self.assertTrue(self._unused_block_remover.should_transform(blk))

  def test_should_not_transform_data(self):
    data = building_blocks.Data('b', tf.int32)
    self.assertFalse(self._unused_block_remover.should_transform(data))

  def test_removes_block_with_unused_reference(self):
    input_data = building_blocks.Data('b', tf.int32)
    blk = building_blocks.Block([('x', building_blocks.Data('a', tf.int32))],
                                input_data)
    data, modified = tree_transformations._apply_transforms(
        blk, self._unused_block_remover)
    self.assertTrue(modified)
    self.assertEqual(data.compact_representation(),
                     input_data.compact_representation())

  def test_unwraps_block_with_empty_locals(self):
    input_data = building_blocks.Data('b', tf.int32)
    blk = building_blocks.Block([], input_data)
    data, modified = tree_transformations._apply_transforms(
        blk, self._unused_block_remover)
    self.assertTrue(modified)
    self.assertEqual(data.compact_representation(),
                     input_data.compact_representation())

  def test_removes_nested_blocks_with_unused_reference(self):
    input_data = building_blocks.Data('b', tf.int32)
    blk = building_blocks.Block([('x', building_blocks.Data('a', tf.int32))],
                                input_data)
    higher_level_blk = building_blocks.Block([('y', input_data)], blk)
    data, modified = tree_transformations._apply_transforms(
        higher_level_blk, self._unused_block_remover)
    self.assertTrue(modified)
    self.assertEqual(data.compact_representation(),
                     input_data.compact_representation())

  def test_leaves_single_used_reference(self):
    blk = building_blocks.Block([('x', building_blocks.Data('a', tf.int32))],
                                building_blocks.Reference('x', tf.int32))
    transformed_blk, modified = tree_transformations._apply_transforms(
        blk, self._unused_block_remover)
    self.assertFalse(modified)
    self.assertEqual(transformed_blk.compact_representation(),
                     blk.compact_representation())

  def test_leaves_chained_used_references(self):
    blk = building_blocks.Block(
        [('x', building_blocks.Data('a', tf.int32)),
         ('y', building_blocks.Reference('x', tf.int32))],
        building_blocks.Reference('y', tf.int32))
    transformed_blk, modified = tree_transformations._apply_transforms(
        blk, self._unused_block_remover)
    self.assertFalse(modified)
    self.assertEqual(transformed_blk.compact_representation(),
                     blk.compact_representation())

  def test_removes_locals_referencing_each_other_but_unreferenced_in_result(
      self):
    input_data = building_blocks.Data('b', tf.int32)
    blk = building_blocks.Block(
        [('x', building_blocks.Data('a', tf.int32)),
         ('y', building_blocks.Reference('x', tf.int32))], input_data)
    transformed_blk, modified = tree_transformations._apply_transforms(
        blk, self._unused_block_remover)
    self.assertTrue(modified)
    self.assertEqual(transformed_blk.compact_representation(),
                     input_data.compact_representation())

  def test_leaves_lone_referenced_local(self):
    ref = building_blocks.Reference('y', tf.int32)
    blk = building_blocks.Block([('x', building_blocks.Data('a', tf.int32)),
                                 ('y', building_blocks.Data('b', tf.int32))],
                                ref)
    transformed_blk, modified = tree_transformations._apply_transforms(
        blk, self._unused_block_remover)
    self.assertTrue(modified)
    self.assertEqual(transformed_blk.compact_representation(), '(let y=b in y)')


class ReplaceCalledLambdaWithBlockTest(test_case.TestCase):

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      tree_transformations.replace_called_lambda_with_block(None)

  def test_replaces_called_lambda(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    arg = building_blocks.Data('data', tf.int32)
    call = building_blocks.Call(fn, arg)
    comp = call

    transformed_comp, modified = tree_transformations.replace_called_lambda_with_block(
        comp)

    self.assertEqual(comp.compact_representation(), '(a -> a)(data)')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let a=data in a)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replaces_called_lambda_bound_to_block_local(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    ref = building_blocks.Reference('x', fn.type_signature)
    arg = building_blocks.Data('data', tf.int32)
    call = building_blocks.Call(ref, arg)
    blk = building_blocks.Block([('x', fn)], call)

    transformed_comp, modified = tree_transformations.replace_called_lambda_with_block(
        blk)

    self.assertEqual(blk.compact_representation(),
                     '(let x=(a -> a) in x(data))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let a=data in a)')
    self.assertEqual(transformed_comp.type_signature, blk.type_signature)
    self.assertTrue(modified)

  def test_replaces_nested_called_lambda(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    arg = building_blocks.Data('data', tf.int32)
    call = building_blocks.Call(fn, arg)
    block = compiler_test_utils.create_whimsy_block(call, variable_name='b')
    comp = block

    transformed_comp, modified = tree_transformations.replace_called_lambda_with_block(
        comp)

    self.assertEqual(comp.compact_representation(),
                     '(let b=data in (a -> a)(data))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let b=data in (let a=data in a))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replaces_chained_called_lambdas(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    arg = building_blocks.Data('data', tf.int32)
    call = compiler_test_utils.create_chained_calls([fn, fn], arg)
    comp = call

    transformed_comp, modified = tree_transformations.replace_called_lambda_with_block(
        comp)

    self.assertEqual(comp.compact_representation(), '(a -> a)((a -> a)(data))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let a=(let a=data in a) in a)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_does_not_replace_uncalled_lambda(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    comp = fn

    transformed_comp, modified = tree_transformations.replace_called_lambda_with_block(
        comp)

    self.assertEqual(transformed_comp.compact_representation(),
                     comp.compact_representation())
    self.assertEqual(transformed_comp.compact_representation(), '(a -> a)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_does_not_replace_separated_called_lambda(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    block = compiler_test_utils.create_whimsy_block(fn, variable_name='b')
    arg = building_blocks.Data('data', tf.int32)
    call = building_blocks.Call(block, arg)
    comp = call

    transformed_comp, modified = tree_transformations.replace_called_lambda_with_block(
        comp)

    self.assertEqual(transformed_comp.compact_representation(),
                     comp.compact_representation())
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let b=data in (a -> a))(data)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)


class ReplaceSelectionFromTupleWithElementTest(test_case.TestCase):

  def test_fails_on_none_comp(self):
    with self.assertRaises(TypeError):
      tree_transformations.replace_selection_from_tuple_with_element(None)

  def test_leaves_selection_from_ref_by_index_alone(self):
    ref_to_tuple = building_blocks.Reference('tup', [('a', tf.int32),
                                                     ('b', tf.float32)])
    a_selected = building_blocks.Selection(ref_to_tuple, index=0)
    b_selected = building_blocks.Selection(ref_to_tuple, index=1)

    a_returned, a_transformed = tree_transformations.replace_selection_from_tuple_with_element(
        a_selected)
    b_returned, b_transformed = tree_transformations.replace_selection_from_tuple_with_element(
        b_selected)

    self.assertFalse(a_transformed)
    self.assertEqual(a_returned.proto, a_selected.proto)
    self.assertFalse(b_transformed)
    self.assertEqual(b_returned.proto, b_selected.proto)

  def test_leaves_selection_from_ref_by_name_alone(self):
    ref_to_tuple = building_blocks.Reference('tup', [('a', tf.int32),
                                                     ('b', tf.float32)])
    a_selected = building_blocks.Selection(ref_to_tuple, name='a')
    b_selected = building_blocks.Selection(ref_to_tuple, name='b')

    a_returned, a_transformed = tree_transformations.replace_selection_from_tuple_with_element(
        a_selected)
    b_returned, b_transformed = tree_transformations.replace_selection_from_tuple_with_element(
        b_selected)

    self.assertFalse(a_transformed)
    self.assertEqual(a_returned.proto, a_selected.proto)
    self.assertFalse(b_transformed)
    self.assertEqual(b_returned.proto, b_selected.proto)

  def test_by_index_grabs_correct_element(self):
    x_data = building_blocks.Data('x', tf.int32)
    y_data = building_blocks.Data('y', [('a', tf.float32)])
    tup = building_blocks.Struct([x_data, y_data])
    x_selected = building_blocks.Selection(tup, index=0)
    y_selected = building_blocks.Selection(tup, index=1)

    collapsed_selection_x, x_transformed = tree_transformations.replace_selection_from_tuple_with_element(
        x_selected)
    collapsed_selection_y, y_transformed = tree_transformations.replace_selection_from_tuple_with_element(
        y_selected)

    self.assertTrue(x_transformed)
    self.assertTrue(y_transformed)
    self.assertEqual(collapsed_selection_x.proto, x_data.proto)
    self.assertEqual(collapsed_selection_y.proto, y_data.proto)

  def test_by_name_grabs_correct_element(self):
    x_data = building_blocks.Data('x', tf.int32)
    y_data = building_blocks.Data('y', [('a', tf.float32)])
    tup = building_blocks.Struct([('a', x_data), ('b', y_data)])
    x_selected = building_blocks.Selection(tup, name='a')
    y_selected = building_blocks.Selection(tup, name='b')

    collapsed_selection_x, x_transformed = tree_transformations.replace_selection_from_tuple_with_element(
        x_selected)
    collapsed_selection_y, y_transformed = tree_transformations.replace_selection_from_tuple_with_element(
        y_selected)

    self.assertTrue(x_transformed)
    self.assertTrue(y_transformed)
    self.assertEqual(collapsed_selection_x.proto, x_data.proto)
    self.assertEqual(collapsed_selection_y.proto, y_data.proto)


class UniquifyReferenceNamesTest(test_case.TestCase):

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      tree_transformations.uniquify_reference_names(None)

  def test_renames_lambda_but_not_unbound_reference(self):
    ref = building_blocks.Reference('x', tf.int32)
    lambda_binding_y = building_blocks.Lambda('y', tf.float32, ref)

    transformed_comp, modified = tree_transformations.uniquify_reference_names(
        lambda_binding_y)

    self.assertEqual(lambda_binding_y.compact_representation(), '(y -> x)')
    self.assertEqual(transformed_comp.compact_representation(), '(_var1 -> x)')
    self.assertEqual(transformed_comp.type_signature,
                     lambda_binding_y.type_signature)
    self.assertTrue(modified)

  def test_single_level_block(self):
    ref = building_blocks.Reference('a', tf.int32)
    data = building_blocks.Data('data', tf.int32)
    block = building_blocks.Block((('a', data), ('a', ref), ('a', ref)), ref)

    transformed_comp, modified = tree_transformations.uniquify_reference_names(
        block)

    self.assertEqual(block.compact_representation(),
                     '(let a=data,a=a,a=a in a)')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let _var1=data,_var2=_var1,_var3=_var2 in _var3)')
    tree_analysis.check_has_unique_names(transformed_comp)
    self.assertTrue(modified)

  def test_nested_blocks(self):
    x_ref = building_blocks.Reference('a', tf.int32)
    data = building_blocks.Data('data', tf.int32)
    block1 = building_blocks.Block([('a', data), ('a', x_ref)], x_ref)
    block2 = building_blocks.Block([('a', data), ('a', x_ref)], block1)

    transformed_comp, modified = tree_transformations.uniquify_reference_names(
        block2)

    self.assertEqual(block2.compact_representation(),
                     '(let a=data,a=a in (let a=data,a=a in a))')
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(let _var1=data,_var2=_var1 in (let _var3=data,_var4=_var3 in _var4))')
    tree_analysis.check_has_unique_names(transformed_comp)
    self.assertTrue(modified)

  def test_nested_lambdas(self):
    data = building_blocks.Data('data', tf.int32)
    input1 = building_blocks.Reference('a', data.type_signature)
    first_level_call = building_blocks.Call(
        building_blocks.Lambda('a', input1.type_signature, input1), data)
    input2 = building_blocks.Reference('b', first_level_call.type_signature)
    second_level_call = building_blocks.Call(
        building_blocks.Lambda('b', input2.type_signature, input2),
        first_level_call)

    transformed_comp, modified = tree_transformations.uniquify_reference_names(
        second_level_call)

    self.assertEqual(transformed_comp.compact_representation(),
                     '(_var1 -> _var1)((_var2 -> _var2)(data))')
    tree_analysis.check_has_unique_names(transformed_comp)
    self.assertTrue(modified)

  def test_block_lambda_block_lambda(self):
    x_ref = building_blocks.Reference('a', tf.int32)
    inner_lambda = building_blocks.Lambda('a', tf.int32, x_ref)
    called_lambda = building_blocks.Call(inner_lambda, x_ref)
    lower_block = building_blocks.Block([('a', x_ref), ('a', x_ref)],
                                        called_lambda)
    second_lambda = building_blocks.Lambda('a', tf.int32, lower_block)
    second_call = building_blocks.Call(second_lambda, x_ref)
    data = building_blocks.Data('data', tf.int32)
    last_block = building_blocks.Block([('a', data), ('a', x_ref)], second_call)

    transformed_comp, modified = tree_transformations.uniquify_reference_names(
        last_block)

    self.assertEqual(
        last_block.compact_representation(),
        '(let a=data,a=a in (a -> (let a=a,a=a in (a -> a)(a)))(a))')
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(let _var1=data,_var2=_var1 in (_var3 -> (let _var4=_var3,_var5=_var4 in (_var6 -> _var6)(_var5)))(_var2))'
    )
    tree_analysis.check_has_unique_names(transformed_comp)
    self.assertTrue(modified)

  def test_blocks_nested_inside_of_locals(self):
    data = building_blocks.Data('data', tf.int32)
    lower_block = building_blocks.Block([('a', data)], data)
    middle_block = building_blocks.Block([('a', lower_block)], data)
    higher_block = building_blocks.Block([('a', middle_block)], data)
    y_ref = building_blocks.Reference('a', tf.int32)
    lower_block_with_y_ref = building_blocks.Block([('a', y_ref)], data)
    middle_block_with_y_ref = building_blocks.Block(
        [('a', lower_block_with_y_ref)], data)
    higher_block_with_y_ref = building_blocks.Block(
        [('a', middle_block_with_y_ref)], data)
    multiple_bindings_highest_block = building_blocks.Block(
        [('a', higher_block),
         ('a', higher_block_with_y_ref)], higher_block_with_y_ref)

    transformed_comp, modified = tree_transformations.uniquify_reference_names(
        multiple_bindings_highest_block)

    self.assertEqual(higher_block.compact_representation(),
                     '(let a=(let a=(let a=data in data) in data) in data)')
    self.assertEqual(higher_block_with_y_ref.compact_representation(),
                     '(let a=(let a=(let a=a in data) in data) in data)')
    self.assertEqual(transformed_comp.locals[0][0], '_var4')
    self.assertEqual(
        transformed_comp.locals[0][1].compact_representation(),
        '(let _var3=(let _var2=(let _var1=data in data) in data) in data)')
    self.assertEqual(transformed_comp.locals[1][0], '_var8')
    self.assertEqual(
        transformed_comp.locals[1][1].compact_representation(),
        '(let _var7=(let _var6=(let _var5=_var4 in data) in data) in data)')
    self.assertEqual(
        transformed_comp.result.compact_representation(),
        '(let _var11=(let _var10=(let _var9=_var8 in data) in data) in data)')
    tree_analysis.check_has_unique_names(transformed_comp)
    self.assertTrue(modified)

  def test_renames_names_ignores_existing_names(self):
    data = building_blocks.Data('data', tf.int32)
    block = building_blocks.Block([('a', data), ('b', data)], data)
    comp = block

    transformed_comp, modified = tree_transformations.uniquify_reference_names(
        comp)

    self.assertEqual(block.compact_representation(),
                     '(let a=data,b=data in data)')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let _var1=data,_var2=data in data)')
    self.assertTrue(modified)

    transformed_comp, modified = tree_transformations.uniquify_reference_names(
        comp)

    self.assertEqual(transformed_comp.compact_representation(),
                     '(let _var1=data,_var2=data in data)')
    self.assertTrue(modified)


def _is_called_graph_pattern(comp):
  return (comp.is_call() and comp.function.is_compiled_computation() and
          comp.argument.is_reference())


class StripPlacementTest(test_case.TestCase, parameterized.TestCase):

  def assert_has_no_intrinsics_nor_federated_types(self, comp):

    def _check(x):
      if x.type_signature.is_federated():
        raise AssertionError(f'Unexpected federated type: {x.type_signature}')
      if x.is_intrinsic():
        raise AssertionError(f'Unexpected intrinsic: {x}')

    tree_analysis.visit_postorder(comp, _check)

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      tree_transformations.strip_placement(None)

  def test_computation_non_federated_type(self):
    before = building_blocks.Data('x', tf.int32)
    after, modified = tree_transformations.strip_placement(before)
    self.assertEqual(before, after)
    self.assertFalse(modified)

  def test_raises_disallowed_intrinsic(self):
    fed_ref = building_blocks.Reference(
        'x', computation_types.FederatedType(tf.int32, placements.SERVER))
    broadcaster = building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_BROADCAST.uri,
        computation_types.FunctionType(
            fed_ref.type_signature,
            computation_types.FederatedType(
                fed_ref.type_signature.member,
                placements.CLIENTS,
                all_equal=True)))
    called_broadcast = building_blocks.Call(broadcaster, fed_ref)
    with self.assertRaises(ValueError):
      tree_transformations.strip_placement(called_broadcast)

  def test_raises_multiple_placements(self):
    server_placed_data = building_blocks.Reference(
        'x', computation_types.at_server(tf.int32))
    clients_placed_data = building_blocks.Reference(
        'y', computation_types.at_clients(tf.int32))
    block_holding_both = building_blocks.Block([('x', server_placed_data)],
                                               clients_placed_data)
    with self.assertRaisesRegex(ValueError, 'multiple different placements'):
      tree_transformations.strip_placement(block_holding_both)

  def test_passes_unbound_type_signature_obscured_under_block(self):
    fed_ref = building_blocks.Reference(
        'x', computation_types.FederatedType(tf.int32, placements.SERVER))
    block = building_blocks.Block(
        [('y', fed_ref), ('x', building_blocks.Data('whimsy', tf.int32)),
         ('z', building_blocks.Reference('x', tf.int32))],
        building_blocks.Reference('y', fed_ref.type_signature))
    tree_transformations.strip_placement(block)

  def test_passes_noarg_lambda(self):
    lam = building_blocks.Lambda(None, None,
                                 building_blocks.Data('a', tf.int32))
    fed_int_type = computation_types.FederatedType(tf.int32, placements.SERVER)
    fed_eval = building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_EVAL_AT_SERVER.uri,
        computation_types.FunctionType(lam.type_signature, fed_int_type))
    called_eval = building_blocks.Call(fed_eval, lam)
    tree_transformations.strip_placement(called_eval)

  def test_removes_federated_types_under_function(self):
    int_type = tf.int32
    server_int_type = computation_types.at_server(int_type)
    int_ref = building_blocks.Reference('x', int_type)
    int_id = building_blocks.Lambda('x', int_type, int_ref)
    fed_ref = building_blocks.Reference('x', server_int_type)
    applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, fed_ref)
    before = building_block_factory.create_federated_map_or_apply(
        int_id, applied_id)
    after, modified = tree_transformations.strip_placement(before)
    self.assertTrue(modified)
    self.assert_has_no_intrinsics_nor_federated_types(after)

  def test_strip_placement_removes_federated_applys(self):
    int_type = computation_types.TensorType(tf.int32)
    server_int_type = computation_types.at_server(int_type)
    int_ref = building_blocks.Reference('x', int_type)
    int_id = building_blocks.Lambda('x', int_type, int_ref)
    fed_ref = building_blocks.Reference('x', server_int_type)
    applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, fed_ref)
    before = building_block_factory.create_federated_map_or_apply(
        int_id, applied_id)
    after, modified = tree_transformations.strip_placement(before)
    self.assertTrue(modified)
    self.assert_has_no_intrinsics_nor_federated_types(after)
    self.assert_types_identical(before.type_signature, server_int_type)
    self.assert_types_identical(after.type_signature, int_type)
    self.assertEqual(
        before.compact_representation(),
        'federated_apply(<(x -> x),federated_apply(<(x -> x),x>)>)')
    self.assertEqual(after.compact_representation(), '(x -> x)((x -> x)(x))')

  def test_strip_placement_removes_federated_maps(self):
    int_type = computation_types.TensorType(tf.int32)
    clients_int_type = computation_types.at_clients(int_type)
    int_ref = building_blocks.Reference('x', int_type)
    int_id = building_blocks.Lambda('x', int_type, int_ref)
    fed_ref = building_blocks.Reference('x', clients_int_type)
    applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, fed_ref)
    before = building_block_factory.create_federated_map_or_apply(
        int_id, applied_id)
    after, modified = tree_transformations.strip_placement(before)
    self.assertTrue(modified)
    self.assert_has_no_intrinsics_nor_federated_types(after)
    self.assert_types_identical(before.type_signature, clients_int_type)
    self.assert_types_identical(after.type_signature, int_type)
    self.assertEqual(before.compact_representation(),
                     'federated_map(<(x -> x),federated_map(<(x -> x),x>)>)')
    self.assertEqual(after.compact_representation(), '(x -> x)((x -> x)(x))')

  def test_unwrap_removes_federated_zips_at_server(self):
    list_type = computation_types.to_type([tf.int32, tf.float32] * 2)
    server_list_type = computation_types.at_server(list_type)
    fed_tuple = building_blocks.Reference('tup', server_list_type)
    unzipped = building_block_factory.create_federated_unzip(fed_tuple)
    before = building_block_factory.create_federated_zip(unzipped)
    after, modified = tree_transformations.strip_placement(before)
    self.assertTrue(modified)
    self.assert_has_no_intrinsics_nor_federated_types(after)
    self.assert_types_identical(before.type_signature, server_list_type)
    self.assert_types_identical(after.type_signature, list_type)

  def test_unwrap_removes_federated_zips_at_clients(self):
    list_type = computation_types.to_type([tf.int32, tf.float32] * 2)
    clients_list_type = computation_types.at_server(list_type)
    fed_tuple = building_blocks.Reference('tup', clients_list_type)
    unzipped = building_block_factory.create_federated_unzip(fed_tuple)
    before = building_block_factory.create_federated_zip(unzipped)
    after, modified = tree_transformations.strip_placement(before)
    self.assertTrue(modified)
    self.assert_has_no_intrinsics_nor_federated_types(after)
    self.assert_types_identical(before.type_signature, clients_list_type)
    self.assert_types_identical(after.type_signature, list_type)

  def test_strip_placement_removes_federated_value_at_server(self):
    int_data = building_blocks.Data('x', tf.int32)
    float_data = building_blocks.Data('x', tf.float32)
    fed_int = building_block_factory.create_federated_value(
        int_data, placements.SERVER)
    fed_float = building_block_factory.create_federated_value(
        float_data, placements.SERVER)
    tup = building_blocks.Struct([fed_int, fed_float], container_type=tuple)
    before = building_block_factory.create_federated_zip(tup)
    after, modified = tree_transformations.strip_placement(before)
    self.assertTrue(modified)
    self.assert_has_no_intrinsics_nor_federated_types(after)
    tuple_type = computation_types.StructWithPythonType([(None, tf.int32),
                                                         (None, tf.float32)],
                                                        tuple)
    self.assert_types_identical(before.type_signature,
                                computation_types.at_server(tuple_type))
    self.assert_types_identical(after.type_signature, tuple_type)

  def test_strip_placement_federated_value_at_clients(self):
    int_data = building_blocks.Data('x', tf.int32)
    float_data = building_blocks.Data('x', tf.float32)
    fed_int = building_block_factory.create_federated_value(
        int_data, placements.CLIENTS)
    fed_float = building_block_factory.create_federated_value(
        float_data, placements.CLIENTS)
    tup = building_blocks.Struct([fed_int, fed_float], container_type=tuple)
    before = building_block_factory.create_federated_zip(tup)
    after, modified = tree_transformations.strip_placement(before)
    self.assertTrue(modified)
    self.assert_has_no_intrinsics_nor_federated_types(after)
    tuple_type = computation_types.StructWithPythonType([(None, tf.int32),
                                                         (None, tf.float32)],
                                                        tuple)
    self.assert_types_identical(before.type_signature,
                                computation_types.at_clients(tuple_type))
    self.assert_types_identical(after.type_signature, tuple_type)

  def test_strip_placement_with_called_lambda(self):
    int_type = computation_types.TensorType(tf.int32)
    server_int_type = computation_types.at_server(int_type)
    federated_ref = building_blocks.Reference('outer', server_int_type)
    inner_federated_ref = building_blocks.Reference('inner', server_int_type)
    identity_lambda = building_blocks.Lambda('inner', server_int_type,
                                             inner_federated_ref)
    before = building_blocks.Call(identity_lambda, federated_ref)
    after, modified = tree_transformations.strip_placement(before)
    self.assertTrue(modified)
    self.assert_has_no_intrinsics_nor_federated_types(after)
    self.assert_types_identical(before.type_signature, server_int_type)
    self.assert_types_identical(after.type_signature, int_type)

  def test_strip_placement_nested_federated_type(self):
    int_type = computation_types.TensorType(tf.int32)
    server_int_type = computation_types.at_server(int_type)
    tupled_int_type = computation_types.to_type((int_type, int_type))
    tupled_server_int_type = computation_types.to_type(
        (server_int_type, server_int_type))
    fed_ref = building_blocks.Reference('x', server_int_type)
    before = building_blocks.Struct([fed_ref, fed_ref], container_type=tuple)
    after, modified = tree_transformations.strip_placement(before)
    self.assertTrue(modified)
    self.assert_has_no_intrinsics_nor_federated_types(after)
    self.assert_types_identical(before.type_signature, tupled_server_int_type)
    self.assert_types_identical(after.type_signature, tupled_int_type)


if __name__ == '__main__':
  test_case.main()
