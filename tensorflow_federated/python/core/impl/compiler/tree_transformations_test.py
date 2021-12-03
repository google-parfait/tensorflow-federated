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
from tensorflow_federated.python.core.impl.compiler import transformation_utils
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
  ref = building_blocks.Reference('b', federated_type)
  called_federated_broadcast = building_block_factory.create_federated_broadcast(
      ref)
  called_federated_map = building_block_factory.create_federated_map(
      compiled, called_federated_broadcast)
  called_federated_mean = building_block_factory.create_federated_mean(
      called_federated_map, None)
  tup = building_blocks.Struct([called_federated_mean, called_federated_mean])
  return building_blocks.Lambda('b', tf.int32, tup)


class ExtractComputationsTest(TransformTestBase):

  def transform(self, comp):
    return tree_transformations.extract_computations(comp)

  def test_raises_type_error_with_none(self):
    with self.assertRaises(TypeError):
      tree_transformations.extract_computations(None)

  def test_raises_value_error_with_non_unique_variable_names(self):
    data = building_blocks.Data('data', tf.int32)
    block = building_blocks.Block([('a', data), ('a', data)], data)
    with self.assertRaises(ValueError):
      tree_transformations.extract_computations(block)

  def test_extracts_from_no_arg_lambda(self):
    data = building_blocks.Data('data', tf.int32)
    block = building_blocks.Lambda(
        parameter_name=None, parameter_type=None, result=data)
    self.assert_transforms(block,
                           'extract_computations_from_no_arg_lambda.expected')

  def test_extracts_from_no_arg_lambda_to_block(self):
    data = building_blocks.Data('data', tf.int32)
    blk = building_blocks.Block([], data)
    block = building_blocks.Lambda(
        parameter_name=None, parameter_type=None, result=blk)
    self.assert_transforms(
        block, 'extract_computations_from_no_arg_lambda_to_block.expected')

  def test_extracts_from_block_one_comp(self):
    data = building_blocks.Data('data', tf.int32)
    block = building_blocks.Block([('a', data)], data)
    self.assert_transforms(block,
                           'extract_computations_from_block_one_comp.expected')

  def test_extracts_from_block_multiple_comps(self):
    data_1 = building_blocks.Data('data', tf.int32)
    data_2 = building_blocks.Data('data', tf.int32)
    data_3 = building_blocks.Data('data', tf.int32)
    tup = building_blocks.Struct([data_2, data_3])
    block = building_blocks.Block([('a', data_1)], tup)
    self.assert_transforms(
        block, 'extract_computations_from_block_multiple_comps.expected')

  def test_extracts_from_call_one_comp(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    data = building_blocks.Data('data', tf.int32)
    call = building_blocks.Call(fn, data)
    self.assert_transforms(call,
                           'extract_computations_from_call_one_comp.expected')

  def test_extracts_from_call_multiple_comps(self):
    fn = compiler_test_utils.create_identity_function('a', [tf.int32, tf.int32])
    data_1 = building_blocks.Data('data', tf.int32)
    data_2 = building_blocks.Data('data', tf.int32)
    tup = building_blocks.Struct([data_1, data_2])
    call = building_blocks.Call(fn, tup)
    self.assert_transforms(
        call, 'extract_computations_from_call_multiple_comps.expected')

  def test_extracts_from_lambda_one_comp(self):
    data = building_blocks.Data('data', tf.int32)
    fn = building_blocks.Lambda('a', tf.int32, data)
    self.assert_transforms(
        fn, 'extract_computations_from_lambda_one_comp.expected')

  def test_extracts_from_lambda_multiple_comps(self):
    data_1 = building_blocks.Data('data', tf.int32)
    data_2 = building_blocks.Data('data', tf.int32)
    tup = building_blocks.Struct([data_1, data_2])
    fn = building_blocks.Lambda('a', tf.int32, tup)
    self.assert_transforms(
        fn, 'extract_computations_from_lambda_multiple_comps.expected')

  def test_extracts_from_selection_one_comp(self):
    data = building_blocks.Data('data', tf.int32)
    tup = building_blocks.Struct([data])
    sel = building_blocks.Selection(tup, index=0)
    self.assert_transforms(
        sel, 'extract_computations_from_selection_one_comp.expected')

  def test_extracts_from_selection_multiple_comps(self):
    data_1 = building_blocks.Data('data', tf.int32)
    data_2 = building_blocks.Data('data', tf.int32)
    tup = building_blocks.Struct([data_1, data_2])
    sel = building_blocks.Selection(tup, index=0)
    self.assert_transforms(
        sel, 'extract_computations_from_selection_multiple_comps.expected')

  def test_extracts_from_tuple_one_comp(self):
    data = building_blocks.Data('data', tf.int32)
    tup = building_blocks.Struct([data])
    self.assert_transforms(tup,
                           'extract_computations_from_tuple_one_comp.expected')

  def test_extracts_from_tuple_multiple_comps(self):
    data_1 = building_blocks.Data('data', tf.int32)
    data_2 = building_blocks.Data('data', tf.int32)
    tup = building_blocks.Struct([data_1, data_2])
    self.assert_transforms(
        tup, 'extract_computations_from_tuple_multiple_comps.expected')

  def test_extracts_from_tuple_named_comps(self):
    data_1 = building_blocks.Data('data', tf.int32)
    data_2 = building_blocks.Data('data', tf.int32)
    tup = building_blocks.Struct([
        ('a', data_1),
        ('b', data_2),
    ])
    self.assert_transforms(
        tup, 'extract_computations_from_tuple_named_comps.expected')

  def test_extracts_federated_aggregate(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    self.assert_transforms(called_intrinsic,
                           'extract_computations_federated_aggregate.expected')

  def test_extracts_federated_broadcast(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_federated_broadcast(
    )
    self.assert_transforms(called_intrinsic,
                           'extract_computations_federated_broadcast.expected')

  def test_extracts_complex_comp(self):
    complex_comp = _create_complex_computation()
    self.assert_transforms(complex_comp,
                           'extract_computations_complex_comp.expected')


class ExtractIntrinsicsTest(test_case.TestCase):

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      tree_transformations.extract_intrinsics(None)

  def test_raises_value_error_with_non_unique_variable_names(self):
    data = building_blocks.Data('data', tf.int32)
    block = building_blocks.Block([('a', data), ('a', data)], data)
    with self.assertRaises(ValueError):
      tree_transformations.extract_intrinsics(block)

  def test_extracts_from_block_result_intrinsic(self):
    data = building_blocks.Data('data', tf.int32)
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    block = building_blocks.Block((('a', data),), called_intrinsic)
    comp = block

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '(let a=data in intrinsic(a))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let a=data,_var1=intrinsic(a) in _var1)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_block_result_block_one_var_unbound(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    ref = building_blocks.Reference('b', called_intrinsic.type_signature)
    block_1 = building_blocks.Block((('b', called_intrinsic),), ref)
    data = building_blocks.Data('data', tf.int32)
    block_2 = building_blocks.Block((('a', data),), block_1)
    comp = block_2

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '(let a=data in (let b=intrinsic(a) in b))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let a=data,b=intrinsic(a) in b)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_block_result_block_multiple_vars_unbound(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    ref = building_blocks.Reference('b', called_intrinsic.type_signature)
    block_1 = building_blocks.Block((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ), ref)
    data = building_blocks.Data('data', tf.int32)
    block_2 = building_blocks.Block((('a', data),), block_1)
    comp = block_2

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(
        comp.compact_representation(),
        '(let a=data in (let b=intrinsic(a),c=intrinsic(a) in b))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let a=data,b=intrinsic(a),c=intrinsic(a) in b)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_block_variables_block_one_var_unbound(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    ref_1 = building_blocks.Reference('b', called_intrinsic.type_signature)
    block_1 = building_blocks.Block((('b', called_intrinsic),), ref_1)
    ref_2 = building_blocks.Reference('c', tf.int32)
    block_2 = building_blocks.Block((('c', block_1),), ref_2)
    comp = block_2

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '(let c=(let b=intrinsic(a) in b) in c)')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let b=intrinsic(a),c=b in c)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_block_variables_block_multiple_vars_unbound(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    ref_1 = building_blocks.Reference('b', called_intrinsic.type_signature)
    block_1 = building_blocks.Block((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ), ref_1)
    ref_2 = building_blocks.Reference('d', tf.int32)
    block_2 = building_blocks.Block((('d', block_1),), ref_2)
    comp = block_2

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '(let d=(let b=intrinsic(a),c=intrinsic(a) in b) in d)')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let b=intrinsic(a),c=intrinsic(a),d=b in d)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_block_variables_block_one_var_bound_by_lambda(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    ref_1 = building_blocks.Reference('b', called_intrinsic.type_signature)
    block_1 = building_blocks.Block((('b', called_intrinsic),), ref_1)
    ref_2 = building_blocks.Reference('c', tf.int32)
    block_2 = building_blocks.Block((('c', block_1),), ref_2)
    fn = building_blocks.Lambda('a', tf.int32, block_2)
    comp = fn

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '(a -> (let c=(let b=intrinsic(a) in b) in c))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(a -> (let b=intrinsic(a),c=b in c))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_block_variables_block_multiple_vars_bound_by_lambda(
      self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    ref_1 = building_blocks.Reference('b', called_intrinsic.type_signature)
    block_1 = building_blocks.Block((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ), ref_1)
    ref_2 = building_blocks.Reference('d', tf.int32)
    block_2 = building_blocks.Block((('d', block_1),), ref_2)
    fn = building_blocks.Lambda('a', tf.int32, block_2)
    comp = fn

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(
        comp.compact_representation(),
        '(a -> (let d=(let b=intrinsic(a),c=intrinsic(a) in b) in d))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(a -> (let b=intrinsic(a),c=intrinsic(a),d=b in d))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_block_variables_block_one_var_bound_by_block(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    ref_1 = building_blocks.Reference('b', called_intrinsic.type_signature)
    block_1 = building_blocks.Block((('b', called_intrinsic),), ref_1)
    data = building_blocks.Data('data', tf.int32)
    ref_2 = building_blocks.Reference('c', tf.int32)
    block_2 = building_blocks.Block((
        ('a', data),
        ('c', block_1),
    ), ref_2)
    comp = block_2

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '(let a=data,c=(let b=intrinsic(a) in b) in c)')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let a=data,b=intrinsic(a),c=b in c)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_block_variables_block_multiple_vars_bound_by_block(
      self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    ref_1 = building_blocks.Reference('b', called_intrinsic.type_signature)
    block_1 = building_blocks.Block((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ), ref_1)
    data = building_blocks.Data('data', tf.int32)
    ref_2 = building_blocks.Reference('d', tf.int32)
    block_2 = building_blocks.Block((
        ('a', data),
        ('d', block_1),
    ), ref_2)
    comp = block_2

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(
        comp.compact_representation(),
        '(let a=data,d=(let b=intrinsic(a),c=intrinsic(a) in b) in d)')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let a=data,b=intrinsic(a),c=intrinsic(a),d=b in d)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_call_intrinsic(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='b')
    call = building_blocks.Call(fn, called_intrinsic)
    comp = call

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(), '(a -> a)(intrinsic(b))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let _var1=intrinsic(b) in (a -> a)(_var1))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_call_block_one_var(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='b')
    ref = building_blocks.Reference('c', called_intrinsic.type_signature)
    block = building_blocks.Block((('c', called_intrinsic),), ref)
    call = building_blocks.Call(fn, block)
    comp = call

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '(a -> a)((let c=intrinsic(b) in c))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let c=intrinsic(b) in (a -> a)(c))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_call_block_multiple_vars(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='b')
    ref = building_blocks.Reference('c', called_intrinsic.type_signature)
    block = building_blocks.Block((
        ('c', called_intrinsic),
        ('d', called_intrinsic),
    ), ref)
    call = building_blocks.Call(fn, block)
    comp = call

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '(a -> a)((let c=intrinsic(b),d=intrinsic(b) in c))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let c=intrinsic(b),d=intrinsic(b) in (a -> a)(c))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_lambda_intrinsic_unbound(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    fn = building_blocks.Lambda('b', tf.int32, called_intrinsic)
    comp = fn

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(), '(b -> intrinsic(a))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let _var1=intrinsic(a) in (b -> _var1))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_lambda_intrinsic_bound_by_lambda(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    fn = building_blocks.Lambda('a', tf.int32, called_intrinsic)
    comp = fn

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(), '(a -> intrinsic(a))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(a -> (let _var1=intrinsic(a) in _var1))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_lambda_block_one_var_unbound(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    ref = building_blocks.Reference('b', called_intrinsic.type_signature)
    block = building_blocks.Block((('b', called_intrinsic),), ref)
    fn = building_blocks.Lambda('c', tf.int32, block)
    comp = fn

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '(c -> (let b=intrinsic(a) in b))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let b=intrinsic(a) in (c -> b))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_lambda_block_multiple_vars_unbound(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    ref = building_blocks.Reference('b', called_intrinsic.type_signature)
    block = building_blocks.Block((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ), ref)
    fn = building_blocks.Lambda('d', tf.int32, block)
    comp = fn

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '(d -> (let b=intrinsic(a),c=intrinsic(a) in b))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let b=intrinsic(a),c=intrinsic(a) in (d -> b))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_lambda_block_first_var_unbound(self):
    called_intrinsic_1 = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    called_intrinsic_2 = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='b')
    ref = building_blocks.Reference('c', called_intrinsic_2.type_signature)
    block = building_blocks.Block((
        ('c', called_intrinsic_1),
        ('d', called_intrinsic_2),
    ), ref)
    fn = building_blocks.Lambda('b', tf.int32, block)
    comp = fn

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '(b -> (let c=intrinsic(a),d=intrinsic(b) in c))')
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(let c=intrinsic(a) in (b -> (let d=intrinsic(b) in c)))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_lambda_block_last_var_unbound(self):
    called_intrinsic_1 = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    called_intrinsic_2 = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='b')
    ref = building_blocks.Reference('c', called_intrinsic_2.type_signature)
    block = building_blocks.Block((
        ('c', called_intrinsic_1),
        ('d', called_intrinsic_2),
    ), ref)
    fn = building_blocks.Lambda('a', tf.int32, block)
    comp = fn

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '(a -> (let c=intrinsic(a),d=intrinsic(b) in c))')
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(let d=intrinsic(b) in (a -> (let c=intrinsic(a) in c)))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_lambda_block_one_var_bound_by_block(self):
    data = building_blocks.Data('data', tf.int32)
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    ref = building_blocks.Reference('b', called_intrinsic.type_signature)
    block = building_blocks.Block((
        ('a', data),
        ('b', called_intrinsic),
    ), ref)
    fn = building_blocks.Lambda('c', tf.int32, block)
    comp = fn

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '(c -> (let a=data,b=intrinsic(a) in b))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let a=data,b=intrinsic(a) in (c -> b))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_lambda_block_multiple_vars_bound_by_block(self):
    data = building_blocks.Data('data', tf.int32)
    called_intrinsic_1 = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    called_intrinsic_2 = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='b')
    ref = building_blocks.Reference('c', called_intrinsic_2.type_signature)
    block = building_blocks.Block((
        ('a', data),
        ('b', called_intrinsic_1),
        ('c', called_intrinsic_2),
    ), ref)
    fn = building_blocks.Lambda('d', tf.int32, block)
    comp = fn

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '(d -> (let a=data,b=intrinsic(a),c=intrinsic(b) in c))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let a=data,b=intrinsic(a),c=intrinsic(b) in (d -> c))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_selection_intrinsic(self):
    parameter_type = computation_types.StructType((tf.int32, tf.int32))
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a', parameter_type=parameter_type)
    sel = building_blocks.Selection(called_intrinsic, index=0)
    comp = sel

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(), 'intrinsic(a)[0]')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let _var1=intrinsic(a) in _var1[0])')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_selection_named_intrinsic(self):
    parameter_type = computation_types.StructType((
        ('a', tf.int32),
        ('b', tf.int32),
    ))
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='c', parameter_type=parameter_type)
    sel = building_blocks.Selection(called_intrinsic, index=0)
    comp = sel

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(), 'intrinsic(c)[0]')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let _var1=intrinsic(c) in _var1[0])')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_selection_block_one_var(self):
    parameter_type = computation_types.StructType((tf.int32, tf.int32))
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a', parameter_type=parameter_type)
    ref = building_blocks.Reference('b', called_intrinsic.type_signature)
    block = building_blocks.Block((('b', called_intrinsic),), ref)
    sel = building_blocks.Selection(block, index=0)
    comp = sel

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '(let b=intrinsic(a) in b)[0]')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let b=intrinsic(a) in b[0])')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_selection_block_multiple_vars(self):
    parameter_type = computation_types.StructType((tf.int32, tf.int32))
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a', parameter_type=parameter_type)
    ref = building_blocks.Reference('b', called_intrinsic.type_signature)
    block = building_blocks.Block((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ), ref)
    sel = building_blocks.Selection(block, index=0)
    comp = sel

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '(let b=intrinsic(a),c=intrinsic(a) in b)[0]')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let b=intrinsic(a),c=intrinsic(a) in b[0])')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_tuple_one_intrinsic(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    tup = building_blocks.Struct((called_intrinsic,))
    comp = tup

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(), '<intrinsic(a)>')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let _var1=intrinsic(a) in <_var1>)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_tuple_multiple_intrinsics(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    tup = building_blocks.Struct((called_intrinsic, called_intrinsic))
    comp = tup

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '<intrinsic(a),intrinsic(a)>')
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(let _var1=intrinsic(a),_var2=intrinsic(a) in <_var1,_var2>)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_tuple_named_intrinsics(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    tup = building_blocks.Struct((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ))
    comp = tup

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '<b=intrinsic(a),c=intrinsic(a)>')
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(let _var1=intrinsic(a),_var2=intrinsic(a) in <b=_var1,c=_var2>)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_tuple_one_block_one_var(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    ref = building_blocks.Reference('b', called_intrinsic.type_signature)
    block = building_blocks.Block((('b', called_intrinsic),), ref)
    tup = building_blocks.Struct((block,))
    comp = tup

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '<(let b=intrinsic(a) in b)>')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let b=intrinsic(a),_var1=b in <_var1>)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_tuple_one_block_multiple_vars(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    ref = building_blocks.Reference('b', called_intrinsic.type_signature)
    block = building_blocks.Block((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ), ref)
    tup = building_blocks.Struct((block,))
    comp = tup

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '<(let b=intrinsic(a),c=intrinsic(a) in b)>')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let b=intrinsic(a),c=intrinsic(a),_var1=b in <_var1>)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_tuple_multiple_blocks_one_var(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    ref_1 = building_blocks.Reference('b', called_intrinsic.type_signature)
    block_1 = building_blocks.Block((('b', called_intrinsic),), ref_1)
    ref_2 = building_blocks.Reference('d', called_intrinsic.type_signature)
    block_2 = building_blocks.Block((('d', called_intrinsic),), ref_2)
    tup = building_blocks.Struct((block_1, block_2))
    comp = tup

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '<(let b=intrinsic(a) in b),(let d=intrinsic(a) in d)>')
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(let b=intrinsic(a),_var1=b,d=intrinsic(a),_var2=d in <_var1,_var2>)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_tuple_multiple_blocks_multiple_vars(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    ref_1 = building_blocks.Reference('b', called_intrinsic.type_signature)
    block_1 = building_blocks.Block((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ), ref_1)
    ref_2 = building_blocks.Reference('d', called_intrinsic.type_signature)
    block_2 = building_blocks.Block((
        ('d', called_intrinsic),
        ('e', called_intrinsic),
    ), ref_2)
    tup = building_blocks.Struct((block_1, block_2))
    comp = tup

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(
        comp.compact_representation(),
        '<(let b=intrinsic(a),c=intrinsic(a) in b),(let d=intrinsic(a),e=intrinsic(a) in d)>'
    )
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(let b=intrinsic(a),c=intrinsic(a),_var1=b,d=intrinsic(a),e=intrinsic(a),_var2=d in <_var1,_var2>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_one_intrinsic(self):
    data = building_blocks.Data('data', tf.int32)
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    tup = building_blocks.Struct((called_intrinsic,))
    sel = building_blocks.Selection(tup, index=0)
    block = building_blocks.Block((('b', data),), sel)
    fn_1 = compiler_test_utils.create_identity_function('c', tf.int32)
    call_1 = building_blocks.Call(fn_1, block)
    fn_2 = compiler_test_utils.create_identity_function('d', tf.int32)
    call_2 = building_blocks.Call(fn_2, call_1)
    fn_3 = building_blocks.Lambda('e', tf.int32, call_2)
    comp = fn_3

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(
        comp.compact_representation(),
        '(e -> (d -> d)((c -> c)((let b=data in <intrinsic(a)>[0]))))')
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(let b=data,_var1=intrinsic(a) in (e -> (d -> d)((c -> c)(<_var1>[0]))))'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_multiple_intrinsics(self):
    data = building_blocks.Data('data', tf.int32)
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    tup = building_blocks.Struct((called_intrinsic, called_intrinsic))
    sel = building_blocks.Selection(tup, index=0)
    block = building_blocks.Block((
        ('b', data),
        ('c', called_intrinsic),
    ), sel)
    fn_1 = compiler_test_utils.create_identity_function('d', tf.int32)
    call_1 = building_blocks.Call(fn_1, block)
    fn_2 = compiler_test_utils.create_identity_function('e', tf.int32)
    call_2 = building_blocks.Call(fn_2, call_1)
    fn_3 = building_blocks.Lambda('f', tf.int32, call_2)
    comp = fn_3

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(
        comp.compact_representation(),
        '(f -> (e -> e)((d -> d)((let b=data,c=intrinsic(a) in <intrinsic(a),intrinsic(a)>[0]))))'
    )
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(let b=data,c=intrinsic(a),_var1=intrinsic(a),_var2=intrinsic(a) in (f -> (e -> e)((d -> d)(<_var1,_var2>[0]))))'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_multiple_intrinsics_dependent_bindings(self):
    called_intrinsic_1 = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    fn_1 = building_blocks.Lambda('a', tf.int32, called_intrinsic_1)
    data = building_blocks.Data('data', tf.int32)
    call_1 = building_blocks.Call(fn_1, data)
    intrinsic_type = computation_types.FunctionType(tf.int32, tf.int32)
    intrinsic = building_blocks.Intrinsic('intrinsic', intrinsic_type)
    called_intrinsic_2 = building_blocks.Call(intrinsic, call_1)
    fn_2 = building_blocks.Lambda('b', tf.int32, called_intrinsic_2)
    comp = fn_2

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(comp.compact_representation(),
                     '(b -> intrinsic((a -> intrinsic(a))(data)))')
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(let _var2=intrinsic((a -> (let _var1=intrinsic(a) in _var1))(data)) in (b -> _var2))'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_does_not_extract_from_block_variables_intrinsic(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    ref = building_blocks.Reference('b', called_intrinsic.type_signature)
    block = building_blocks.Block((('b', called_intrinsic),), ref)
    comp = block

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(transformed_comp.compact_representation(),
                     comp.compact_representation())
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let b=intrinsic(a) in b)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_does_not_extract_from_lambda_block_one_var_bound_by_lambda(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    ref = building_blocks.Reference('b', called_intrinsic.type_signature)
    block = building_blocks.Block((('b', called_intrinsic),), ref)
    fn = building_blocks.Lambda('a', tf.int32, block)
    comp = fn

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(transformed_comp.compact_representation(),
                     comp.compact_representation())
    self.assertEqual(transformed_comp.compact_representation(),
                     '(a -> (let b=intrinsic(a) in b))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_does_not_extract_from_lambda_block_multiple_vars_bound_by_lambda(
      self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_intrinsic(
        parameter_name='a')
    ref = building_blocks.Reference('b', called_intrinsic.type_signature)
    block = building_blocks.Block((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ), ref)
    fn = building_blocks.Lambda('a', tf.int32, block)
    comp = fn

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(transformed_comp.compact_representation(),
                     comp.compact_representation())
    self.assertEqual(transformed_comp.compact_representation(),
                     '(a -> (let b=intrinsic(a),c=intrinsic(a) in b))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_does_not_extract_called_lambda(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    arg = building_blocks.Data('data', tf.int32)
    call = building_blocks.Call(fn, arg)
    comp = call

    transformed_comp, modified = tree_transformations.extract_intrinsics(comp)

    self.assertEqual(transformed_comp.compact_representation(),
                     comp.compact_representation())
    self.assertEqual(transformed_comp.compact_representation(),
                     '(a -> a)(data)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)


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


class InlineSelectionsFromTuplesTest(test_case.TestCase):

  def test_should_transform_selection_from_tuple(self):
    tup = building_blocks.Struct([building_blocks.Data('x', tf.int32)])
    sel = building_blocks.Selection(tup, index=0)
    selection_inliner = tree_transformations.InlineSelectionsFromTuples()
    symbol_tree = transformation_utils.SymbolTree(
        transformation_utils.ReferenceCounter)
    self.assertTrue(selection_inliner.should_transform(sel, symbol_tree))

  def test_should_transform_selection_from_reference_to_bound_tuple(self):
    tup = building_blocks.Struct([building_blocks.Data('x', tf.int32)])
    ref = building_blocks.Reference('a', tup.type_signature)
    sel = building_blocks.Selection(ref, index=0)
    symbol_tree = transformation_utils.SymbolTree(
        transformation_utils.ReferenceCounter)
    symbol_tree.ingest_variable_binding('a', tup)
    selection_inliner = tree_transformations.InlineSelectionsFromTuples()
    self.assertTrue(selection_inliner.should_transform(sel, symbol_tree))

  def test_should_not_transform_selection_from_unbound_reference(self):
    tup = building_blocks.Struct([building_blocks.Data('x', tf.int32)])
    ref = building_blocks.Reference('a', tup.type_signature)
    sel = building_blocks.Selection(ref, index=0)
    symbol_tree = transformation_utils.SymbolTree(
        transformation_utils.ReferenceCounter)
    symbol_tree.ingest_variable_binding('a', tup)
    selection_inliner = tree_transformations.InlineSelectionsFromTuples()
    self.assertTrue(selection_inliner.should_transform(sel, symbol_tree))

  def test_reduces_selection_from_direct_tuple_by_index(self):
    data = building_blocks.Data('x', tf.int32)
    tup = building_blocks.Struct([data])
    sel = building_blocks.Selection(tup, index=0)
    collapsed, modified = tree_transformations.inline_selections_from_tuple(sel)
    self.assertTrue(modified)
    self.assertEqual(data.compact_representation(),
                     collapsed.compact_representation())

  def test_reduces_selection_from_direct_tuple_by_name(self):
    data = building_blocks.Data('x', tf.int32)
    tup = building_blocks.Struct([('a', data)])
    sel = building_blocks.Selection(tup, name='a')
    collapsed, modified = tree_transformations.inline_selections_from_tuple(sel)
    self.assertTrue(modified)
    self.assertEqual(data.compact_representation(),
                     collapsed.compact_representation())

  def test_inlines_selection_from_reference_to_tuple_by_index(self):
    data = building_blocks.Data('x', tf.int32)
    tup = building_blocks.Struct([data])
    ref_to_b = building_blocks.Reference('b', tup.type_signature)
    sel = building_blocks.Selection(ref_to_b, index=0)
    blk = building_blocks.Block([('b', tup)], sel)
    collapsed, modified = tree_transformations.inline_selections_from_tuple(blk)

    expected_blk = building_blocks.Block([('b', tup)], data)

    self.assertTrue(modified)
    self.assertEqual(collapsed.compact_representation(),
                     expected_blk.compact_representation())

  def test_inlines_selection_from_reference_to_tuple_by_name(self):
    data = building_blocks.Data('x', tf.int32)
    tup = building_blocks.Struct([('a', data)])
    ref_to_b = building_blocks.Reference('b', tup.type_signature)
    sel = building_blocks.Selection(ref_to_b, name='a')
    blk = building_blocks.Block([('b', tup)], sel)
    collapsed, modified = tree_transformations.inline_selections_from_tuple(blk)

    expected_blk = building_blocks.Block([('b', tup)], data)

    self.assertTrue(modified)
    self.assertEqual(collapsed.compact_representation(),
                     expected_blk.compact_representation())


class MergeChainedBlocksTest(test_case.TestCase):

  def test_fails_on_none(self):
    with self.assertRaises(TypeError):
      tree_transformations.merge_chained_blocks(None)

  def test_raises_non_unique_names(self):
    data = building_blocks.Data('a', tf.int32)
    x_ref = building_blocks.Reference('x', tf.int32)
    block1 = building_blocks.Block([('x', data)], x_ref)
    block2 = building_blocks.Block([('x', data)], block1)
    with self.assertRaises(ValueError):
      _ = tree_transformations.merge_chained_blocks(block2)

  def test_single_level_of_nesting(self):
    input1 = building_blocks.Reference('input1', tf.int32)
    result = building_blocks.Reference('result', tf.int32)
    block1 = building_blocks.Block([('result', input1)], result)
    input2 = building_blocks.Data('input2', tf.int32)
    block2 = building_blocks.Block([('input1', input2)], block1)
    self.assertEqual(block2.compact_representation(),
                     '(let input1=input2 in (let result=input1 in result))')
    merged_blocks, modified = tree_transformations.merge_chained_blocks(block2)
    self.assertEqual(merged_blocks.compact_representation(),
                     '(let input1=input2,result=input1 in result)')
    self.assertTrue(modified)

  def test_leaves_names(self):
    input1 = building_blocks.Data('input1', tf.int32)
    result_tuple = building_blocks.Struct([
        ('a', building_blocks.Data('result_a', tf.int32)),
        ('b', building_blocks.Data('result_b', tf.int32))
    ])
    block1 = building_blocks.Block([('x', input1)], result_tuple)
    result_block = block1
    input2 = building_blocks.Data('input2', tf.int32)
    block2 = building_blocks.Block([('y', input2)], result_block)
    self.assertEqual(
        block2.compact_representation(),
        '(let y=input2 in (let x=input1 in <a=result_a,b=result_b>))')
    merged, modified = tree_transformations.merge_chained_blocks(block2)
    self.assertEqual(merged.compact_representation(),
                     '(let y=input2,x=input1 in <a=result_a,b=result_b>)')
    self.assertTrue(modified)

  def test_leaves_separated_chained_blocks_alone(self):
    input1 = building_blocks.Data('input1', tf.int32)
    result = building_blocks.Data('result', tf.int32)
    block1 = building_blocks.Block([('x', input1)], result)
    result_block = block1
    result_tuple = building_blocks.Struct([result_block])
    input2 = building_blocks.Data('input2', tf.int32)
    block2 = building_blocks.Block([('y', input2)], result_tuple)
    self.assertEqual(block2.compact_representation(),
                     '(let y=input2 in <(let x=input1 in result)>)')
    merged, modified = tree_transformations.merge_chained_blocks(block2)
    self.assertEqual(merged.compact_representation(),
                     '(let y=input2 in <(let x=input1 in result)>)')
    self.assertFalse(modified)

  def test_two_levels_of_nesting(self):
    input1 = building_blocks.Reference('input1', tf.int32)
    result = building_blocks.Reference('result', tf.int32)
    block1 = building_blocks.Block([('result', input1)], result)
    input2 = building_blocks.Reference('input2', tf.int32)
    block2 = building_blocks.Block([('input1', input2)], block1)
    input3 = building_blocks.Data('input3', tf.int32)
    block3 = building_blocks.Block([('input2', input3)], block2)
    self.assertEqual(
        block3.compact_representation(),
        '(let input2=input3 in (let input1=input2 in (let result=input1 in result)))'
    )
    merged_blocks, modified = tree_transformations.merge_chained_blocks(block3)
    self.assertEqual(
        merged_blocks.compact_representation(),
        '(let input2=input3,input1=input2,result=input1 in result)')
    self.assertTrue(modified)

  def test_with_block_bound_to_local(self):
    input1 = building_blocks.Data('input1', tf.int32)
    result = building_blocks.Reference('result', tf.int32)
    block1 = building_blocks.Block([('result', input1)], result)
    input2 = building_blocks.Reference('input2', tf.int32)
    block2 = building_blocks.Block([('input2', block1)], input2)

    merged_blocks, modified = tree_transformations.merge_chained_blocks(block2)

    self.assertEqual(block2.compact_representation(),
                     '(let input2=(let result=input1 in result) in input2)')
    self.assertEqual(merged_blocks.compact_representation(),
                     '(let result=input1,input2=result in input2)')
    self.assertTrue(modified)

  def test_with_block_bound_to_local_in_result(self):
    input1 = building_blocks.Data('input1', tf.int32)
    result = building_blocks.Reference('result', tf.int32)
    block1 = building_blocks.Block([('result', input1)], result)
    input2 = building_blocks.Reference('input2', tf.int32)
    block2 = building_blocks.Block([('input2', block1)], input2)
    input3 = building_blocks.Data('input3', tf.int32)
    block3 = building_blocks.Block([('input3', input3)], block2)

    merged_blocks, modified = tree_transformations.merge_chained_blocks(block3)

    self.assertEqual(
        block3.compact_representation(),
        '(let input3=input3 in (let input2=(let result=input1 in result) in input2))'
    )
    self.assertEqual(
        merged_blocks.compact_representation(),
        '(let input3=input3,result=input1,input2=result in input2)')
    self.assertTrue(modified)


class MergeChainedFederatedMapOrApplysTest(test_case.TestCase):

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      tree_transformations.merge_chained_federated_maps_or_applys(None)

  def test_merges_federated_applys(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.SERVER)
    arg = building_blocks.Data('data', arg_type)
    call = _create_chained_whimsy_federated_applys([fn, fn], arg)
    comp = call

    transformed_comp, modified = tree_transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(
        comp.compact_representation(),
        'federated_apply(<(a -> a),federated_apply(<(a -> a),data>)>)')
    self.assertEqual(
        transformed_comp.compact_representation(),
        'federated_apply(<(let _var1=<(a -> a),(a -> a)> in (_var2 -> _var1[1](_var1[0](_var2)))),data>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), 'int32@SERVER')
    self.assertTrue(modified)

  def test_merges_federated_maps(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    call = _create_chained_whimsy_federated_maps([fn, fn], arg)
    comp = call

    transformed_comp, modified = tree_transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(
        comp.compact_representation(),
        'federated_map(<(a -> a),federated_map(<(a -> a),data>)>)')
    self.assertEqual(
        transformed_comp.compact_representation(),
        'federated_map(<(let _var1=<(a -> a),(a -> a)> in (_var2 -> _var1[1](_var1[0](_var2)))),data>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), '{int32}@CLIENTS')
    self.assertTrue(modified)

  def test_merges_federated_maps_with_different_names(self):
    fn_1 = compiler_test_utils.create_identity_function('a', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    fn_2 = compiler_test_utils.create_identity_function('b', tf.int32)
    call = _create_chained_whimsy_federated_maps([fn_1, fn_2], arg)
    comp = call

    transformed_comp, modified = tree_transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(
        comp.compact_representation(),
        'federated_map(<(b -> b),federated_map(<(a -> a),data>)>)')
    self.assertEqual(
        transformed_comp.compact_representation(),
        'federated_map(<(let _var1=<(a -> a),(b -> b)> in (_var2 -> _var1[1](_var1[0](_var2)))),data>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), '{int32}@CLIENTS')
    self.assertTrue(modified)

  def test_merges_federated_maps_with_different_types(self):
    fn_1 = _create_lambda_to_whimsy_cast('a', tf.int32, tf.float32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    fn_2 = compiler_test_utils.create_identity_function('b', tf.float32)
    call = _create_chained_whimsy_federated_maps([fn_1, fn_2], arg)
    comp = call

    transformed_comp, modified = tree_transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(
        comp.compact_representation(),
        'federated_map(<(b -> b),federated_map(<(a -> data),data>)>)')
    self.assertEqual(
        transformed_comp.compact_representation(),
        'federated_map(<(let _var1=<(a -> data),(b -> b)> in (_var2 -> _var1[1](_var1[0](_var2)))),data>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), '{float32}@CLIENTS')
    self.assertTrue(modified)

  def test_merges_federated_maps_with_named_parameter_type(self):
    parameter_type = [('b', tf.int32), ('c', tf.int32)]
    fn = compiler_test_utils.create_identity_function('a', parameter_type)
    arg_type = computation_types.FederatedType(parameter_type,
                                               placements.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    call = _create_chained_whimsy_federated_maps([fn, fn], arg)
    comp = call

    transformed_comp, modified = tree_transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(
        comp.compact_representation(),
        'federated_map(<(a -> a),federated_map(<(a -> a),data>)>)')
    self.assertEqual(
        transformed_comp.compact_representation(),
        'federated_map(<(let _var1=<(a -> a),(a -> a)> in (_var2 -> _var1[1](_var1[0](_var2)))),data>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature), '{<b=int32,c=int32>}@CLIENTS')
    self.assertTrue(modified)

  def test_merges_federated_maps_with_unbound_references(self):
    ref = building_blocks.Reference('a', tf.int32)
    fn = building_blocks.Lambda('b', tf.int32, ref)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    call = _create_chained_whimsy_federated_maps([fn, fn], arg)
    comp = call

    transformed_comp, modified = tree_transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(
        comp.compact_representation(),
        'federated_map(<(b -> a),federated_map(<(b -> a),data>)>)')
    self.assertEqual(
        transformed_comp.compact_representation(),
        'federated_map(<(let _var1=<(b -> a),(b -> a)> in (_var2 -> _var1[1](_var1[0](_var2)))),data>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), '{int32}@CLIENTS')
    self.assertTrue(modified)

  def test_merges_nested_federated_maps(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    call = _create_chained_whimsy_federated_maps([fn, fn], arg)
    block = compiler_test_utils.create_whimsy_block(call, variable_name='b')
    comp = block

    transformed_comp, modified = tree_transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(
        comp.compact_representation(),
        '(let b=data in federated_map(<(a -> a),federated_map(<(a -> a),data>)>))'
    )
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(let b=data in federated_map(<(let _var1=<(a -> a),(a -> a)> in (_var2 -> _var1[1](_var1[0](_var2)))),data>))'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), '{int32}@CLIENTS')
    self.assertTrue(modified)

  def test_merges_multiple_federated_maps(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    call = _create_chained_whimsy_federated_maps([fn, fn, fn], arg)
    comp = call

    transformed_comp, modified = tree_transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(
        comp.compact_representation(),
        'federated_map(<(a -> a),federated_map(<(a -> a),federated_map(<(a -> a),data>)>)>)'
    )
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        'federated_map(<\n'
        '  (let\n'
        '    _var3=<\n'
        '      (let\n'
        '        _var1=<\n'
        '          (a -> a),\n'
        '          (a -> a)\n'
        '        >\n'
        '       in (_var2 -> _var1[1](_var1[0](_var2)))),\n'
        '      (a -> a)\n'
        '    >\n'
        '   in (_var4 -> _var3[1](_var3[0](_var4)))),\n'
        '  data\n'
        '>)'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), '{int32}@CLIENTS')
    self.assertTrue(modified)

  def test_does_not_merge_one_federated_map(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    call = building_block_factory.create_federated_map(fn, arg)
    comp = call

    transformed_comp, modified = tree_transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(transformed_comp.compact_representation(),
                     comp.compact_representation())
    self.assertEqual(transformed_comp.compact_representation(),
                     'federated_map(<(a -> a),data>)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), '{int32}@CLIENTS')
    self.assertFalse(modified)

  def test_does_not_merge_separated_federated_maps(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    call_1 = building_block_factory.create_federated_map(fn, arg)
    block = compiler_test_utils.create_whimsy_block(call_1, variable_name='b')
    call_2 = building_block_factory.create_federated_map(fn, block)
    comp = call_2

    transformed_comp, modified = tree_transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(transformed_comp.compact_representation(),
                     comp.compact_representation())
    self.assertEqual(
        transformed_comp.compact_representation(),
        'federated_map(<(a -> a),(let b=data in federated_map(<(a -> a),data>))>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), '{int32}@CLIENTS')
    self.assertFalse(modified)


class MergeTupleIntrinsicsTest(test_case.TestCase):

  def test_raises_type_error_with_none_comp(self):
    with self.assertRaises(TypeError):
      tree_transformations.merge_tuple_intrinsics(
          None, intrinsic_defs.FEDERATED_MAP.uri)

  def test_raises_type_error_with_none_uri(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_federated_map(
        parameter_name='a')
    calls = building_blocks.Struct((called_intrinsic, called_intrinsic))
    comp = calls
    with self.assertRaises(TypeError):
      tree_transformations.merge_tuple_intrinsics(comp, None)

  def test_raises_value_error(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_federated_map(
        parameter_name='a')
    calls = building_blocks.Struct((called_intrinsic, called_intrinsic))
    comp = calls
    with self.assertRaises(ValueError):
      tree_transformations.merge_tuple_intrinsics(comp, 'whimsy')

  def test_merges_federated_aggregates(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    calls = building_blocks.Struct((called_intrinsic, called_intrinsic))
    comp = calls

    transformed_comp, modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_AGGREGATE.uri)

    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        '<\n'
        '  federated_aggregate(<\n'
        '    data,\n'
        '    data,\n'
        '    (a -> data),\n'
        '    (b -> data),\n'
        '    (c -> data)\n'
        '  >),\n'
        '  federated_aggregate(<\n'
        '    data,\n'
        '    data,\n'
        '    (a -> data),\n'
        '    (b -> data),\n'
        '    (c -> data)\n'
        '  >)\n'
        '>'
    )
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(x -> <\n'
        '  x[0],\n'
        '  x[1]\n'
        '>)((let\n'
        '  value=federated_aggregate(<\n'
        '    federated_zip_at_clients(<\n'
        '      data,\n'
        '      data\n'
        '    >),\n'
        '    <\n'
        '      data,\n'
        '      data\n'
        '    >,\n'
        '    (let\n'
        '      _var1=<\n'
        '        (a -> data),\n'
        '        (a -> data)\n'
        '      >\n'
        '     in (_var2 -> <\n'
        '      _var1[0](<\n'
        '        <\n'
        '          _var2[0][0],\n'
        '          _var2[1][0]\n'
        '        >,\n'
        '        <\n'
        '          _var2[0][1],\n'
        '          _var2[1][1]\n'
        '        >\n'
        '      >[0]),\n'
        '      _var1[1](<\n'
        '        <\n'
        '          _var2[0][0],\n'
        '          _var2[1][0]\n'
        '        >,\n'
        '        <\n'
        '          _var2[0][1],\n'
        '          _var2[1][1]\n'
        '        >\n'
        '      >[1])\n'
        '    >)),\n'
        '    (let\n'
        '      _var3=<\n'
        '        (b -> data),\n'
        '        (b -> data)\n'
        '      >\n'
        '     in (_var4 -> <\n'
        '      _var3[0](<\n'
        '        <\n'
        '          _var4[0][0],\n'
        '          _var4[1][0]\n'
        '        >,\n'
        '        <\n'
        '          _var4[0][1],\n'
        '          _var4[1][1]\n'
        '        >\n'
        '      >[0]),\n'
        '      _var3[1](<\n'
        '        <\n'
        '          _var4[0][0],\n'
        '          _var4[1][0]\n'
        '        >,\n'
        '        <\n'
        '          _var4[0][1],\n'
        '          _var4[1][1]\n'
        '        >\n'
        '      >[1])\n'
        '    >)),\n'
        '    (let\n'
        '      _var5=<\n'
        '        (c -> data),\n'
        '        (c -> data)\n'
        '      >\n'
        '     in (_var6 -> <\n'
        '      _var5[0](_var6[0]),\n'
        '      _var5[1](_var6[1])\n'
        '    >))\n'
        '  >)\n'
        ' in <\n'
        '  federated_apply(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >),\n'
        '  federated_apply(<\n'
        '    (arg -> arg[1]),\n'
        '    value\n'
        '  >)\n'
        '>))'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature), '<int32@SERVER,int32@SERVER>')
    self.assertTrue(modified)

  def test_merges_federated_aggregates_with_unknown_parameter_dim(self):
    value_type = tf.int32
    federated_value_type = computation_types.FederatedType(
        value_type, placements.CLIENTS)
    value = building_blocks.Data('data', federated_value_type)

    # Concrete zero has fixed dimension, but the federated aggregate will
    # declare a parameter with unknown dimension.
    zero = building_blocks.Data(
        'data', computation_types.TensorType(tf.float32, shape=[0]))
    zero_type = computation_types.TensorType(tf.float32, shape=[None])
    accumulate_type = computation_types.StructType((zero_type, value_type))
    accumulate_result = building_blocks.Data('data', zero_type)
    accumulate = building_blocks.Lambda('a', accumulate_type, accumulate_result)
    merge_type = computation_types.StructType((zero_type, zero_type))
    merge_result = building_blocks.Data('data', zero_type)
    merge = building_blocks.Lambda('b', merge_type, merge_result)
    report_result = building_blocks.Data('data', tf.bool)
    report = building_blocks.Lambda('c', zero_type, report_result)

    called_intrinsic = building_block_factory.create_federated_aggregate(
        value, zero, accumulate, merge, report)
    calls = building_blocks.Struct((called_intrinsic, called_intrinsic))
    comp = calls

    transformed_comp, modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_AGGREGATE.uri)
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_merges_multiple_federated_aggregates(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    calls = building_blocks.Struct(
        (called_intrinsic, called_intrinsic, called_intrinsic))
    comp = calls

    transformed_comp, modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_AGGREGATE.uri)

    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        '<\n'
        '  federated_aggregate(<\n'
        '    data,\n'
        '    data,\n'
        '    (a -> data),\n'
        '    (b -> data),\n'
        '    (c -> data)\n'
        '  >),\n'
        '  federated_aggregate(<\n'
        '    data,\n'
        '    data,\n'
        '    (a -> data),\n'
        '    (b -> data),\n'
        '    (c -> data)\n'
        '  >),\n'
        '  federated_aggregate(<\n'
        '    data,\n'
        '    data,\n'
        '    (a -> data),\n'
        '    (b -> data),\n'
        '    (c -> data)\n'
        '  >)\n'
        '>'
    )
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(x -> <\n'
        '  x[0],\n'
        '  x[1],\n'
        '  x[2]\n'
        '>)((let\n'
        '  value=federated_aggregate(<\n'
        '    federated_zip_at_clients(<\n'
        '      data,\n'
        '      data,\n'
        '      data\n'
        '    >),\n'
        '    <\n'
        '      data,\n'
        '      data,\n'
        '      data\n'
        '    >,\n'
        '    (let\n'
        '      _var1=<\n'
        '        (a -> data),\n'
        '        (a -> data),\n'
        '        (a -> data)\n'
        '      >\n'
        '     in (_var2 -> <\n'
        '      _var1[0](<\n'
        '        <\n'
        '          _var2[0][0],\n'
        '          _var2[1][0]\n'
        '        >,\n'
        '        <\n'
        '          _var2[0][1],\n'
        '          _var2[1][1]\n'
        '        >,\n'
        '        <\n'
        '          _var2[0][2],\n'
        '          _var2[1][2]\n'
        '        >\n'
        '      >[0]),\n'
        '      _var1[1](<\n'
        '        <\n'
        '          _var2[0][0],\n'
        '          _var2[1][0]\n'
        '        >,\n'
        '        <\n'
        '          _var2[0][1],\n'
        '          _var2[1][1]\n'
        '        >,\n'
        '        <\n'
        '          _var2[0][2],\n'
        '          _var2[1][2]\n'
        '        >\n'
        '      >[1]),\n'
        '      _var1[2](<\n'
        '        <\n'
        '          _var2[0][0],\n'
        '          _var2[1][0]\n'
        '        >,\n'
        '        <\n'
        '          _var2[0][1],\n'
        '          _var2[1][1]\n'
        '        >,\n'
        '        <\n'
        '          _var2[0][2],\n'
        '          _var2[1][2]\n'
        '        >\n'
        '      >[2])\n'
        '    >)),\n'
        '    (let\n'
        '      _var3=<\n'
        '        (b -> data),\n'
        '        (b -> data),\n'
        '        (b -> data)\n'
        '      >\n'
        '     in (_var4 -> <\n'
        '      _var3[0](<\n'
        '        <\n'
        '          _var4[0][0],\n'
        '          _var4[1][0]\n'
        '        >,\n'
        '        <\n'
        '          _var4[0][1],\n'
        '          _var4[1][1]\n'
        '        >,\n'
        '        <\n'
        '          _var4[0][2],\n'
        '          _var4[1][2]\n'
        '        >\n'
        '      >[0]),\n'
        '      _var3[1](<\n'
        '        <\n'
        '          _var4[0][0],\n'
        '          _var4[1][0]\n'
        '        >,\n'
        '        <\n'
        '          _var4[0][1],\n'
        '          _var4[1][1]\n'
        '        >,\n'
        '        <\n'
        '          _var4[0][2],\n'
        '          _var4[1][2]\n'
        '        >\n'
        '      >[1]),\n'
        '      _var3[2](<\n'
        '        <\n'
        '          _var4[0][0],\n'
        '          _var4[1][0]\n'
        '        >,\n'
        '        <\n'
        '          _var4[0][1],\n'
        '          _var4[1][1]\n'
        '        >,\n'
        '        <\n'
        '          _var4[0][2],\n'
        '          _var4[1][2]\n'
        '        >\n'
        '      >[2])\n'
        '    >)),\n'
        '    (let\n'
        '      _var5=<\n'
        '        (c -> data),\n'
        '        (c -> data),\n'
        '        (c -> data)\n'
        '      >\n'
        '     in (_var6 -> <\n'
        '      _var5[0](_var6[0]),\n'
        '      _var5[1](_var6[1]),\n'
        '      _var5[2](_var6[2])\n'
        '    >))\n'
        '  >)\n'
        ' in <\n'
        '  federated_apply(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >),\n'
        '  federated_apply(<\n'
        '    (arg -> arg[1]),\n'
        '    value\n'
        '  >),\n'
        '  federated_apply(<\n'
        '    (arg -> arg[2]),\n'
        '    value\n'
        '  >)\n'
        '>))'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature),
        '<int32@SERVER,int32@SERVER,int32@SERVER>')
    self.assertTrue(modified)

  def test_merges_federated_applys(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_federated_apply(
        parameter_name='a')
    calls = building_blocks.Struct((called_intrinsic, called_intrinsic))
    comp = calls

    transformed_comp, modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_APPLY.uri)

    self.assertEqual(
        comp.compact_representation(),
        '<federated_apply(<(a -> a),data>),federated_apply(<(a -> a),data>)>')
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(x -> <\n'
        '  x[0],\n'
        '  x[1]\n'
        '>)((let\n'
        '  value=federated_apply(<\n'
        '    (let\n'
        '      _var1=<\n'
        '        (a -> a),\n'
        '        (a -> a)\n'
        '      >\n'
        '     in (_var2 -> <\n'
        '      _var1[0](_var2[0]),\n'
        '      _var1[1](_var2[1])\n'
        '    >)),\n'
        '    federated_zip_at_server(<\n'
        '      data,\n'
        '      data\n'
        '    >)\n'
        '  >)\n'
        ' in <\n'
        '  federated_apply(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >),\n'
        '  federated_apply(<\n'
        '    (arg -> arg[1]),\n'
        '    value\n'
        '  >)\n'
        '>))'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature), '<int32@SERVER,int32@SERVER>')
    self.assertTrue(modified)

  def test_merges_federated_broadcasts(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_federated_broadcast(
    )
    calls = building_blocks.Struct((called_intrinsic, called_intrinsic))
    comp = calls

    transformed_comp, modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_BROADCAST.uri)

    self.assertEqual(comp.compact_representation(),
                     '<federated_broadcast(data),federated_broadcast(data)>')
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(x -> <\n'
        '  x[0],\n'
        '  x[1]\n'
        '>)((let\n'
        '  value=federated_broadcast(federated_zip_at_server(<\n'
        '    data,\n'
        '    data\n'
        '  >))\n'
        ' in <\n'
        '  federated_map_all_equal(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >),\n'
        '  federated_map_all_equal(<\n'
        '    (arg -> arg[1]),\n'
        '    value\n'
        '  >)\n'
        '>))'
    )
    # pyformat: enable
    self.assertEqual(str(comp.type_signature), '<int32@CLIENTS,int32@CLIENTS>')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature), '<int32@CLIENTS,int32@CLIENTS>')
    self.assertTrue(modified)

  def test_merges_federated_maps(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_federated_map(
        parameter_name='a')
    calls = building_blocks.Struct((called_intrinsic, called_intrinsic))
    comp = calls

    transformed_comp, modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_MAP.uri)

    self.assertEqual(
        comp.compact_representation(),
        '<federated_map(<(a -> a),data>),federated_map(<(a -> a),data>)>')
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(x -> <\n'
        '  x[0],\n'
        '  x[1]\n'
        '>)((let\n'
        '  value=federated_map(<\n'
        '    (let\n'
        '      _var1=<\n'
        '        (a -> a),\n'
        '        (a -> a)\n'
        '      >\n'
        '     in (_var2 -> <\n'
        '      _var1[0](_var2[0]),\n'
        '      _var1[1](_var2[1])\n'
        '    >)),\n'
        '    federated_zip_at_clients(<\n'
        '      data,\n'
        '      data\n'
        '    >)\n'
        '  >)\n'
        ' in <\n'
        '  federated_map(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> arg[1]),\n'
        '    value\n'
        '  >)\n'
        '>))'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature),
        '<{int32}@CLIENTS,{int32}@CLIENTS>')
    self.assertTrue(modified)

  def test_merges_federated_maps_with_different_names(self):
    called_intrinsic_1 = compiler_test_utils.create_whimsy_called_federated_map(
        parameter_name='a')
    called_intrinsic_2 = compiler_test_utils.create_whimsy_called_federated_map(
        parameter_name='b')
    calls = building_blocks.Struct((called_intrinsic_1, called_intrinsic_2))
    comp = calls

    transformed_comp, modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_MAP.uri)

    self.assertEqual(
        comp.compact_representation(),
        '<federated_map(<(a -> a),data>),federated_map(<(b -> b),data>)>')
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(x -> <\n'
        '  x[0],\n'
        '  x[1]\n'
        '>)((let\n'
        '  value=federated_map(<\n'
        '    (let\n'
        '      _var1=<\n'
        '        (a -> a),\n'
        '        (b -> b)\n'
        '      >\n'
        '     in (_var2 -> <\n'
        '      _var1[0](_var2[0]),\n'
        '      _var1[1](_var2[1])\n'
        '    >)),\n'
        '    federated_zip_at_clients(<\n'
        '      data,\n'
        '      data\n'
        '    >)\n'
        '  >)\n'
        ' in <\n'
        '  federated_map(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> arg[1]),\n'
        '    value\n'
        '  >)\n'
        '>))'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature),
        '<{int32}@CLIENTS,{int32}@CLIENTS>')
    self.assertTrue(modified)

  def test_merges_federated_maps_with_different_type(self):
    called_intrinsic_1 = compiler_test_utils.create_whimsy_called_federated_map(
        parameter_name='a', parameter_type=tf.int32)
    called_intrinsic_2 = compiler_test_utils.create_whimsy_called_federated_map(
        parameter_name='b', parameter_type=tf.float32)
    calls = building_blocks.Struct((called_intrinsic_1, called_intrinsic_2))
    comp = calls

    transformed_comp, modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_MAP.uri)

    self.assertEqual(
        comp.compact_representation(),
        '<federated_map(<(a -> a),data>),federated_map(<(b -> b),data>)>')
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(x -> <\n'
        '  x[0],\n'
        '  x[1]\n'
        '>)((let\n'
        '  value=federated_map(<\n'
        '    (let\n'
        '      _var1=<\n'
        '        (a -> a),\n'
        '        (b -> b)\n'
        '      >\n'
        '     in (_var2 -> <\n'
        '      _var1[0](_var2[0]),\n'
        '      _var1[1](_var2[1])\n'
        '    >)),\n'
        '    federated_zip_at_clients(<\n'
        '      data,\n'
        '      data\n'
        '    >)\n'
        '  >)\n'
        ' in <\n'
        '  federated_map(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> arg[1]),\n'
        '    value\n'
        '  >)\n'
        '>))'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature),
        '<{int32}@CLIENTS,{float32}@CLIENTS>')
    self.assertTrue(modified)

  def test_merges_federated_maps_with_named_parameter_type(self):
    parameter_type = [('b', tf.int32), ('c', tf.float32)]
    called_intrinsic = compiler_test_utils.create_whimsy_called_federated_map(
        parameter_name='a', parameter_type=parameter_type)
    calls = building_blocks.Struct((called_intrinsic, called_intrinsic))
    comp = calls
    transformed_comp, modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_MAP.uri)

    self.assertEqual(
        comp.compact_representation(),
        '<federated_map(<(a -> a),data>),federated_map(<(a -> a),data>)>')
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(x -> <\n'
        '  x[0],\n'
        '  x[1]\n'
        '>)((let\n'
        '  value=federated_map(<\n'
        '    (let\n'
        '      _var1=<\n'
        '        (a -> a),\n'
        '        (a -> a)\n'
        '      >\n'
        '     in (_var2 -> <\n'
        '      _var1[0](_var2[0]),\n'
        '      _var1[1](_var2[1])\n'
        '    >)),\n'
        '    federated_zip_at_clients(<\n'
        '      data,\n'
        '      data\n'
        '    >)\n'
        '  >)\n'
        ' in <\n'
        '  federated_map(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> arg[1]),\n'
        '    value\n'
        '  >)\n'
        '>))'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature),
        '<{<b=int32,c=float32>}@CLIENTS,{<b=int32,c=float32>}@CLIENTS>')
    self.assertTrue(modified)

  def test_merges_federated_maps_with_different_named_parameter_types(self):
    parameter_type_1 = [('b', tf.int32), ('c', tf.float32)]
    called_intrinsic_1 = compiler_test_utils.create_whimsy_called_federated_map(
        parameter_name='a', parameter_type=parameter_type_1)
    parameter_type_2 = [('e', tf.bool), ('f', tf.string)]
    called_intrinsic_2 = compiler_test_utils.create_whimsy_called_federated_map(
        parameter_name='d', parameter_type=parameter_type_2)
    calls = building_blocks.Struct((called_intrinsic_1, called_intrinsic_2))
    comp = calls
    transformed_comp, modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_MAP.uri)

    self.assertEqual(
        comp.compact_representation(),
        '<federated_map(<(a -> a),data>),federated_map(<(d -> d),data>)>')
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(x -> <\n'
        '  x[0],\n'
        '  x[1]\n'
        '>)((let\n'
        '  value=federated_map(<\n'
        '    (let\n'
        '      _var1=<\n'
        '        (a -> a),\n'
        '        (d -> d)\n'
        '      >\n'
        '     in (_var2 -> <\n'
        '      _var1[0](_var2[0]),\n'
        '      _var1[1](_var2[1])\n'
        '    >)),\n'
        '    federated_zip_at_clients(<\n'
        '      data,\n'
        '      data\n'
        '    >)\n'
        '  >)\n'
        ' in <\n'
        '  federated_map(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> arg[1]),\n'
        '    value\n'
        '  >)\n'
        '>))'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature),
        '<{<b=int32,c=float32>}@CLIENTS,{<e=bool,f=string>}@CLIENTS>')
    self.assertTrue(modified)

  def test_merges_federated_maps_with_unbound_reference(self):
    ref = building_blocks.Reference('a', tf.int32)
    fn = building_blocks.Lambda('b', tf.int32, ref)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    called_intrinsic = building_block_factory.create_federated_map(fn, arg)
    calls = building_blocks.Struct((called_intrinsic, called_intrinsic))
    comp = calls

    transformed_comp, modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_MAP.uri)

    self.assertEqual(
        comp.compact_representation(),
        '<federated_map(<(b -> a),data>),federated_map(<(b -> a),data>)>')
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(x -> <\n'
        '  x[0],\n'
        '  x[1]\n'
        '>)((let\n'
        '  value=federated_map(<\n'
        '    (let\n'
        '      _var1=<\n'
        '        (b -> a),\n'
        '        (b -> a)\n'
        '      >\n'
        '     in (_var2 -> <\n'
        '      _var1[0](_var2[0]),\n'
        '      _var1[1](_var2[1])\n'
        '    >)),\n'
        '    federated_zip_at_clients(<\n'
        '      data,\n'
        '      data\n'
        '    >)\n'
        '  >)\n'
        ' in <\n'
        '  federated_map(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> arg[1]),\n'
        '    value\n'
        '  >)\n'
        '>))'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature),
        '<{int32}@CLIENTS,{int32}@CLIENTS>')
    self.assertTrue(modified)

  def test_merges_named_federated_maps(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_federated_map(
        parameter_name='a')
    calls = building_blocks.Struct(
        (('b', called_intrinsic), ('c', called_intrinsic)))
    comp = calls

    transformed_comp, modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_MAP.uri)

    self.assertEqual(
        comp.compact_representation(),
        '<b=federated_map(<(a -> a),data>),c=federated_map(<(a -> a),data>)>')
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(x -> <\n'
        '  b=x[0],\n'
        '  c=x[1]\n'
        '>)((let\n'
        '  value=federated_map(<\n'
        '    (let\n'
        '      _var1=<\n'
        '        (a -> a),\n'
        '        (a -> a)\n'
        '      >\n'
        '     in (_var2 -> <\n'
        '      _var1[0](_var2[0]),\n'
        '      _var1[1](_var2[1])\n'
        '    >)),\n'
        '    federated_zip_at_clients(<\n'
        '      data,\n'
        '      data\n'
        '    >)\n'
        '  >)\n'
        ' in <\n'
        '  federated_map(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> arg[1]),\n'
        '    value\n'
        '  >)\n'
        '>))'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature),
        '<b={int32}@CLIENTS,c={int32}@CLIENTS>')
    self.assertTrue(modified)

  def test_merges_nested_federated_maps(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_federated_map(
        parameter_name='a')
    calls = building_blocks.Struct((called_intrinsic, called_intrinsic))
    block = compiler_test_utils.create_whimsy_block(calls, variable_name='a')
    comp = block

    transformed_comp, modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_MAP.uri)

    self.assertEqual(
        comp.compact_representation(),
        '(let a=data in <federated_map(<(a -> a),data>),federated_map(<(a -> a),data>)>)'
    )
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(let\n'
        '  a=data\n'
        ' in (x -> <\n'
        '  x[0],\n'
        '  x[1]\n'
        '>)((let\n'
        '  value=federated_map(<\n'
        '    (let\n'
        '      _var1=<\n'
        '        (a -> a),\n'
        '        (a -> a)\n'
        '      >\n'
        '     in (_var2 -> <\n'
        '      _var1[0](_var2[0]),\n'
        '      _var1[1](_var2[1])\n'
        '    >)),\n'
        '    federated_zip_at_clients(<\n'
        '      data,\n'
        '      data\n'
        '    >)\n'
        '  >)\n'
        ' in <\n'
        '  federated_map(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> arg[1]),\n'
        '    value\n'
        '  >)\n'
        '>)))'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature),
        '<{int32}@CLIENTS,{int32}@CLIENTS>')
    self.assertTrue(modified)

  def test_merges_multiple_federated_maps(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_federated_map(
        parameter_name='a')
    calls = building_blocks.Struct(
        (called_intrinsic, called_intrinsic, called_intrinsic))
    comp = calls

    transformed_comp, modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_MAP.uri)

    self.assertEqual(
        comp.compact_representation(),
        '<federated_map(<(a -> a),data>),federated_map(<(a -> a),data>),federated_map(<(a -> a),data>)>'
    )
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(x -> <\n'
        '  x[0],\n'
        '  x[1],\n'
        '  x[2]\n'
        '>)((let\n'
        '  value=federated_map(<\n'
        '    (let\n'
        '      _var1=<\n'
        '        (a -> a),\n'
        '        (a -> a),\n'
        '        (a -> a)\n'
        '      >\n'
        '     in (_var2 -> <\n'
        '      _var1[0](_var2[0]),\n'
        '      _var1[1](_var2[1]),\n'
        '      _var1[2](_var2[2])\n'
        '    >)),\n'
        '    federated_zip_at_clients(<\n'
        '      data,\n'
        '      data,\n'
        '      data\n'
        '    >)\n'
        '  >)\n'
        ' in <\n'
        '  federated_map(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> arg[1]),\n'
        '    value\n'
        '  >),\n'
        '  federated_map(<\n'
        '    (arg -> arg[2]),\n'
        '    value\n'
        '  >)\n'
        '>))'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature),
        '<{int32}@CLIENTS,{int32}@CLIENTS,{int32}@CLIENTS>')
    self.assertTrue(modified)

  def test_merges_one_federated_map(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_federated_map(
        parameter_name='a')
    calls = building_blocks.Struct((called_intrinsic,))
    comp = calls

    transformed_comp, modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_MAP.uri)

    self.assertEqual(comp.compact_representation(),
                     '<federated_map(<(a -> a),data>)>')
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(x -> <\n'
        '  x[0]\n'
        '>)((let\n'
        '  value=federated_map(<\n'
        '    (let\n'
        '      _var1=<\n'
        '        (a -> a)\n'
        '      >\n'
        '     in (_var2 -> <\n'
        '      _var1[0](_var2[0])\n'
        '    >)),\n'
        '    federated_zip_at_clients(<\n'
        '      data\n'
        '    >)\n'
        '  >)\n'
        ' in <\n'
        '  federated_map(<\n'
        '    (arg -> arg[0]),\n'
        '    value\n'
        '  >)\n'
        '>))'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), '<{int32}@CLIENTS>')
    self.assertTrue(modified)

  def test_does_not_merge_intrinsics_with_different_uris(self):
    called_intrinsic_1 = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    called_intrinsic_2 = compiler_test_utils.create_whimsy_called_federated_map(
        parameter_name='a')
    calls = building_blocks.Struct((called_intrinsic_1, called_intrinsic_2))
    comp = calls

    transformed_comp, modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_MAP.uri)

    self.assertEqual(transformed_comp.compact_representation(),
                     comp.compact_representation())
    self.assertEqual(
        transformed_comp.compact_representation(),
        '<federated_aggregate(<data,data,(a -> data),(b -> data),(c -> data)>),federated_map(<(a -> a),data>)>'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature), '<int32@SERVER,{int32}@CLIENTS>')
    self.assertFalse(modified)

  def test_does_not_merge_intrinsics_with_different_uri(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_federated_map(
        parameter_name='a')
    calls = building_blocks.Struct((called_intrinsic, called_intrinsic))
    comp = calls

    transformed_comp, modified = tree_transformations.merge_tuple_intrinsics(
        comp, intrinsic_defs.FEDERATED_AGGREGATE.uri)

    self.assertEqual(transformed_comp.compact_representation(),
                     comp.compact_representation())
    self.assertEqual(
        transformed_comp.compact_representation(),
        '<federated_map(<(a -> a),data>),federated_map(<(a -> a),data>)>')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature),
        '<{int32}@CLIENTS,{int32}@CLIENTS>')
    self.assertFalse(modified)


class RemoveDuplicateBlockLocals(test_case.TestCase):

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
    tup = building_blocks.Struct([
        called_intrinsic,
        called_intrinsic,
    ])
    comp = tup

    intermediate_comp, _ = tree_transformations.uniquify_reference_names(comp)
    intermediate_comp, _ = tree_transformations.extract_computations(
        intermediate_comp)
    transformed_comp, modified = tree_transformations.remove_duplicate_block_locals(
        intermediate_comp)
    transformed_comp, _ = tree_transformations.uniquify_reference_names(
        transformed_comp)

    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        '<\n'
        '  federated_aggregate(<\n'
        '    data,\n'
        '    data,\n'
        '    (a -> data),\n'
        '    (b -> data),\n'
        '    (c -> data)\n'
        '  >),\n'
        '  federated_aggregate(<\n'
        '    data,\n'
        '    data,\n'
        '    (a -> data),\n'
        '    (b -> data),\n'
        '    (c -> data)\n'
        '  >)\n'
        '>'
    )
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(let\n'
        '  _dep1=federated_aggregate,\n'
        '  _dep2=data,\n'
        '  _dep3=data,\n'
        '  _dep5=(_dep4 -> _dep3),\n'
        '  _dep7=(_dep6 -> _dep3),\n'
        '  _dep8=<\n'
        '    _dep2,\n'
        '    _dep3,\n'
        '    _dep5,\n'
        '    _dep5,\n'
        '    _dep7\n'
        '  >,\n'
        '  _dep9=_dep1(_dep8),\n'
        '  _dep10=<\n'
        '    _dep9,\n'
        '    _dep9\n'
        '  >\n'
        ' in _dep10)'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_removes_federated_broadcast(self):
    called_intrinsic = compiler_test_utils.create_whimsy_called_federated_broadcast(
    )
    tup = building_blocks.Struct([
        called_intrinsic,
        called_intrinsic,
    ])
    comp = tup

    intermediate_comp, _ = tree_transformations.extract_computations(comp)
    transformed_comp, modified = tree_transformations.remove_duplicate_block_locals(
        intermediate_comp)

    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        '<\n'
        '  federated_broadcast(data),\n'
        '  federated_broadcast(data)\n'
        '>'
    )
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(let\n'
        '  _var1=federated_broadcast,\n'
        '  _var2=data,\n'
        '  _var3=_var1(_var2),\n'
        '  _var9=<\n'
        '    _var3,\n'
        '    _var3\n'
        '  >\n'
        ' in _var9)'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_removes_complex_comp(self):
    complex_comp = _create_complex_computation()
    comp = complex_comp

    intermediate_comp, _ = tree_transformations.extract_computations(comp)
    transformed_comp, modified = tree_transformations.remove_duplicate_block_locals(
        intermediate_comp)

    # pyformat: disable
    self.assertEqual(
        comp.formatted_representation(),
        '(b -> <\n'
        '  federated_mean(federated_map(<\n'
        '    comp#a,\n'
        '    federated_broadcast(b)\n'
        '  >)),\n'
        '  federated_mean(federated_map(<\n'
        '    comp#a,\n'
        '    federated_broadcast(b)\n'
        '  >))\n'
        '>)'
    )
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(b -> (let\n'
        '  _var9=federated_mean,\n'
        '  _var6=federated_map,\n'
        '  _var3=comp#a,\n'
        '  _var1=federated_broadcast,\n'
        '  _var2=_var1(b),\n'
        '  _var5=<\n'
        '    _var3,\n'
        '    _var2\n'
        '  >,\n'
        '  _var8=_var6(_var5),\n'
        '  _var11=_var9(_var8),\n'
        '  _var25=<\n'
        '    _var11,\n'
        '    _var11\n'
        '  >\n'
        ' in _var25))'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_removes_chained_references_bound_by_lambda(self):
    ref_1 = building_blocks.Reference('a', tf.int32)
    ref_2 = building_blocks.Reference('b', tf.int32)
    tup = building_blocks.Struct([ref_2])
    block = building_blocks.Block([(ref_2.name, ref_1)], tup)
    fn = building_blocks.Lambda(ref_1.name, ref_1.type_signature, block)
    comp = fn

    intermediate_comp, _ = tree_transformations.extract_computations(comp)
    transformed_comp, modified = tree_transformations.remove_duplicate_block_locals(
        intermediate_comp)

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
        transformed_comp.formatted_representation(),
        '(a -> (let\n'
        '  _var1=<\n'
        '    a\n'
        '  >\n'
        ' in _var1))'
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

    intermediate_comp, _ = tree_transformations.extract_computations(comp)
    transformed_comp, modified = tree_transformations.remove_duplicate_block_locals(
        intermediate_comp)

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
        transformed_comp.formatted_representation(),
        '(let\n'
        '  _var1=<\n'
        '    a\n'
        '  >\n'
        ' in _var1)'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_strips_reference_chain_culminating_in_unbound_ref(self):
    ref_1 = building_blocks.Reference('a', tf.int32)
    ref_2 = building_blocks.Reference('b', tf.int32)
    ref_3 = building_blocks.Reference('c', tf.int32)
    tup = building_blocks.Struct([ref_3])
    block = building_blocks.Block([(ref_2.name, ref_1), (ref_3.name, ref_2)],
                                  tup)
    comp = block

    intermediate_comp, _ = tree_transformations.extract_computations(comp)
    transformed_comp, modified = tree_transformations.remove_duplicate_block_locals(
        intermediate_comp)

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
        transformed_comp.formatted_representation(),
        '(let\n'
        '  _var1=<\n'
        '    a\n'
        '  >\n'
        ' in _var1)'
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


class UniquifyCompiledComputationNamesTest(parameterized.TestCase):

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      tree_transformations.uniquify_compiled_computation_names(None)

  def test_replaces_name(self):
    tensor_type = computation_types.TensorType(tf.int32)
    compiled = building_block_factory.create_compiled_identity(tensor_type)
    comp = compiled

    transformed_comp, modified = tree_transformations.uniquify_compiled_computation_names(
        comp)

    self.assertNotEqual(transformed_comp._name, comp._name)
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replaces_multiple_names(self):
    elements = []
    for _ in range(10):
      tensor_type = computation_types.TensorType(tf.int32)
      compiled = building_block_factory.create_compiled_identity(tensor_type)
      elements.append(compiled)
    compiled_comps = building_blocks.Struct(elements)
    comp = compiled_comps

    transformed_comp, modified = tree_transformations.uniquify_compiled_computation_names(
        comp)

    comp_names = [element._name for element in comp]
    transformed_comp_names = [element._name for element in transformed_comp]
    self.assertNotEqual(transformed_comp_names, comp_names)
    self.assertEqual(
        len(transformed_comp_names), len(set(transformed_comp_names)),
        'The transformed computation names are not unique: {}.'.format(
            transformed_comp_names))
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_does_not_replace_other_name(self):
    comp = building_blocks.Reference('name', tf.int32)

    transformed_comp, modified = tree_transformations.uniquify_compiled_computation_names(
        comp)

    self.assertEqual(transformed_comp._name, comp._name)
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)


class ResolveHigherOrderFunctionsTest(test_case.TestCase):

  def _apply_resolution_and_remove_unused_block_locals(self, comp):
    fns_resolved, modified = tree_transformations.resolve_higher_order_functions(
        comp)
    locals_removed, locals_modified = tree_transformations.remove_unused_block_locals(
        fns_resolved)
    return locals_removed, modified or locals_modified

  def test_raises_with_nonunique_names(self):
    whimsy_int = building_blocks.Data('data', tf.int32)
    blk_with_renamed = building_blocks.Block([('x', whimsy_int),
                                              ('x', whimsy_int)], whimsy_int)
    with self.assertRaises(ValueError):
      tree_transformations.resolve_higher_order_functions(blk_with_renamed)

  def test_raises_with_unsafe_rebinding(self):
    whimsy_int = building_blocks.Data('data', tf.int32)
    ref_to_x = building_blocks.Reference('x', tf.int32)
    blk_with_renamed = building_blocks.Block([('y', ref_to_x),
                                              ('x', whimsy_int)], whimsy_int)
    with self.assertRaises(ValueError):
      tree_transformations.resolve_higher_order_functions(blk_with_renamed)

  def test_resolves_selection_from_call(self):
    int_identity = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    whimsy_int = building_blocks.Data('data', tf.int32)
    tup_containing_fn = building_blocks.Struct([int_identity, whimsy_int])
    lambda_returning_tup = building_blocks.Lambda('z', tf.int32,
                                                  tup_containing_fn)
    called_tup = building_blocks.Call(lambda_returning_tup, whimsy_int)
    selected_identity = building_blocks.Selection(source=called_tup, index=0)
    called_id = building_blocks.Call(selected_identity, whimsy_int)

    resolved, transformed = self._apply_resolution_and_remove_unused_block_locals(
        called_id)

    self.assertTrue(transformed)
    self.assertEqual(resolved.compact_representation(), '(x -> x)(data)')

  def test_resolves_selection_from_call_with_no_param_lambda(self):
    materialize_int = building_blocks.Lambda(
        None, None, building_blocks.Data('x', tf.int32))
    whimsy_int = building_blocks.Data('data', tf.int32)
    tup_containing_fn = building_blocks.Struct([materialize_int, whimsy_int])
    lambda_returning_tup = building_blocks.Lambda('z', tf.int32,
                                                  tup_containing_fn)
    called_tup = building_blocks.Call(lambda_returning_tup, whimsy_int)
    selected_materialize_int = building_blocks.Selection(
        source=called_tup, index=0)
    called_id = building_blocks.Call(selected_materialize_int)

    resolved, transformed = self._apply_resolution_and_remove_unused_block_locals(
        called_id)

    self.assertTrue(transformed)
    self.assertEqual(resolved.compact_representation(), '( -> x)()')

  def test_resolves_selection_from_symbol_bound_to_call(self):
    int_identity = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    whimsy_int = building_blocks.Data('data', tf.int32)
    tup_containing_fn = building_blocks.Struct([int_identity, whimsy_int])
    lambda_returning_tup = building_blocks.Lambda('z', tf.int32,
                                                  tup_containing_fn)
    called_tup = building_blocks.Call(lambda_returning_tup, whimsy_int)
    ref_to_y = building_blocks.Reference('y', called_tup.type_signature)
    selected_identity = building_blocks.Selection(source=ref_to_y, index=0)
    called_id = building_blocks.Call(selected_identity, whimsy_int)
    blk = building_blocks.Block([('y', called_tup)], called_id)

    resolved, transformed = self._apply_resolution_and_remove_unused_block_locals(
        blk)

    self.assertTrue(transformed)
    self.assertEqual(resolved.compact_representation(), '(x -> x)(data)')

  def test_resolves_selection_from_tuple(self):
    int_identity = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    whimsy_int = building_blocks.Data('data', tf.int32)
    tup_containing_fn = building_blocks.Struct([int_identity, whimsy_int])
    selected_identity = building_blocks.Selection(
        source=tup_containing_fn, index=0)
    called_id = building_blocks.Call(selected_identity, whimsy_int)

    resolved, transformed = self._apply_resolution_and_remove_unused_block_locals(
        called_id)

    self.assertTrue(transformed)
    self.assertEqual(resolved.compact_representation(), '(x -> x)(data)')

  def test_resolves_selection_from_nested_tuple(self):
    int_identity = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    whimsy_int = building_blocks.Data('data', tf.int32)
    tup_containing_fn = building_blocks.Struct([int_identity, whimsy_int])
    nested_tuple = building_blocks.Struct([tup_containing_fn])
    selected_tuple = building_blocks.Selection(source=nested_tuple, index=0)
    selected_identity = building_blocks.Selection(
        source=selected_tuple, index=0)
    called_id = building_blocks.Call(selected_identity, whimsy_int)

    resolved, transformed = self._apply_resolution_and_remove_unused_block_locals(
        called_id)

    self.assertTrue(transformed)
    self.assertEqual(resolved.compact_representation(), '(x -> x)(data)')

  def test_resolves_selection_from_symbol_bound_to_tuple(self):
    int_identity = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    whimsy_int = building_blocks.Data('data', tf.int32)
    tup_containing_fn = building_blocks.Struct([int_identity, whimsy_int])
    ref_to_y = building_blocks.Reference('y', tup_containing_fn.type_signature)
    selected_identity = building_blocks.Selection(source=ref_to_y, index=0)
    called_id = building_blocks.Call(selected_identity, whimsy_int)
    blk = building_blocks.Block([('y', tup_containing_fn)], called_id)

    resolved, transformed = self._apply_resolution_and_remove_unused_block_locals(
        blk)

    self.assertTrue(transformed)
    self.assertEqual(resolved.compact_representation(), '(x -> x)(data)')

  def test_resolves_selection_from_nested_symbol_bound_to_nested_tuple(self):
    int_identity = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    whimsy_int = building_blocks.Data('data', tf.int32)
    tup_containing_fn = building_blocks.Struct([int_identity, whimsy_int])
    nested_tuple = building_blocks.Struct([tup_containing_fn])
    ref_to_nested_tuple = building_blocks.Reference('nested_tuple',
                                                    nested_tuple.type_signature)
    selected_tuple = building_blocks.Selection(
        source=ref_to_nested_tuple, index=0)
    selected_identity = building_blocks.Selection(
        source=selected_tuple, index=0)
    called_id = building_blocks.Call(selected_identity, whimsy_int)
    blk = building_blocks.Block([('nested_tuple', nested_tuple)], called_id)

    resolved, transformed = self._apply_resolution_and_remove_unused_block_locals(
        blk)

    self.assertTrue(transformed)
    self.assertEqual(resolved.compact_representation(), '(x -> x)(data)')

  def test_leaves_called_tf_unchanged(self):
    called_tf = building_block_factory.create_tensorflow_constant(
        computation_types.TensorType(tf.int32), 0)

    resolved, modified = self._apply_resolution_and_remove_unused_block_locals(
        called_tf)

    self.assertFalse(modified)
    self.assertRegexMatch(resolved.compact_representation(),
                          [r'comp#[a-zA-Z0-9]*\(\)'])

  def test_leaves_directly_called_lambda_unchanged(self):
    ref_to_z = building_blocks.Reference('z', tf.int32)
    int_identity = building_blocks.Lambda('z', tf.int32, ref_to_z)
    data = building_blocks.Data('data', tf.int32)
    called_identity = building_blocks.Call(int_identity, data)
    resolved, modified = self._apply_resolution_and_remove_unused_block_locals(
        called_identity)

    self.assertFalse(modified)
    self.assertEqual(resolved.compact_representation(), '(z -> z)(data)')

  def test_leaves_call_to_lambda_parameter_unchanged(self):
    ref_to_fn = building_blocks.Reference(
        'fn', computation_types.FunctionType(tf.int32, tf.int32))
    data = building_blocks.Data('data', tf.int32)
    called_fn = building_blocks.Call(ref_to_fn, data)
    lam = building_blocks.Lambda(ref_to_fn.name, ref_to_fn.type_signature,
                                 called_fn)

    resolved, transformed = self._apply_resolution_and_remove_unused_block_locals(
        lam)

    self.assertFalse(transformed)
    self.assertEqual(resolved.compact_representation(), '(fn -> fn(data))')

  def test_resolves_nested_calls(self):
    data = building_blocks.Data('data', tf.int32)
    ref_to_z = building_blocks.Reference('z', tf.int32)
    int_identity = building_blocks.Lambda('z', tf.int32, ref_to_z)
    middle_lambda = building_blocks.Lambda('x', tf.int32, int_identity)
    called_middle_lam = building_blocks.Call(middle_lambda, data)
    top_level_call = building_blocks.Call(called_middle_lam, data)

    resolved, modified = self._apply_resolution_and_remove_unused_block_locals(
        top_level_call)

    self.assertTrue(modified)
    self.assertEqual(resolved.compact_representation(), '(z -> z)(data)')

  def test_resolves_call_to_block_with_functional_result(self):
    data = building_blocks.Data('data', tf.int32)
    ref_to_z = building_blocks.Reference('z', tf.int32)
    ref_to_x = building_blocks.Reference('x', tf.int32)
    lowest_lam = building_blocks.Lambda(
        'z', tf.int32, building_blocks.Struct([ref_to_z, ref_to_x]))
    blk = building_blocks.Block([('x', data)], lowest_lam)
    called_blk = building_blocks.Call(blk, data)

    resolved, modified = self._apply_resolution_and_remove_unused_block_locals(
        called_blk)

    self.assertTrue(modified)
    self.assertEqual(resolved.compact_representation(),
                     '(let x=data in (z -> <z,x>)(data))')

  def test_resolves_call_to_block_under_tuple_selection(self):
    data = building_blocks.Data('data', tf.int32)
    ref_to_z = building_blocks.Reference('z', tf.int32)
    ref_to_x = building_blocks.Reference('x', tf.int32)
    lowest_lam = building_blocks.Lambda(
        'z', tf.int32, building_blocks.Struct([ref_to_z, ref_to_x]))
    blk = building_blocks.Block([('x', data)], lowest_lam)
    tuple_holding_blk = building_blocks.Struct([blk])
    zeroth_selection_from_tuple = building_blocks.Selection(
        source=tuple_holding_blk, index=0)
    called_sel = building_blocks.Call(zeroth_selection_from_tuple, data)

    resolved, modified = self._apply_resolution_and_remove_unused_block_locals(
        called_sel)

    self.assertTrue(modified)
    self.assertEqual(resolved.compact_representation(),
                     '(let x=data in (z -> <z,x>)(data))')

  def test_resolves_call_to_block_bound_to_symbol(self):
    data = building_blocks.Data('data', tf.int32)
    ref_to_z = building_blocks.Reference('z', tf.int32)
    ref_to_x = building_blocks.Reference('x', tf.int32)
    lowest_lam = building_blocks.Lambda(
        'z', tf.int32, building_blocks.Struct([ref_to_z, ref_to_x]))
    blk = building_blocks.Block([('x', data)], lowest_lam)
    ref_to_y = building_blocks.Reference('y', blk.type_signature)

    called_ref = building_blocks.Call(ref_to_y, data)
    higher_blk = building_blocks.Block([('y', blk)], called_ref)

    resolved, modified = self._apply_resolution_and_remove_unused_block_locals(
        higher_blk)

    self.assertTrue(modified)
    self.assertEqual(resolved.compact_representation(),
                     '(let x=data in (z -> <z,x>)(data))')

  def test_resolves_call_to_functional_reference(self):
    data = building_blocks.Data('data', tf.int32)
    ref_to_z = building_blocks.Reference('z', tf.int32)
    int_identity = building_blocks.Lambda('z', tf.int32, ref_to_z)
    ref_to_fn = building_blocks.Reference('fn', int_identity.type_signature)
    called_ref = building_blocks.Call(ref_to_fn, data)
    blk = building_blocks.Block([('fn', int_identity)], called_ref)

    resolved, modified = self._apply_resolution_and_remove_unused_block_locals(
        blk)

    self.assertTrue(modified)
    self.assertEqual(resolved.compact_representation(), '(z -> z)(data)')

  def test_resolves_call_to_functional_reference_under_tuple_selection(self):
    int_identity = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    ref_to_fn = building_blocks.Reference('function',
                                          int_identity.type_signature)
    whimsy_int = building_blocks.Data('data', tf.int32)
    tup_containing_fn = building_blocks.Struct([ref_to_fn, whimsy_int])
    nested_tuple = building_blocks.Struct([tup_containing_fn])
    selected_tuple = building_blocks.Selection(source=nested_tuple, index=0)
    selected_identity = building_blocks.Selection(
        source=selected_tuple, index=0)
    called_id = building_blocks.Call(selected_identity, whimsy_int)
    blk = building_blocks.Block([('function', int_identity)], called_id)

    resolved, transformed = self._apply_resolution_and_remove_unused_block_locals(
        blk)

    self.assertTrue(transformed)
    self.assertEqual(resolved.compact_representation(), '(x -> x)(data)')

  def test_resolves_call_which_correctly_returns_lambda(self):
    int_identity = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    higher_lambda = building_blocks.Lambda('z', tf.int32, int_identity)
    data = building_blocks.Data('data', tf.int32)
    call = building_blocks.Call(higher_lambda, data)

    resolved, transformed = self._apply_resolution_and_remove_unused_block_locals(
        call)

    self.assertTrue(transformed)
    self.assertEqual(resolved.compact_representation(), '(x -> x)')

  def test_resolves_capturing_function(self):
    # This test indicates that the current implementation of resolving higher
    # order functions does not respect necessary semantics for nondeterminism.
    noarg_lambda = building_blocks.Lambda(
        None, None, building_blocks.Data('data', tf.int32))
    called_lam = building_blocks.Call(noarg_lambda)
    ref_to_data = building_blocks.Reference('a', called_lam.type_signature)
    capturing_fn = building_blocks.Lambda('x', tf.int32, ref_to_data)
    blk_representing_captures = building_blocks.Block([('a', called_lam)],
                                                      capturing_fn)
    called_blk = building_blocks.Call(blk_representing_captures,
                                      building_blocks.Data('arg', tf.int32))
    tup_holding_blocks = building_blocks.Struct([called_blk, called_blk])

    tup_holding_blocks, _ = tree_transformations.uniquify_reference_names(
        tup_holding_blocks)
    resolved, transformed = self._apply_resolution_and_remove_unused_block_locals(
        tup_holding_blocks)

    self.assertTrue(transformed)
    self.assertEqual(
        resolved.compact_representation(),
        '<(let _var1=( -> data)() in (_var2 -> _var1)(arg)),(let _var3=( -> data)() in (_var4 -> _var3)(arg))>'
    )


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


class InsertTensorFlowIdentityAtLeavesTest(test_case.TestCase):

  def test_rasies_on_none(self):
    with self.assertRaises(TypeError):
      tree_transformations.insert_called_tf_identity_at_leaves(None)

  def test_transforms_simple_lambda(self):
    identity_lam = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    new_lambda, modified = tree_transformations.insert_called_tf_identity_at_leaves(
        identity_lam)
    self.assertTrue(modified)
    self.assertEqual(new_lambda.type_signature, identity_lam.type_signature)
    self.assertEqual(
        tree_analysis.count_types(new_lambda,
                                  building_blocks.CompiledComputation), 1)
    self.assertEqual(
        tree_analysis.count(new_lambda, _is_called_graph_pattern), 1)

  def test_transforms_reference_under_tuple(self):
    one_element_tuple = building_blocks.Struct(
        [building_blocks.Reference('x', tf.int32)])
    transformed_tuple, _ = tree_transformations.insert_called_tf_identity_at_leaves(
        one_element_tuple)
    self.assertIsInstance(transformed_tuple[0], building_blocks.Call)
    self.assertIsInstance(transformed_tuple[0].function,
                          building_blocks.CompiledComputation)
    self.assertIsInstance(transformed_tuple[0].argument,
                          building_blocks.Reference)
    self.assertEqual(transformed_tuple[0].argument.name, 'x')
    self.assertEqual(transformed_tuple[0].argument.type_signature,
                     computation_types.TensorType(tf.int32))

  def test_does_not_transform_references_to_federated_types(self):
    fed_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    identity_lam = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', fed_type))
    untransformed_lam, _ = tree_transformations.insert_called_tf_identity_at_leaves(
        identity_lam)
    self.assertEqual(identity_lam.compact_representation(),
                     untransformed_lam.compact_representation())

  def test_transforms_under_selection(self):
    ref_to_x = building_blocks.Reference('x', [tf.int32])
    sel = building_blocks.Selection(ref_to_x, index=0)
    lam = building_blocks.Lambda('x', [tf.int32], sel)
    new_lambda, modified = tree_transformations.insert_called_tf_identity_at_leaves(
        lam)
    self.assertTrue(modified)
    self.assertEqual(lam.type_signature, new_lambda.type_signature)
    self.assertEqual(
        tree_analysis.count_types(new_lambda,
                                  building_blocks.CompiledComputation), 1)
    self.assertEqual(
        tree_analysis.count(new_lambda, _is_called_graph_pattern), 1)

  def test_transforms_under_tuple(self):
    ref_to_x = building_blocks.Reference('x', tf.int32)
    tup = building_blocks.Struct([ref_to_x, ref_to_x])
    lam = building_blocks.Lambda('x', tf.int32, tup)
    new_lambda, modified = tree_transformations.insert_called_tf_identity_at_leaves(
        lam)
    self.assertTrue(modified)
    self.assertEqual(lam.type_signature, new_lambda.type_signature)
    self.assertEqual(
        tree_analysis.count_types(new_lambda,
                                  building_blocks.CompiledComputation), 2)
    self.assertEqual(
        tree_analysis.count(new_lambda, _is_called_graph_pattern), 2)

  def test_transforms_in_block_result(self):
    ref_to_x = building_blocks.Reference('x', tf.int32)
    block = building_blocks.Block([], ref_to_x)
    lam = building_blocks.Lambda('x', tf.int32, block)
    new_lambda, modified = tree_transformations.insert_called_tf_identity_at_leaves(
        lam)
    self.assertTrue(modified)
    self.assertEqual(lam.type_signature, new_lambda.type_signature)
    self.assertEqual(
        tree_analysis.count_types(new_lambda,
                                  building_blocks.CompiledComputation), 1)
    self.assertEqual(
        tree_analysis.count(new_lambda, _is_called_graph_pattern), 1)

  def test_transforms_in_block_locals(self):
    ref_to_x = building_blocks.Reference('x', tf.int32)
    data = building_blocks.Data('x', tf.int32)
    block = building_blocks.Block([('y', ref_to_x)], data)
    lam = building_blocks.Lambda('x', tf.int32, block)
    new_lambda, modified = tree_transformations.insert_called_tf_identity_at_leaves(
        lam)
    self.assertTrue(modified)
    self.assertEqual(lam.type_signature, new_lambda.type_signature)
    self.assertEqual(
        tree_analysis.count_types(new_lambda,
                                  building_blocks.CompiledComputation), 1)
    self.assertEqual(
        tree_analysis.count(new_lambda, _is_called_graph_pattern), 1)

  def test_transforms_under_call_without_compiled_computation(self):
    ref_to_x = building_blocks.Reference('x', [tf.int32])
    sel = building_blocks.Selection(ref_to_x, index=0)
    lam = building_blocks.Lambda('x', [tf.int32], sel)
    call = building_blocks.Call(lam, ref_to_x)
    lam = building_blocks.Lambda('x', [tf.int32], call)
    new_lambda, modified = tree_transformations.insert_called_tf_identity_at_leaves(
        lam)
    self.assertTrue(modified)
    self.assertEqual(lam.type_signature, new_lambda.type_signature)
    self.assertEqual(
        tree_analysis.count_types(new_lambda,
                                  building_blocks.CompiledComputation), 2)
    self.assertEqual(
        tree_analysis.count(new_lambda, _is_called_graph_pattern), 2)

  def test_noops_on_call_with_compiled_computation(self):
    ref_to_x = building_blocks.Reference('x', tf.int32)
    tensor_type = computation_types.TensorType(tf.int32)
    compiled = building_block_factory.create_compiled_identity(tensor_type)
    call = building_blocks.Call(compiled, ref_to_x)
    lam = building_blocks.Lambda('x', tf.int32, call)
    _, modified = tree_transformations.insert_called_tf_identity_at_leaves(lam)
    self.assertFalse(modified)


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


class GroupBlockLocalsByNamespaceTest(test_case.TestCase):

  def test_raises_non_block(self):
    ref = building_blocks.Reference('x', tf.int32)
    with self.assertRaises(TypeError):
      tree_transformations.group_block_locals_by_namespace(ref)

  def test_constructs_single_empty_list_with_no_block_locals(self):
    single_data = building_blocks.Data('a', tf.int32)
    block = building_blocks.Block([], single_data)
    classes = tree_transformations.group_block_locals_by_namespace(block)
    self.assertLen(classes, 1)
    self.assertLen(classes[0], 0)

  def test_adds_single_list_with_single_block_local(self):
    single_data = building_blocks.Data('a', tf.int32)
    ref_to_x = building_blocks.Reference('x', tf.int32)
    block = building_blocks.Block([('x', single_data)], ref_to_x)
    classes = tree_transformations.group_block_locals_by_namespace(block)
    self.assertLen(classes, 2)
    self.assertLen(classes[0], 1)
    self.assertEqual(classes[0][0][0], 'x')
    self.assertEqual(classes[0][0][1], single_data)

  def test_puts_computation_not_referencing_variable_into_first_list(self):
    first_data = building_blocks.Data('a', tf.int32)
    ref_to_x = building_blocks.Reference('x', tf.int32)
    second_data = building_blocks.Data('a', tf.int32)
    block = building_blocks.Block([('x', first_data), ('y', second_data)],
                                  ref_to_x)
    classes = tree_transformations.group_block_locals_by_namespace(block)
    self.assertLen(classes, 3)
    self.assertLen(classes[0], 2)
    self.assertEqual(classes[0][0][0], 'x')
    self.assertEqual(classes[0][0][1], first_data)
    self.assertEqual(classes[0][1][0], 'y')
    self.assertEqual(classes[0][1][1], second_data)

  def test_maintains_distinct_elements_in_partition_with_identical_python_objects_in_locals(
      self):
    data = building_blocks.Data('a', tf.int32)
    ref_to_x = building_blocks.Reference('x', tf.int32)
    block = building_blocks.Block([('x', data), ('y', data)], ref_to_x)
    classes = tree_transformations.group_block_locals_by_namespace(block)
    self.assertLen(classes, 3)
    self.assertLen(classes[0], 2)
    self.assertEqual(classes[0][0][0], 'x')
    self.assertEqual(classes[0][0][1], data)
    self.assertEqual(classes[0][1][0], 'y')
    self.assertEqual(classes[0][1][1], data)

  def test_leaves_computations_referencing_each_sequential_variable_in_singleton_lists(
      self):
    data = building_blocks.Data('a', tf.int32)
    ref_to_x = building_blocks.Reference('x', tf.int32)
    ref_to_y = building_blocks.Reference('y', tf.int32)
    block = building_blocks.Block([('x', data), ('y', ref_to_x),
                                   ('z', ref_to_y)], ref_to_x)
    classes = tree_transformations.group_block_locals_by_namespace(block)
    self.assertLen(classes, 4)
    self.assertLen(classes[0], 1)
    self.assertLen(classes[1], 1)
    self.assertLen(classes[2], 1)
    self.assertLen(classes[3], 0)
    self.assertEqual(classes[0][0][0], 'x')
    self.assertEqual(classes[0][0][1], data)
    self.assertEqual(classes[1][0][0], 'y')
    self.assertEqual(classes[1][0][1], ref_to_x)
    self.assertEqual(classes[2][0][0], 'z')
    self.assertEqual(classes[2][0][1], ref_to_y)

  def test_moves_computation_at_end_no_unbound_ref_to_beginning(self):
    first_data = building_blocks.Data('a', tf.int32)
    ref_to_x = building_blocks.Reference('x', tf.int32)
    second_data = building_blocks.Data('y', tf.int32)
    block = building_blocks.Block([('x', first_data), ('y', ref_to_x),
                                   ('z', second_data)], ref_to_x)
    classes = tree_transformations.group_block_locals_by_namespace(block)
    self.assertLen(classes, 4)
    self.assertLen(classes[0], 2)
    self.assertLen(classes[1], 1)
    self.assertLen(classes[2], 0)
    self.assertLen(classes[3], 0)
    self.assertEqual(classes[0][0][0], 'x')
    self.assertEqual(classes[0][0][1], first_data)
    self.assertEqual(classes[0][1][0], 'z')
    self.assertEqual(classes[0][1][1], second_data)
    self.assertEqual(classes[1][0][0], 'y')
    self.assertEqual(classes[1][0][1], ref_to_x)


if __name__ == '__main__':
  test_case.main()
