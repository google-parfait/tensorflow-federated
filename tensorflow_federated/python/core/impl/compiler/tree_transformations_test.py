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

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.compiler import building_block_analysis
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import test_utils as compiler_test_utils
from tensorflow_federated.python.core.impl.compiler import transformation_utils
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.compiler import tree_transformations
from tensorflow_federated.python.core.impl.types import placement_literals


def _create_chained_dummy_federated_applys(functions, arg):
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


def _create_chained_dummy_federated_maps(functions, arg):
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


def _create_lambda_to_dummy_cast(parameter_name, parameter_type, result_type):
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
  federated_type = computation_types.FederatedType(tf.int32,
                                                   placement_literals.SERVER)
  ref = building_blocks.Reference('b', federated_type)
  called_federated_broadcast = building_block_factory.create_federated_broadcast(
      ref)
  called_federated_map = building_block_factory.create_federated_map(
      compiled, called_federated_broadcast)
  called_federated_mean = building_block_factory.create_federated_mean(
      called_federated_map, None)
  tup = building_blocks.Struct([called_federated_mean, called_federated_mean])
  return building_blocks.Lambda('b', tf.int32, tup)


class ExtractComputationsTest(test_case.TestCase):

  def test_raises_type_error_with_none(self):
    with self.assertRaises(TypeError):
      tree_transformations.extract_computations(None)

  def test_raises_value_error_with_non_unique_variable_names(self):
    data = building_blocks.Data('data', tf.int32)
    block = building_blocks.Block([('a', data), ('a', data)], data)
    with self.assertRaises(ValueError):
      tree_transformations.extract_computations(block)

  def test_extracts_from_no_arg_lamda(self):
    data = building_blocks.Data('data', tf.int32)
    block = building_blocks.Lambda(
        parameter_name=None, parameter_type=None, result=data)
    comp = block

    transformed_comp, modified = tree_transformations.extract_computations(comp)

    self.assertEqual(comp.compact_representation(), '( -> data)')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let _var1=data,_var2=( -> _var1) in _var2)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_no_arg_lamda_to_block(self):
    data = building_blocks.Data('data', tf.int32)
    blk = building_blocks.Block([], data)
    block = building_blocks.Lambda(
        parameter_name=None, parameter_type=None, result=blk)
    comp = block

    transformed_comp, modified = tree_transformations.extract_computations(comp)

    self.assertEqual(comp.compact_representation(), '( -> (let  in data))')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let _var1=data,_var2=_var1,_var3=( -> _var2) in _var3)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_block_one_comp(self):
    data = building_blocks.Data('data', tf.int32)
    block = building_blocks.Block([('a', data)], data)
    comp = block

    transformed_comp, modified = tree_transformations.extract_computations(comp)

    self.assertEqual(comp.compact_representation(), '(let a=data in data)')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let a=data,_var1=data in _var1)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_block_multiple_comps(self):
    data_1 = building_blocks.Data('data', tf.int32)
    data_2 = building_blocks.Data('data', tf.int32)
    data_3 = building_blocks.Data('data', tf.int32)
    tup = building_blocks.Struct([data_2, data_3])
    block = building_blocks.Block([('a', data_1)], tup)
    comp = block

    transformed_comp, modified = tree_transformations.extract_computations(comp)

    self.assertEqual(comp.compact_representation(),
                     '(let a=data in <data,data>)')
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(let\n'
        '  a=data,\n'
        '  _var1=data,\n'
        '  _var2=data,\n'
        '  _var3=<\n'
        '    _var1,\n'
        '    _var2\n'
        '  >,\n'
        '  _var4=_var3\n'
        ' in _var4)'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_call_one_comp(self):
    fn = compiler_test_utils.create_identity_function('a', tf.int32)
    data = building_blocks.Data('data', tf.int32)
    call = building_blocks.Call(fn, data)
    comp = call

    transformed_comp, modified = tree_transformations.extract_computations(comp)

    self.assertEqual(comp.compact_representation(), '(a -> a)(data)')
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(let\n'
        '  _var1=(a -> a),\n'
        '  _var2=data,\n'
        '  _var3=_var1(_var2)\n'
        ' in _var3)'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_call_multiple_comps(self):
    fn = compiler_test_utils.create_identity_function('a', [tf.int32, tf.int32])
    data_1 = building_blocks.Data('data', tf.int32)
    data_2 = building_blocks.Data('data', tf.int32)
    tup = building_blocks.Struct([data_1, data_2])
    call = building_blocks.Call(fn, tup)
    comp = call

    transformed_comp, modified = tree_transformations.extract_computations(comp)

    self.assertEqual(comp.compact_representation(), '(a -> a)(<data,data>)')
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(let\n'
        '  _var4=(a -> a),\n'
        '  _var1=data,\n'
        '  _var2=data,\n'
        '  _var3=<\n'
        '    _var1,\n'
        '    _var2\n'
        '  >,\n'
        '  _var5=_var3,\n'
        '  _var6=_var4(_var5)\n'
        ' in _var6)'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_lambda_one_comp(self):
    data = building_blocks.Data('data', tf.int32)
    fn = building_blocks.Lambda('a', tf.int32, data)
    comp = fn

    transformed_comp, modified = tree_transformations.extract_computations(comp)

    self.assertEqual(comp.compact_representation(), '(a -> data)')
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(let\n'
        '  _var1=data,\n'
        '  _var2=(a -> _var1)\n'
        ' in _var2)'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_lambda_multiple_comps(self):
    data_1 = building_blocks.Data('data', tf.int32)
    data_2 = building_blocks.Data('data', tf.int32)
    tup = building_blocks.Struct([data_1, data_2])
    fn = building_blocks.Lambda('a', tf.int32, tup)
    comp = fn

    transformed_comp, modified = tree_transformations.extract_computations(comp)

    self.assertEqual(comp.compact_representation(), '(a -> <data,data>)')
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(let\n'
        '  _var1=data,\n'
        '  _var2=data,\n'
        '  _var3=<\n'
        '    _var1,\n'
        '    _var2\n'
        '  >,\n'
        '  _var4=_var3,\n'
        '  _var5=(a -> _var4)\n'
        ' in _var5)'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_selection_one_comp(self):
    data = building_blocks.Data('data', tf.int32)
    tup = building_blocks.Struct([data])
    sel = building_blocks.Selection(tup, index=0)
    comp = sel

    transformed_comp, modified = tree_transformations.extract_computations(comp)

    self.assertEqual(comp.compact_representation(), '<data>[0]')
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(let\n'
        '  _var1=data,\n'
        '  _var2=<\n'
        '    _var1\n'
        '  >,\n'
        '  _var3=_var2,\n'
        '  _var4=_var3[0]\n'
        ' in _var4)'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_selection_multiple_comps(self):
    data_1 = building_blocks.Data('data', tf.int32)
    data_2 = building_blocks.Data('data', tf.int32)
    tup = building_blocks.Struct([data_1, data_2])
    sel = building_blocks.Selection(tup, index=0)
    comp = sel

    transformed_comp, modified = tree_transformations.extract_computations(comp)

    self.assertEqual(comp.compact_representation(), '<data,data>[0]')
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(let\n'
        '  _var1=data,\n'
        '  _var2=data,\n'
        '  _var3=<\n'
        '    _var1,\n'
        '    _var2\n'
        '  >,\n'
        '  _var4=_var3,\n'
        '  _var5=_var4[0]\n'
        ' in _var5)'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_tuple_one_comp(self):
    data = building_blocks.Data('data', tf.int32)
    tup = building_blocks.Struct([data])
    comp = tup

    transformed_comp, modified = tree_transformations.extract_computations(comp)

    self.assertEqual(comp.compact_representation(), '<data>')
    self.assertEqual(transformed_comp.compact_representation(),
                     '(let _var1=data,_var2=<_var1> in _var2)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_tuple_multiple_comps(self):
    data_1 = building_blocks.Data('data', tf.int32)
    data_2 = building_blocks.Data('data', tf.int32)
    tup = building_blocks.Struct([data_1, data_2])
    comp = tup

    transformed_comp, modified = tree_transformations.extract_computations(comp)

    self.assertEqual(comp.compact_representation(), '<data,data>')
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(let _var1=data,_var2=data,_var3=<_var1,_var2> in _var3)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_tuple_named_comps(self):
    data_1 = building_blocks.Data('data', tf.int32)
    data_2 = building_blocks.Data('data', tf.int32)
    tup = building_blocks.Struct([
        ('a', data_1),
        ('b', data_2),
    ])
    comp = tup

    transformed_comp, modified = tree_transformations.extract_computations(comp)

    self.assertEqual(comp.compact_representation(), '<a=data,b=data>')
    self.assertEqual(
        transformed_comp.compact_representation(),
        '(let _var1=data,_var2=data,_var3=<a=_var1,b=_var2> in _var3)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_federated_aggregate(self):
    called_intrinsic = compiler_test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    comp = called_intrinsic

    transformed_comp, modified = tree_transformations.extract_computations(comp)

    self.assertEqual(
        comp.compact_representation(),
        'federated_aggregate(<data,data,(a -> data),(b -> data),(c -> data)>)')
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(let\n'
        '  _var13=federated_aggregate,\n'
        '  _var7=data,\n'
        '  _var8=data,\n'
        '  _var1=data,\n'
        '  _var2=(a -> _var1),\n'
        '  _var9=_var2,\n'
        '  _var3=data,\n'
        '  _var4=(b -> _var3),\n'
        '  _var10=_var4,\n'
        '  _var5=data,\n'
        '  _var6=(c -> _var5),\n'
        '  _var11=_var6,\n'
        '  _var12=<\n'
        '    _var7,\n'
        '    _var8,\n'
        '    _var9,\n'
        '    _var10,\n'
        '    _var11\n'
        '  >,\n'
        '  _var14=_var12,\n'
        '  _var15=_var13(_var14)\n'
        ' in _var15)'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_federated_broadcast(self):
    called_intrinsic = compiler_test_utils.create_dummy_called_federated_broadcast(
    )
    comp = called_intrinsic

    transformed_comp, modified = tree_transformations.extract_computations(comp)

    self.assertEqual(comp.compact_representation(), 'federated_broadcast(data)')
    # pyformat: disable
    self.assertEqual(
        transformed_comp.formatted_representation(),
        '(let\n'
        '  _var1=federated_broadcast,\n'
        '  _var2=data,\n'
        '  _var3=_var1(_var2)\n'
        ' in _var3)'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_complex_comp(self):
    complex_comp = _create_complex_computation()
    comp = complex_comp

    transformed_comp, modified = tree_transformations.extract_computations(comp)

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
        '  _var4=_var2,\n'
        '  _var5=<\n'
        '    _var3,\n'
        '    _var4\n'
        '  >,\n'
        '  _var7=_var5,\n'
        '  _var8=_var6(_var7),\n'
        '  _var10=_var8,\n'
        '  _var11=_var9(_var10),\n'
        '  _var23=_var11,\n'
        '  _var20=federated_mean,\n'
        '  _var17=federated_map,\n'
        '  _var14=comp#a,\n'
        '  _var12=federated_broadcast,\n'
        '  _var13=_var12(b),\n'
        '  _var15=_var13,\n'
        '  _var16=<\n'
        '    _var14,\n'
        '    _var15\n'
        '  >,\n'
        '  _var18=_var16,\n'
        '  _var19=_var17(_var18),\n'
        '  _var21=_var19,\n'
        '  _var22=_var20(_var21),\n'
        '  _var24=_var22,\n'
        '  _var25=<\n'
        '    _var23,\n'
        '    _var24\n'
        '  >,\n'
        '  _var26=_var25\n'
        ' in _var26))'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)


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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic_1 = compiler_test_utils.create_dummy_called_intrinsic(
        parameter_name='a')
    called_intrinsic_2 = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic_1 = compiler_test_utils.create_dummy_called_intrinsic(
        parameter_name='a')
    called_intrinsic_2 = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic_1 = compiler_test_utils.create_dummy_called_intrinsic(
        parameter_name='a')
    called_intrinsic_2 = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic_1 = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_intrinsic(
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
    block = compiler_test_utils.create_identity_block_with_dummy_data(
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
    block = compiler_test_utils.create_identity_block_with_dummy_data(
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
    block_1 = compiler_test_utils.create_identity_block_with_dummy_data(
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
    block = compiler_test_utils.create_identity_block_with_dummy_data(
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
    arg_type = computation_types.FederatedType(tf.int32,
                                               placement_literals.SERVER)
    arg = building_blocks.Data('data', arg_type)
    call = _create_chained_dummy_federated_applys([fn, fn], arg)
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
    arg_type = computation_types.FederatedType(tf.int32,
                                               placement_literals.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    call = _create_chained_dummy_federated_maps([fn, fn], arg)
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
    arg_type = computation_types.FederatedType(tf.int32,
                                               placement_literals.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    fn_2 = compiler_test_utils.create_identity_function('b', tf.int32)
    call = _create_chained_dummy_federated_maps([fn_1, fn_2], arg)
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
    fn_1 = _create_lambda_to_dummy_cast('a', tf.int32, tf.float32)
    arg_type = computation_types.FederatedType(tf.int32,
                                               placement_literals.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    fn_2 = compiler_test_utils.create_identity_function('b', tf.float32)
    call = _create_chained_dummy_federated_maps([fn_1, fn_2], arg)
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
                                               placement_literals.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    call = _create_chained_dummy_federated_maps([fn, fn], arg)
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
    arg_type = computation_types.FederatedType(tf.int32,
                                               placement_literals.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    call = _create_chained_dummy_federated_maps([fn, fn], arg)
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
    arg_type = computation_types.FederatedType(tf.int32,
                                               placement_literals.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    call = _create_chained_dummy_federated_maps([fn, fn], arg)
    block = compiler_test_utils.create_dummy_block(call, variable_name='b')
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
    arg_type = computation_types.FederatedType(tf.int32,
                                               placement_literals.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    call = _create_chained_dummy_federated_maps([fn, fn, fn], arg)
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
    arg_type = computation_types.FederatedType(tf.int32,
                                               placement_literals.CLIENTS)
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
    arg_type = computation_types.FederatedType(tf.int32,
                                               placement_literals.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    call_1 = building_block_factory.create_federated_map(fn, arg)
    block = compiler_test_utils.create_dummy_block(call_1, variable_name='b')
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
    called_intrinsic = compiler_test_utils.create_dummy_called_federated_map(
        parameter_name='a')
    calls = building_blocks.Struct((called_intrinsic, called_intrinsic))
    comp = calls
    with self.assertRaises(TypeError):
      tree_transformations.merge_tuple_intrinsics(comp, None)

  def test_raises_value_error(self):
    called_intrinsic = compiler_test_utils.create_dummy_called_federated_map(
        parameter_name='a')
    calls = building_blocks.Struct((called_intrinsic, called_intrinsic))
    comp = calls
    with self.assertRaises(ValueError):
      tree_transformations.merge_tuple_intrinsics(comp, 'dummy')

  def test_merges_federated_aggregates(self):
    called_intrinsic = compiler_test_utils.create_dummy_called_federated_aggregate(
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
        str(transformed_comp.type_signature), '<bool@SERVER,bool@SERVER>')
    self.assertTrue(modified)

  def test_merges_federated_aggregates_with_unknown_parameter_dim(self):
    value_type = tf.int32
    federated_value_type = computation_types.FederatedType(
        value_type, placement_literals.CLIENTS)
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
    called_intrinsic = compiler_test_utils.create_dummy_called_federated_aggregate(
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
        '    federated_map(<\n'
        '      (zipped_tree -> <\n'
        '        zipped_tree[0][0],\n'
        '        zipped_tree[0][1],\n'
        '        zipped_tree[1]\n'
        '      >),\n'
        '      (let\n'
        '        value=<\n'
        '          data,\n'
        '          data,\n'
        '          data\n'
        '        >\n'
        '       in federated_zip_at_clients(<\n'
        '        federated_zip_at_clients(<\n'
        '          value[0],\n'
        '          value[1]\n'
        '        >),\n'
        '        value[2]\n'
        '      >))\n'
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
        '<bool@SERVER,bool@SERVER,bool@SERVER>')
    self.assertTrue(modified)

  def test_merges_federated_applys(self):
    called_intrinsic = compiler_test_utils.create_dummy_called_federated_apply(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_federated_broadcast(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_federated_map(
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
    called_intrinsic_1 = compiler_test_utils.create_dummy_called_federated_map(
        parameter_name='a')
    called_intrinsic_2 = compiler_test_utils.create_dummy_called_federated_map(
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
    called_intrinsic_1 = compiler_test_utils.create_dummy_called_federated_map(
        parameter_name='a', parameter_type=tf.int32)
    called_intrinsic_2 = compiler_test_utils.create_dummy_called_federated_map(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_federated_map(
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
    called_intrinsic_1 = compiler_test_utils.create_dummy_called_federated_map(
        parameter_name='a', parameter_type=parameter_type_1)
    parameter_type_2 = [('e', tf.bool), ('f', tf.string)]
    called_intrinsic_2 = compiler_test_utils.create_dummy_called_federated_map(
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
    arg_type = computation_types.FederatedType(tf.int32,
                                               placement_literals.CLIENTS)
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
    called_intrinsic = compiler_test_utils.create_dummy_called_federated_map(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_federated_map(
        parameter_name='a')
    calls = building_blocks.Struct((called_intrinsic, called_intrinsic))
    block = compiler_test_utils.create_dummy_block(calls, variable_name='a')
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
    called_intrinsic = compiler_test_utils.create_dummy_called_federated_map(
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
        '    federated_map(<\n'
        '      (zipped_tree -> <\n'
        '        zipped_tree[0][0],\n'
        '        zipped_tree[0][1],\n'
        '        zipped_tree[1]\n'
        '      >),\n'
        '      (let\n'
        '        value=<\n'
        '          data,\n'
        '          data,\n'
        '          data\n'
        '        >\n'
        '       in federated_zip_at_clients(<\n'
        '        federated_zip_at_clients(<\n'
        '          value[0],\n'
        '          value[1]\n'
        '        >),\n'
        '        value[2]\n'
        '      >))\n'
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
    called_intrinsic = compiler_test_utils.create_dummy_called_federated_map(
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
        '    federated_map(<\n'
        '      (arg -> <\n'
        '        arg\n'
        '      >),\n'
        '      <\n'
        '        data\n'
        '      >[0]\n'
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
    called_intrinsic_1 = compiler_test_utils.create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    called_intrinsic_2 = compiler_test_utils.create_dummy_called_federated_map(
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
        str(transformed_comp.type_signature), '<bool@SERVER,{int32}@CLIENTS>')
    self.assertFalse(modified)

  def test_does_not_merge_intrinsics_with_different_uri(self):
    called_intrinsic = compiler_test_utils.create_dummy_called_federated_map(
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
    called_intrinsic = compiler_test_utils.create_dummy_called_federated_aggregate(
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
        '  _dep8=data,\n'
        '  _dep10=(_dep9 -> _dep8),\n'
        '  _dep11=<\n'
        '    _dep2,\n'
        '    _dep3,\n'
        '    _dep5,\n'
        '    _dep7,\n'
        '    _dep10\n'
        '  >,\n'
        '  _dep12=_dep1(_dep11),\n'
        '  _dep13=<\n'
        '    _dep12,\n'
        '    _dep12\n'
        '  >\n'
        ' in _dep13)'
    )
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_removes_federated_broadcast(self):
    called_intrinsic = compiler_test_utils.create_dummy_called_federated_broadcast(
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


class DeduplicateBuildingBlocksTest(test_case.TestCase):

  def test_removes_multiple_selections(self):
    id_lam = building_blocks.Lambda(
        'x', [tf.int32, tf.float32],
        building_blocks.Reference('x', [tf.int32, tf.float32]))
    ref_to_a = building_blocks.Reference('a', [tf.int32, tf.float32])
    called_id = building_blocks.Call(id_lam, ref_to_a)
    sel_0 = building_blocks.Selection(called_id, index=0)
    tup = building_blocks.Struct([sel_0, sel_0])
    fake_lam = building_blocks.Lambda('a', [tf.int32, tf.float32], tup)
    dups_removed, modified = tree_transformations.remove_duplicate_building_blocks(
        fake_lam)
    self.assertTrue(modified)
    self.assertEqual(
        tree_analysis.count_types(dups_removed, building_blocks.Selection), 1)


class RemoveMappedOrAppliedIdentityTest(parameterized.TestCase):

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      tree_transformations.remove_mapped_or_applied_identity(None)

  # pyformat: disable
  @parameterized.named_parameters(
      ('federated_apply',
       intrinsic_defs.FEDERATED_APPLY.uri,
       compiler_test_utils.create_dummy_called_federated_apply),
      ('federated_map',
       intrinsic_defs.FEDERATED_MAP.uri,
       compiler_test_utils.create_dummy_called_federated_map),
      ('federated_map_all_equal',
       intrinsic_defs.FEDERATED_MAP_ALL_EQUAL.uri,
       compiler_test_utils.create_dummy_called_federated_map_all_equal),
      ('sequence_map',
       intrinsic_defs.SEQUENCE_MAP.uri,
       compiler_test_utils.create_dummy_called_sequence_map),
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
                                               placement_literals.CLIENTS)
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
    called_intrinsic = compiler_test_utils.create_dummy_called_federated_map(
        parameter_name='a')
    block = compiler_test_utils.create_dummy_block(
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
    arg_type = computation_types.FederatedType(tf.int32,
                                               placement_literals.CLIENTS)
    arg = building_blocks.Data('data', arg_type)
    call = _create_chained_dummy_federated_maps([fn, fn], arg)
    comp = call

    transformed_comp, modified = tree_transformations.remove_mapped_or_applied_identity(
        comp)

    self.assertEqual(
        comp.compact_representation(),
        'federated_map(<(a -> a),federated_map(<(a -> a),data>)>)')
    self.assertEqual(transformed_comp.compact_representation(), 'data')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_does_not_remove_dummy_intrinsic(self):
    comp = compiler_test_utils.create_dummy_called_intrinsic(parameter_name='a')

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
    block = compiler_test_utils.create_dummy_block(call, variable_name='b')
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
    block = compiler_test_utils.create_dummy_block(fn, variable_name='b')
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
    dummy_int = building_blocks.Data('data', tf.int32)
    blk_with_renamed = building_blocks.Block([('x', dummy_int),
                                              ('x', dummy_int)], dummy_int)
    with self.assertRaises(ValueError):
      tree_transformations.resolve_higher_order_functions(blk_with_renamed)

  def test_raises_with_unsafe_rebinding(self):
    dummy_int = building_blocks.Data('data', tf.int32)
    ref_to_x = building_blocks.Reference('x', tf.int32)
    blk_with_renamed = building_blocks.Block([('y', ref_to_x),
                                              ('x', dummy_int)], dummy_int)
    with self.assertRaises(ValueError):
      tree_transformations.resolve_higher_order_functions(blk_with_renamed)

  def test_resolves_selection_from_call(self):
    int_identity = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    dummy_int = building_blocks.Data('data', tf.int32)
    tup_containing_fn = building_blocks.Struct([int_identity, dummy_int])
    lambda_returning_tup = building_blocks.Lambda('z', tf.int32,
                                                  tup_containing_fn)
    called_tup = building_blocks.Call(lambda_returning_tup, dummy_int)
    selected_identity = building_blocks.Selection(source=called_tup, index=0)
    called_id = building_blocks.Call(selected_identity, dummy_int)

    resolved, transformed = self._apply_resolution_and_remove_unused_block_locals(
        called_id)

    self.assertTrue(transformed)
    self.assertEqual(resolved.compact_representation(), '(x -> x)(data)')

  def test_resolves_selection_from_call_with_no_param_lambda(self):
    materialize_int = building_blocks.Lambda(
        None, None, building_blocks.Data('x', tf.int32))
    dummy_int = building_blocks.Data('data', tf.int32)
    tup_containing_fn = building_blocks.Struct([materialize_int, dummy_int])
    lambda_returning_tup = building_blocks.Lambda('z', tf.int32,
                                                  tup_containing_fn)
    called_tup = building_blocks.Call(lambda_returning_tup, dummy_int)
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
    dummy_int = building_blocks.Data('data', tf.int32)
    tup_containing_fn = building_blocks.Struct([int_identity, dummy_int])
    lambda_returning_tup = building_blocks.Lambda('z', tf.int32,
                                                  tup_containing_fn)
    called_tup = building_blocks.Call(lambda_returning_tup, dummy_int)
    ref_to_y = building_blocks.Reference('y', called_tup.type_signature)
    selected_identity = building_blocks.Selection(source=ref_to_y, index=0)
    called_id = building_blocks.Call(selected_identity, dummy_int)
    blk = building_blocks.Block([('y', called_tup)], called_id)

    resolved, transformed = self._apply_resolution_and_remove_unused_block_locals(
        blk)

    self.assertTrue(transformed)
    self.assertEqual(resolved.compact_representation(), '(x -> x)(data)')

  def test_resolves_selection_from_tuple(self):
    int_identity = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    dummy_int = building_blocks.Data('data', tf.int32)
    tup_containing_fn = building_blocks.Struct([int_identity, dummy_int])
    selected_identity = building_blocks.Selection(
        source=tup_containing_fn, index=0)
    called_id = building_blocks.Call(selected_identity, dummy_int)

    resolved, transformed = self._apply_resolution_and_remove_unused_block_locals(
        called_id)

    self.assertTrue(transformed)
    self.assertEqual(resolved.compact_representation(), '(x -> x)(data)')

  def test_resolves_selection_from_nested_tuple(self):
    int_identity = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    dummy_int = building_blocks.Data('data', tf.int32)
    tup_containing_fn = building_blocks.Struct([int_identity, dummy_int])
    nested_tuple = building_blocks.Struct([tup_containing_fn])
    selected_tuple = building_blocks.Selection(source=nested_tuple, index=0)
    selected_identity = building_blocks.Selection(
        source=selected_tuple, index=0)
    called_id = building_blocks.Call(selected_identity, dummy_int)

    resolved, transformed = self._apply_resolution_and_remove_unused_block_locals(
        called_id)

    self.assertTrue(transformed)
    self.assertEqual(resolved.compact_representation(), '(x -> x)(data)')

  def test_resolves_selection_from_symbol_bound_to_tuple(self):
    int_identity = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    dummy_int = building_blocks.Data('data', tf.int32)
    tup_containing_fn = building_blocks.Struct([int_identity, dummy_int])
    ref_to_y = building_blocks.Reference('y', tup_containing_fn.type_signature)
    selected_identity = building_blocks.Selection(source=ref_to_y, index=0)
    called_id = building_blocks.Call(selected_identity, dummy_int)
    blk = building_blocks.Block([('y', tup_containing_fn)], called_id)

    resolved, transformed = self._apply_resolution_and_remove_unused_block_locals(
        blk)

    self.assertTrue(transformed)
    self.assertEqual(resolved.compact_representation(), '(x -> x)(data)')

  def test_resolves_selection_from_nested_symbol_bound_to_nested_tuple(self):
    int_identity = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    dummy_int = building_blocks.Data('data', tf.int32)
    tup_containing_fn = building_blocks.Struct([int_identity, dummy_int])
    nested_tuple = building_blocks.Struct([tup_containing_fn])
    ref_to_nested_tuple = building_blocks.Reference('nested_tuple',
                                                    nested_tuple.type_signature)
    selected_tuple = building_blocks.Selection(
        source=ref_to_nested_tuple, index=0)
    selected_identity = building_blocks.Selection(
        source=selected_tuple, index=0)
    called_id = building_blocks.Call(selected_identity, dummy_int)
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
    dummy_int = building_blocks.Data('data', tf.int32)
    tup_containing_fn = building_blocks.Struct([ref_to_fn, dummy_int])
    nested_tuple = building_blocks.Struct([tup_containing_fn])
    selected_tuple = building_blocks.Selection(source=nested_tuple, index=0)
    selected_identity = building_blocks.Selection(
        source=selected_tuple, index=0)
    called_id = building_blocks.Call(selected_identity, dummy_int)
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
    self.assertTrue(transformation_utils.has_unique_names(transformed_comp))
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
    self.assertTrue(transformation_utils.has_unique_names(transformed_comp))
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
    self.assertTrue(transformation_utils.has_unique_names(transformed_comp))
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
    self.assertTrue(transformation_utils.has_unique_names(transformed_comp))
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
    self.assertTrue(transformation_utils.has_unique_names(transformed_comp))
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
    fed_type = computation_types.FederatedType(tf.int32,
                                               placement_literals.CLIENTS)
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


class UnwrapPlacementTest(parameterized.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      tree_transformations.unwrap_placement(None)

  def test_raises_computation_non_federated_type(self):
    with self.assertRaises(TypeError):
      tree_transformations.unwrap_placement(building_blocks.Data('x', tf.int32))

  def test_raises_unbound_reference_non_federated_type(self):
    block = building_blocks.Block(
        [('x', building_blocks.Reference('y', tf.int32))],
        building_blocks.Reference(
            'x',
            computation_types.FederatedType(tf.int32,
                                            placement_literals.CLIENTS)))
    with self.assertRaisesRegex(TypeError, 'lone unbound reference'):
      tree_transformations.unwrap_placement(block)

  def test_raises_two_unbound_references(self):
    ref_to_x = building_blocks.Reference(
        'x', computation_types.FederatedType(tf.int32,
                                             placement_literals.SERVER))
    ref_to_y = building_blocks.Reference(
        'y', computation_types.FunctionType(tf.int32, tf.float32))
    applied = building_block_factory.create_federated_apply(ref_to_y, ref_to_x)
    with self.assertRaises(ValueError):
      tree_transformations.unwrap_placement(applied)

  def test_raises_disallowed_intrinsic(self):
    fed_ref = building_blocks.Reference(
        'x', computation_types.FederatedType(tf.int32,
                                             placement_literals.SERVER))
    broadcaster = building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_BROADCAST.uri,
        computation_types.FunctionType(
            fed_ref.type_signature,
            computation_types.FederatedType(
                fed_ref.type_signature.member,
                placement_literals.CLIENTS,
                all_equal=True)))
    called_broadcast = building_blocks.Call(broadcaster, fed_ref)
    with self.assertRaises(ValueError):
      tree_transformations.unwrap_placement(called_broadcast)

  def test_raises_multiple_placement_literals(self):
    server_placed_data = building_blocks.Data(
        'x', computation_types.FederatedType(tf.int32,
                                             placement_literals.SERVER))
    clients_placed_data = building_blocks.Data(
        'y',
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    block_holding_both = building_blocks.Block([('x', server_placed_data)],
                                               clients_placed_data)
    with self.assertRaisesRegex(ValueError, 'contains a placement other than'):
      tree_transformations.unwrap_placement(block_holding_both)

  def test_passes_unbound_type_signature_obscured_under_block(self):
    fed_ref = building_blocks.Reference(
        'x', computation_types.FederatedType(tf.int32,
                                             placement_literals.SERVER))
    block = building_blocks.Block(
        [('y', fed_ref), ('x', building_blocks.Data('dummy', tf.int32)),
         ('z', building_blocks.Reference('x', tf.int32))],
        building_blocks.Reference('y', fed_ref.type_signature))
    tree_transformations.unwrap_placement(block)

  def test_passes_noarg_lambda(self):
    lam = building_blocks.Lambda(None, None,
                                 building_blocks.Data('a', tf.int32))
    fed_int_type = computation_types.FederatedType(tf.int32,
                                                   placement_literals.SERVER)
    fed_eval = building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_EVAL_AT_SERVER.uri,
        computation_types.FunctionType(lam.type_signature, fed_int_type))
    called_eval = building_blocks.Call(fed_eval, lam)
    tree_transformations.unwrap_placement(called_eval)

  def test_removes_federated_types_under_function(self):
    int_ref = building_blocks.Reference('x', tf.int32)
    int_id = building_blocks.Lambda('x', tf.int32, int_ref)
    fed_ref = building_blocks.Reference(
        'x', computation_types.FederatedType(tf.int32,
                                             placement_literals.SERVER))
    applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, fed_ref)
    second_applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, applied_id)
    placement_unwrapped, modified = tree_transformations.unwrap_placement(
        second_applied_id)
    self.assertTrue(modified)

    def _fed_type_predicate(x):
      return x.type_signature.is_federated()

    self.assertEqual(placement_unwrapped.function.uri,
                     intrinsic_defs.FEDERATED_APPLY.uri)
    self.assertEqual(
        tree_analysis.count(placement_unwrapped.argument[0],
                            _fed_type_predicate), 0)

  def test_unwrap_placement_removes_one_federated_apply(self):
    int_ref = building_blocks.Reference('x', tf.int32)
    int_id = building_blocks.Lambda('x', tf.int32, int_ref)
    fed_ref = building_blocks.Reference(
        'x', computation_types.FederatedType(tf.int32,
                                             placement_literals.SERVER))
    applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, fed_ref)
    second_applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, applied_id)
    placement_unwrapped, modified = tree_transformations.unwrap_placement(
        second_applied_id)
    self.assertTrue(modified)

    self.assertEqual(
        second_applied_id.compact_representation(),
        'federated_apply(<(x -> x),federated_apply(<(x -> x),x>)>)')
    self.assertEqual(
        _count_called_intrinsics(second_applied_id,
                                 intrinsic_defs.FEDERATED_APPLY.uri), 2)
    self.assertEqual(
        _count_called_intrinsics(placement_unwrapped,
                                 intrinsic_defs.FEDERATED_APPLY.uri), 1)
    self.assertEqual(placement_unwrapped.type_signature,
                     second_applied_id.type_signature)
    self.assertIsInstance(placement_unwrapped, building_blocks.Call)
    self.assertIsInstance(placement_unwrapped.argument[0],
                          building_blocks.Lambda)
    self.assertIsInstance(placement_unwrapped.argument[0].result,
                          building_blocks.Call)
    self.assertEqual(
        placement_unwrapped.argument[0].result.function.compact_representation(
        ), '(x -> x)')
    self.assertEqual(
        placement_unwrapped.argument[0].result.argument.compact_representation(
        ), '(x -> x)(_var1)')

  def test_unwrap_placement_removes_two_federated_applys(self):
    int_ref = building_blocks.Reference('x', tf.int32)
    int_id = building_blocks.Lambda('x', tf.int32, int_ref)
    fed_ref = building_blocks.Reference(
        'x', computation_types.FederatedType(tf.int32,
                                             placement_literals.SERVER))
    applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, fed_ref)
    second_applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, applied_id)
    third_applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, second_applied_id)
    placement_unwrapped, modified = tree_transformations.unwrap_placement(
        second_applied_id)
    self.assertTrue(modified)

    self.assertEqual(
        _count_called_intrinsics(third_applied_id,
                                 intrinsic_defs.FEDERATED_APPLY.uri), 3)
    self.assertEqual(
        _count_called_intrinsics(placement_unwrapped,
                                 intrinsic_defs.FEDERATED_APPLY.uri), 1)

  def test_unwrap_placement_removes_one_federated_map(self):
    int_ref = building_blocks.Reference('x', tf.int32)
    int_id = building_blocks.Lambda('x', tf.int32, int_ref)
    fed_ref = building_blocks.Reference(
        'x',
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, fed_ref)
    second_applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, applied_id)
    placement_unwrapped, modified = tree_transformations.unwrap_placement(
        second_applied_id)
    self.assertTrue(modified)

    self.assertEqual(second_applied_id.compact_representation(),
                     'federated_map(<(x -> x),federated_map(<(x -> x),x>)>)')
    self.assertEqual(
        _count_called_intrinsics(second_applied_id,
                                 intrinsic_defs.FEDERATED_MAP.uri), 2)
    self.assertEqual(
        _count_called_intrinsics(placement_unwrapped,
                                 intrinsic_defs.FEDERATED_MAP.uri), 1)
    self.assertEqual(placement_unwrapped.type_signature,
                     second_applied_id.type_signature)
    self.assertIsInstance(placement_unwrapped, building_blocks.Call)
    self.assertIsInstance(placement_unwrapped.argument[0],
                          building_blocks.Lambda)
    self.assertIsInstance(placement_unwrapped.argument[0].result,
                          building_blocks.Call)
    self.assertEqual(
        placement_unwrapped.argument[0].result.function.compact_representation(
        ), '(x -> x)')
    self.assertEqual(
        placement_unwrapped.argument[0].result.argument.compact_representation(
        ), '(x -> x)(_var1)')

  def test_unwrap_placement_removes_two_federated_maps(self):
    int_ref = building_blocks.Reference('x', tf.int32)
    int_id = building_blocks.Lambda('x', tf.int32, int_ref)
    fed_ref = building_blocks.Reference(
        'x',
        computation_types.FederatedType(tf.int32, placement_literals.CLIENTS))
    applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, fed_ref)
    second_applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, applied_id)
    third_applied_id = building_block_factory.create_federated_map_or_apply(
        int_id, second_applied_id)
    placement_unwrapped, modified = tree_transformations.unwrap_placement(
        third_applied_id)
    self.assertTrue(modified)

    self.assertEqual(
        _count_called_intrinsics(third_applied_id,
                                 intrinsic_defs.FEDERATED_MAP.uri), 3)
    self.assertEqual(
        _count_called_intrinsics(placement_unwrapped,
                                 intrinsic_defs.FEDERATED_MAP.uri), 1)

  def test_unwrap_removes_all_federated_zips_at_server(self):
    fed_tuple = building_blocks.Reference(
        'tup',
        computation_types.FederatedType([tf.int32, tf.float32] * 2,
                                        placement_literals.SERVER))
    unzipped = building_block_factory.create_federated_unzip(fed_tuple)
    zipped = building_block_factory.create_federated_zip(unzipped)
    placement_unwrapped, modified = tree_transformations.unwrap_placement(
        zipped)
    self.assertTrue(modified)

    self.assertIsInstance(zipped.type_signature,
                          computation_types.FederatedType)
    self.assertEqual(
        _count_called_intrinsics(zipped,
                                 intrinsic_defs.FEDERATED_ZIP_AT_SERVER.uri), 3)
    self.assertEqual(
        _count_called_intrinsics(placement_unwrapped,
                                 intrinsic_defs.FEDERATED_ZIP_AT_SERVER.uri), 0)

  def test_unwrap_removes_all_federated_zips_at_clients(self):
    fed_tuple = building_blocks.Reference(
        'tup',
        computation_types.FederatedType([tf.int32, tf.float32] * 2,
                                        placement_literals.CLIENTS))
    unzipped = building_block_factory.create_federated_unzip(fed_tuple)
    zipped = building_block_factory.create_federated_zip(unzipped)
    placement_unwrapped, modified = tree_transformations.unwrap_placement(
        zipped)
    self.assertTrue(modified)

    self.assertIsInstance(zipped.type_signature,
                          computation_types.FederatedType)
    self.assertEqual(
        _count_called_intrinsics(zipped,
                                 intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS.uri),
        3)
    self.assertEqual(
        _count_called_intrinsics(placement_unwrapped,
                                 intrinsic_defs.FEDERATED_ZIP_AT_CLIENTS.uri),
        0)

  def test_unwrap_placement_federated_value_at_server_removes_one_federated_value(
      self):
    int_data = building_blocks.Data('x', tf.int32)
    float_data = building_blocks.Data('x', tf.float32)
    fed_int = building_block_factory.create_federated_value(
        int_data, placement_literals.SERVER)
    fed_float = building_block_factory.create_federated_value(
        float_data, placement_literals.SERVER)
    tup = building_blocks.Struct([fed_int, fed_float])
    zipped = building_block_factory.create_federated_zip(tup)
    placement_unwrapped, modified = tree_transformations.unwrap_placement(
        zipped)
    self.assertTrue(modified)

    # This is destroying a py container but we probably want to fix it
    zipped.type_signature.check_equivalent_to(
        placement_unwrapped.type_signature)
    self.assertEqual(placement_unwrapped.function.uri,
                     intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri)
    self.assertEqual(
        _count_called_intrinsics(zipped,
                                 intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri),
        2)
    self.assertEqual(
        _count_called_intrinsics(placement_unwrapped,
                                 intrinsic_defs.FEDERATED_VALUE_AT_SERVER.uri),
        1)

  def test_unwrap_placement_federated_value_at_clients_removes_one_federated_value(
      self):
    int_data = building_blocks.Data('x', tf.int32)
    float_data = building_blocks.Data('x', tf.float32)
    fed_int = building_block_factory.create_federated_value(
        int_data, placement_literals.CLIENTS)
    fed_float = building_block_factory.create_federated_value(
        float_data, placement_literals.CLIENTS)
    tup = building_blocks.Struct([fed_int, fed_float])
    zipped = building_block_factory.create_federated_zip(tup)
    placement_unwrapped, modified = tree_transformations.unwrap_placement(
        zipped)
    self.assertTrue(modified)
    # These two types are no longer literally equal, since we have unwrapped the
    # `federated_value_at_clients` all the way to the top of the tree and
    # therefore have a value with `all_equal=True`; the zip above had destroyed
    # this information in a lossy way.
    self.assertTrue(
        zipped.type_signature.is_assignable_from(
            placement_unwrapped.type_signature))
    self.assertEqual(placement_unwrapped.function.uri,
                     intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri)
    self.assertEqual(
        _count_called_intrinsics(zipped,
                                 intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri),
        2)
    self.assertEqual(
        _count_called_intrinsics(placement_unwrapped,
                                 intrinsic_defs.FEDERATED_VALUE_AT_CLIENTS.uri),
        1)

  def test_unwrap_placement_with_lambda_inserts_federated_apply(self):
    federated_ref = building_blocks.Reference(
        'outer_ref',
        computation_types.FederatedType(tf.int32, placement_literals.SERVER))
    inner_federated_ref = building_blocks.Reference(
        'inner_ref',
        computation_types.FederatedType(tf.int32, placement_literals.SERVER))
    identity_lambda = building_blocks.Lambda('inner_ref',
                                             inner_federated_ref.type_signature,
                                             inner_federated_ref)
    called_lambda = building_blocks.Call(identity_lambda, federated_ref)
    unwrapped, modified = tree_transformations.unwrap_placement(called_lambda)
    self.assertTrue(modified)
    self.assertIsInstance(unwrapped.function, building_blocks.Intrinsic)
    self.assertEqual(unwrapped.function.uri, intrinsic_defs.FEDERATED_APPLY.uri)

  def test_unwrap_placement_with_lambda_produces_lambda_with_unplaced_type_signature(
      self):
    federated_ref = building_blocks.Reference(
        'outer_ref',
        computation_types.FederatedType(tf.int32, placement_literals.SERVER))
    inner_federated_ref = building_blocks.Reference(
        'inner_ref',
        computation_types.FederatedType(tf.int32, placement_literals.SERVER))
    identity_lambda = building_blocks.Lambda('inner_ref',
                                             inner_federated_ref.type_signature,
                                             inner_federated_ref)
    called_lambda = building_blocks.Call(identity_lambda, federated_ref)
    unwrapped, modified = tree_transformations.unwrap_placement(called_lambda)
    self.assertTrue(modified)
    self.assertEqual(unwrapped.argument[0].type_signature,
                     computation_types.FunctionType(tf.int32, tf.int32))


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
