# Lint as: python3
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
from six.moves import range
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_constructing_utils
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import tensorflow_serialization
from tensorflow_federated.python.core.impl import transformation_utils
from tensorflow_federated.python.core.impl import transformations
from tensorflow_federated.python.core.impl import type_utils


def _create_chained_calls(functions, arg):
  r"""Creates a chain of `n` calls.

       Call
      /    \
  Comp      ...
               \
                Call
               /    \
           Comp      Comp

  The first functional computation in `functions` must have a parameter type
  that is assignable from the type of `arg`, each other functional computation
  in `functions` must have a parameter type that is assignable from the previous
  functional computations result type.

  Args:
    functions: A Python list of functional computations.
    arg: A `computation_building_blocks.ComputationBuildingBlock`.

  Returns:
    A `computation_building_blocks.Call`.
  """
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  for fn in functions:
    py_typecheck.check_type(
        fn, computation_building_blocks.ComputationBuildingBlock)
    if not type_utils.is_assignable_from(fn.parameter_type, arg.type_signature):
      raise TypeError(
          'The parameter of the function is of type {}, and the argument is of '
          'an incompatible type {}.'.format(
              str(fn.parameter_type), str(arg.type_signature)))
    call = computation_building_blocks.Call(fn, arg)
    arg = call
  return call


def _create_chained_dummy_federated_applys(functions, arg):
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  for fn in functions:
    py_typecheck.check_type(
        fn, computation_building_blocks.ComputationBuildingBlock)
    if not type_utils.is_assignable_from(fn.parameter_type,
                                         arg.type_signature.member):
      raise TypeError(
          'The parameter of the function is of type {}, and the argument is of '
          'an incompatible type {}.'.format(
              str(fn.parameter_type), str(arg.type_signature.member)))
    call = computation_constructing_utils.create_federated_apply(fn, arg)
    arg = call
  return call


def _create_chained_dummy_federated_maps(functions, arg):
  py_typecheck.check_type(arg,
                          computation_building_blocks.ComputationBuildingBlock)
  for fn in functions:
    py_typecheck.check_type(
        fn, computation_building_blocks.ComputationBuildingBlock)
    if not type_utils.is_assignable_from(fn.parameter_type,
                                         arg.type_signature.member):
      raise TypeError(
          'The parameter of the function is of type {}, and the argument is of '
          'an incompatible type {}.'.format(
              str(fn.parameter_type), str(arg.type_signature.member)))
    call = computation_constructing_utils.create_federated_map(fn, arg)
    arg = call
  return call


def _create_lambda_to_identity(parameter_name, parameter_type=tf.int32):
  r"""Creates a lambda to return the argument.

  Lambda(x)
           \
            Ref(x)

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.

  Returns:
    A `computation_building_blocks.Lambda`.
  """
  ref = computation_building_blocks.Reference(parameter_name, parameter_type)
  return computation_building_blocks.Lambda(ref.name, ref.type_signature, ref)


def _create_lambda_to_dummy_intrinsic(parameter_name, parameter_type=tf.int32):
  r"""Creates a lambda to call a dummy intrinsic.

  Lambda(x)
           \
            Call
           /    \
  Intrinsic      Ref(x)

  Args:
    parameter_name: The name of the parameter.
    parameter_type: The type of the parameter.

  Returns:
    A `computation_building_blocks.Lambda`.
  """
  py_typecheck.check_type(parameter_type, tf.dtypes.DType)
  call = _create_dummy_called_intrinsic(
      parameter_name=parameter_name, parameter_type=parameter_type)
  return computation_building_blocks.Lambda(parameter_name, parameter_type,
                                            call)


def _create_lambda_to_dummy_cast(parameter_name, parameter_type, result_type):
  py_typecheck.check_type(parameter_type, tf.dtypes.DType)
  py_typecheck.check_type(result_type, tf.dtypes.DType)
  arg = computation_building_blocks.Data('data', result_type)
  return computation_building_blocks.Lambda(parameter_name, parameter_type, arg)


def _create_dummy_block(comp, variable_name, variable_type=tf.int32):
  py_typecheck.check_type(comp,
                          computation_building_blocks.ComputationBuildingBlock)
  data = computation_building_blocks.Data('data', variable_type)
  return computation_building_blocks.Block([(variable_name, data)], comp)


def _create_dummy_called_federated_aggregate(accumulate_parameter_name,
                                             merge_parameter_name,
                                             report_parameter_name):
  value_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
  value = computation_building_blocks.Data('data', value_type)
  zero = computation_building_blocks.Data('data', tf.int32)
  accumulate_type = computation_types.NamedTupleType((tf.int32, tf.int32))
  accumulate_result = computation_building_blocks.Data('data', tf.int32)
  accumulate = computation_building_blocks.Lambda(accumulate_parameter_name,
                                                  accumulate_type,
                                                  accumulate_result)
  merge_type = computation_types.NamedTupleType((tf.int32, tf.int32))
  merge_result = computation_building_blocks.Data('data', tf.int32)
  merge = computation_building_blocks.Lambda(merge_parameter_name, merge_type,
                                             merge_result)
  report_ref = computation_building_blocks.Reference(report_parameter_name,
                                                     tf.int32)
  report = computation_building_blocks.Lambda(report_ref.name,
                                              report_ref.type_signature,
                                              report_ref)
  return computation_constructing_utils.create_federated_aggregate(
      value, zero, accumulate, merge, report)


def _create_dummy_called_federated_apply(parameter_name,
                                         parameter_type=tf.int32):
  fn = _create_lambda_to_identity(parameter_name, parameter_type)
  arg_type = computation_types.FederatedType(parameter_type, placements.SERVER)
  arg = computation_building_blocks.Data('data', arg_type)
  return computation_constructing_utils.create_federated_apply(fn, arg)


def _create_dummy_called_federated_map(parameter_name, parameter_type=tf.int32):
  fn = _create_lambda_to_identity(parameter_name, parameter_type)
  arg_type = computation_types.FederatedType(parameter_type, placements.CLIENTS)
  arg = computation_building_blocks.Data('data', arg_type)
  return computation_constructing_utils.create_federated_map(fn, arg)


def _create_dummy_called_sequence_map(parameter_name, parameter_type=tf.int32):
  fn = _create_lambda_to_identity(parameter_name, parameter_type)
  arg_type = computation_types.SequenceType(parameter_type)
  arg = computation_building_blocks.Data('data', arg_type)
  return computation_constructing_utils.create_sequence_map(fn, arg)


def _create_dummy_called_intrinsic(parameter_name, parameter_type=tf.int32):
  intrinsic_type = computation_types.FunctionType(parameter_type,
                                                  parameter_type)
  intrinsic = computation_building_blocks.Intrinsic('intrinsic', intrinsic_type)
  ref = computation_building_blocks.Reference(parameter_name, parameter_type)
  return computation_building_blocks.Call(intrinsic, ref)


def _create_block_wrapping_data(variable_name, variable_type=tf.int32):
  data = computation_building_blocks.Data('data', variable_type)
  ref = computation_building_blocks.Reference(variable_name, variable_type)
  return computation_building_blocks.Block([(variable_name, data)], ref)


class ExtractIntrinsicsTest(absltest.TestCase):

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      transformations.extract_intrinsics(None)

  def test_raises_value_error_with_non_unique_variable_names(self):
    data = computation_building_blocks.Data('data', tf.int32)
    block = computation_building_blocks.Block([('a', data), ('a', data)], data)
    with self.assertRaises(ValueError):
      transformations.extract_intrinsics(block)

  def test_extracts_from_block_result_intrinsic(self):
    data = computation_building_blocks.Data('data', tf.int32)
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    block = computation_building_blocks.Block((('a', data),), called_intrinsic)
    comp = block

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr, '(let a=data in intrinsic(a))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let a=data,_var1=intrinsic(a) in _var1)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_block_result_block_one_var_unbound(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    ref = computation_building_blocks.Reference('b',
                                                called_intrinsic.type_signature)
    block_1 = computation_building_blocks.Block((('b', called_intrinsic),), ref)
    data = computation_building_blocks.Data('data', tf.int32)
    block_2 = computation_building_blocks.Block((('a', data),), block_1)
    comp = block_2

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr, '(let a=data in (let b=intrinsic(a) in b))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let a=data,b=intrinsic(a) in b)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_block_result_block_multiple_vars_unbound(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    ref = computation_building_blocks.Reference('b',
                                                called_intrinsic.type_signature)
    block_1 = computation_building_blocks.Block((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ), ref)
    data = computation_building_blocks.Data('data', tf.int32)
    block_2 = computation_building_blocks.Block((('a', data),), block_1)
    comp = block_2

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(
        comp.tff_repr,
        '(let a=data in (let b=intrinsic(a),c=intrinsic(a) in b))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let a=data,b=intrinsic(a),c=intrinsic(a) in b)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_block_variables_block_one_var_unbound(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    ref_1 = computation_building_blocks.Reference(
        'b', called_intrinsic.type_signature)
    block_1 = computation_building_blocks.Block((('b', called_intrinsic),),
                                                ref_1)
    ref_2 = computation_building_blocks.Reference('c', tf.int32)
    block_2 = computation_building_blocks.Block((('c', block_1),), ref_2)
    comp = block_2

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr, '(let c=(let b=intrinsic(a) in b) in c)')
    self.assertEqual(transformed_comp.tff_repr, '(let b=intrinsic(a),c=b in c)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_block_variables_block_multiple_vars_unbound(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    ref_1 = computation_building_blocks.Reference(
        'b', called_intrinsic.type_signature)
    block_1 = computation_building_blocks.Block((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ), ref_1)
    ref_2 = computation_building_blocks.Reference('d', tf.int32)
    block_2 = computation_building_blocks.Block((('d', block_1),), ref_2)
    comp = block_2

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr,
                     '(let d=(let b=intrinsic(a),c=intrinsic(a) in b) in d)')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let b=intrinsic(a),c=intrinsic(a),d=b in d)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_block_variables_block_one_var_bound_by_lambda(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    ref_1 = computation_building_blocks.Reference(
        'b', called_intrinsic.type_signature)
    block_1 = computation_building_blocks.Block((('b', called_intrinsic),),
                                                ref_1)
    ref_2 = computation_building_blocks.Reference('c', tf.int32)
    block_2 = computation_building_blocks.Block((('c', block_1),), ref_2)
    fn = computation_building_blocks.Lambda('a', tf.int32, block_2)
    comp = fn

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr,
                     '(a -> (let c=(let b=intrinsic(a) in b) in c))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(a -> (let b=intrinsic(a),c=b in c))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_block_variables_block_multiple_vars_bound_by_lambda(
      self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    ref_1 = computation_building_blocks.Reference(
        'b', called_intrinsic.type_signature)
    block_1 = computation_building_blocks.Block((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ), ref_1)
    ref_2 = computation_building_blocks.Reference('d', tf.int32)
    block_2 = computation_building_blocks.Block((('d', block_1),), ref_2)
    fn = computation_building_blocks.Lambda('a', tf.int32, block_2)
    comp = fn

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(
        comp.tff_repr,
        '(a -> (let d=(let b=intrinsic(a),c=intrinsic(a) in b) in d))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(a -> (let b=intrinsic(a),c=intrinsic(a),d=b in d))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_block_variables_block_one_var_bound_by_block(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    ref_1 = computation_building_blocks.Reference(
        'b', called_intrinsic.type_signature)
    block_1 = computation_building_blocks.Block((('b', called_intrinsic),),
                                                ref_1)
    data = computation_building_blocks.Data('data', tf.int32)
    ref_2 = computation_building_blocks.Reference('c', tf.int32)
    block_2 = computation_building_blocks.Block((
        ('a', data),
        ('c', block_1),
    ), ref_2)
    comp = block_2

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr,
                     '(let a=data,c=(let b=intrinsic(a) in b) in c)')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let a=data,b=intrinsic(a),c=b in c)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_block_variables_block_multiple_vars_bound_by_block(
      self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    ref_1 = computation_building_blocks.Reference(
        'b', called_intrinsic.type_signature)
    block_1 = computation_building_blocks.Block((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ), ref_1)
    data = computation_building_blocks.Data('data', tf.int32)
    ref_2 = computation_building_blocks.Reference('d', tf.int32)
    block_2 = computation_building_blocks.Block((
        ('a', data),
        ('d', block_1),
    ), ref_2)
    comp = block_2

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(
        comp.tff_repr,
        '(let a=data,d=(let b=intrinsic(a),c=intrinsic(a) in b) in d)')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let a=data,b=intrinsic(a),c=intrinsic(a),d=b in d)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_call_intrinsic(self):
    fn = _create_lambda_to_identity('a', tf.int32)
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='b')
    call = computation_building_blocks.Call(fn, called_intrinsic)
    comp = call

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr, '(a -> a)(intrinsic(b))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let _var1=intrinsic(b) in (a -> a)(_var1))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_call_block_one_var(self):
    fn = _create_lambda_to_identity('a', tf.int32)
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='b')
    ref = computation_building_blocks.Reference('c',
                                                called_intrinsic.type_signature)
    block = computation_building_blocks.Block((('c', called_intrinsic),), ref)
    call = computation_building_blocks.Call(fn, block)
    comp = call

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr, '(a -> a)((let c=intrinsic(b) in c))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let c=intrinsic(b) in (a -> a)(c))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_call_block_multiple_vars(self):
    fn = _create_lambda_to_identity('a', tf.int32)
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='b')
    ref = computation_building_blocks.Reference('c',
                                                called_intrinsic.type_signature)
    block = computation_building_blocks.Block((
        ('c', called_intrinsic),
        ('d', called_intrinsic),
    ), ref)
    call = computation_building_blocks.Call(fn, block)
    comp = call

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr,
                     '(a -> a)((let c=intrinsic(b),d=intrinsic(b) in c))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let c=intrinsic(b),d=intrinsic(b) in (a -> a)(c))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_lambda_intrinsic_unbound(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    fn = computation_building_blocks.Lambda('b', tf.int32, called_intrinsic)
    comp = fn

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr, '(b -> intrinsic(a))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let _var1=intrinsic(a) in (b -> _var1))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_lambda_intrinsic_bound_by_lambda(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    fn = computation_building_blocks.Lambda('a', tf.int32, called_intrinsic)
    comp = fn

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr, '(a -> intrinsic(a))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(a -> (let _var1=intrinsic(a) in _var1))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_lambda_block_one_var_unbound(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    ref = computation_building_blocks.Reference('b',
                                                called_intrinsic.type_signature)
    block = computation_building_blocks.Block((('b', called_intrinsic),), ref)
    fn = computation_building_blocks.Lambda('c', tf.int32, block)
    comp = fn

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr, '(c -> (let b=intrinsic(a) in b))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let b=intrinsic(a) in (c -> b))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_lambda_block_multiple_vars_unbound(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    ref = computation_building_blocks.Reference('b',
                                                called_intrinsic.type_signature)
    block = computation_building_blocks.Block((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ), ref)
    fn = computation_building_blocks.Lambda('d', tf.int32, block)
    comp = fn

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr,
                     '(d -> (let b=intrinsic(a),c=intrinsic(a) in b))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let b=intrinsic(a),c=intrinsic(a) in (d -> b))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_lambda_block_first_var_unbound(self):
    called_intrinsic_1 = _create_dummy_called_intrinsic(parameter_name='a')
    called_intrinsic_2 = _create_dummy_called_intrinsic(parameter_name='b')
    ref = computation_building_blocks.Reference(
        'c', called_intrinsic_2.type_signature)
    block = computation_building_blocks.Block((
        ('c', called_intrinsic_1),
        ('d', called_intrinsic_2),
    ), ref)
    fn = computation_building_blocks.Lambda('b', tf.int32, block)
    comp = fn

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr,
                     '(b -> (let c=intrinsic(a),d=intrinsic(b) in c))')
    self.assertEqual(
        transformed_comp.tff_repr,
        '(let c=intrinsic(a) in (b -> (let d=intrinsic(b) in c)))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_lambda_block_last_var_unbound(self):
    called_intrinsic_1 = _create_dummy_called_intrinsic(parameter_name='a')
    called_intrinsic_2 = _create_dummy_called_intrinsic(parameter_name='b')
    ref = computation_building_blocks.Reference(
        'c', called_intrinsic_2.type_signature)
    block = computation_building_blocks.Block((
        ('c', called_intrinsic_1),
        ('d', called_intrinsic_2),
    ), ref)
    fn = computation_building_blocks.Lambda('a', tf.int32, block)
    comp = fn

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr,
                     '(a -> (let c=intrinsic(a),d=intrinsic(b) in c))')
    self.assertEqual(
        transformed_comp.tff_repr,
        '(let d=intrinsic(b) in (a -> (let c=intrinsic(a) in c)))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_lambda_block_one_var_bound_by_block(self):
    data = computation_building_blocks.Data('data', tf.int32)
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    ref = computation_building_blocks.Reference('b',
                                                called_intrinsic.type_signature)
    block = computation_building_blocks.Block((
        ('a', data),
        ('b', called_intrinsic),
    ), ref)
    fn = computation_building_blocks.Lambda('c', tf.int32, block)
    comp = fn

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr, '(c -> (let a=data,b=intrinsic(a) in b))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let a=data,b=intrinsic(a) in (c -> b))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_lambda_block_multiple_vars_bound_by_block(self):
    data = computation_building_blocks.Data('data', tf.int32)
    called_intrinsic_1 = _create_dummy_called_intrinsic(parameter_name='a')
    called_intrinsic_2 = _create_dummy_called_intrinsic(parameter_name='b')
    ref = computation_building_blocks.Reference(
        'c', called_intrinsic_2.type_signature)
    block = computation_building_blocks.Block((
        ('a', data),
        ('b', called_intrinsic_1),
        ('c', called_intrinsic_2),
    ), ref)
    fn = computation_building_blocks.Lambda('d', tf.int32, block)
    comp = fn

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr,
                     '(d -> (let a=data,b=intrinsic(a),c=intrinsic(b) in c))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let a=data,b=intrinsic(a),c=intrinsic(b) in (d -> c))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_selection_intrinsic(self):
    parameter_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    called_intrinsic = _create_dummy_called_intrinsic(
        parameter_name='a', parameter_type=parameter_type)
    sel = computation_building_blocks.Selection(called_intrinsic, index=0)
    comp = sel

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr, 'intrinsic(a)[0]')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let _var1=intrinsic(a) in _var1[0])')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_selection_named_intrinsic(self):
    parameter_type = computation_types.NamedTupleType((
        ('a', tf.int32),
        ('b', tf.int32),
    ))
    called_intrinsic = _create_dummy_called_intrinsic(
        parameter_name='c', parameter_type=parameter_type)
    sel = computation_building_blocks.Selection(called_intrinsic, index=0)
    comp = sel

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr, 'intrinsic(c)[0]')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let _var1=intrinsic(c) in _var1[0])')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_selection_block_one_var(self):
    parameter_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    called_intrinsic = _create_dummy_called_intrinsic(
        parameter_name='a', parameter_type=parameter_type)
    ref = computation_building_blocks.Reference('b',
                                                called_intrinsic.type_signature)
    block = computation_building_blocks.Block((('b', called_intrinsic),), ref)
    sel = computation_building_blocks.Selection(block, index=0)
    comp = sel

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr, '(let b=intrinsic(a) in b)[0]')
    self.assertEqual(transformed_comp.tff_repr, '(let b=intrinsic(a) in b[0])')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_selection_block_multiple_vars(self):
    parameter_type = computation_types.NamedTupleType((tf.int32, tf.int32))
    called_intrinsic = _create_dummy_called_intrinsic(
        parameter_name='a', parameter_type=parameter_type)
    ref = computation_building_blocks.Reference('b',
                                                called_intrinsic.type_signature)
    block = computation_building_blocks.Block((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ), ref)
    sel = computation_building_blocks.Selection(block, index=0)
    comp = sel

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr,
                     '(let b=intrinsic(a),c=intrinsic(a) in b)[0]')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let b=intrinsic(a),c=intrinsic(a) in b[0])')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_tuple_one_intrinsic(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    tup = computation_building_blocks.Tuple((called_intrinsic,))
    comp = tup

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr, '<intrinsic(a)>')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let _var1=intrinsic(a) in <_var1>)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_tuple_multiple_intrinsics(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    tup = computation_building_blocks.Tuple(
        (called_intrinsic, called_intrinsic))
    comp = tup

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr, '<intrinsic(a),intrinsic(a)>')
    self.assertEqual(
        transformed_comp.tff_repr,
        '(let _var1=intrinsic(a),_var2=intrinsic(a) in <_var1,_var2>)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_tuple_named_intrinsics(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    tup = computation_building_blocks.Tuple((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ))
    comp = tup

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr, '<b=intrinsic(a),c=intrinsic(a)>')
    self.assertEqual(
        transformed_comp.tff_repr,
        '(let _var1=intrinsic(a),_var2=intrinsic(a) in <b=_var1,c=_var2>)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_tuple_one_block_one_var(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    ref = computation_building_blocks.Reference('b',
                                                called_intrinsic.type_signature)
    block = computation_building_blocks.Block((('b', called_intrinsic),), ref)
    tup = computation_building_blocks.Tuple((block,))
    comp = tup

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr, '<(let b=intrinsic(a) in b)>')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let b=intrinsic(a),_var1=b in <_var1>)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_tuple_one_block_multiple_vars(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    ref = computation_building_blocks.Reference('b',
                                                called_intrinsic.type_signature)
    block = computation_building_blocks.Block((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ), ref)
    tup = computation_building_blocks.Tuple((block,))
    comp = tup

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr,
                     '<(let b=intrinsic(a),c=intrinsic(a) in b)>')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let b=intrinsic(a),c=intrinsic(a),_var1=b in <_var1>)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_tuple_multiple_blocks_one_var(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    ref_1 = computation_building_blocks.Reference(
        'b', called_intrinsic.type_signature)
    block_1 = computation_building_blocks.Block((('b', called_intrinsic),),
                                                ref_1)
    ref_2 = computation_building_blocks.Reference(
        'd', called_intrinsic.type_signature)
    block_2 = computation_building_blocks.Block((('d', called_intrinsic),),
                                                ref_2)
    tup = computation_building_blocks.Tuple((block_1, block_2))
    comp = tup

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr,
                     '<(let b=intrinsic(a) in b),(let d=intrinsic(a) in d)>')
    self.assertEqual(
        transformed_comp.tff_repr,
        '(let b=intrinsic(a),_var1=b,d=intrinsic(a),_var2=d in <_var1,_var2>)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_from_tuple_multiple_blocks_multiple_vars(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    ref_1 = computation_building_blocks.Reference(
        'b', called_intrinsic.type_signature)
    block_1 = computation_building_blocks.Block((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ), ref_1)
    ref_2 = computation_building_blocks.Reference(
        'd', called_intrinsic.type_signature)
    block_2 = computation_building_blocks.Block((
        ('d', called_intrinsic),
        ('e', called_intrinsic),
    ), ref_2)
    tup = computation_building_blocks.Tuple((block_1, block_2))
    comp = tup

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(
        comp.tff_repr,
        '<(let b=intrinsic(a),c=intrinsic(a) in b),(let d=intrinsic(a),e=intrinsic(a) in d)>'
    )
    self.assertEqual(
        transformed_comp.tff_repr,
        '(let b=intrinsic(a),c=intrinsic(a),_var1=b,d=intrinsic(a),e=intrinsic(a),_var2=d in <_var1,_var2>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_one_intrinsic(self):
    data = computation_building_blocks.Data('data', tf.int32)
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    tup = computation_building_blocks.Tuple((called_intrinsic,))
    sel = computation_building_blocks.Selection(tup, index=0)
    block = computation_building_blocks.Block((('b', data),), sel)
    fn_1 = _create_lambda_to_identity('c', tf.int32)
    call_1 = computation_building_blocks.Call(fn_1, block)
    fn_2 = _create_lambda_to_identity('d', tf.int32)
    call_2 = computation_building_blocks.Call(fn_2, call_1)
    fn_3 = computation_building_blocks.Lambda('e', tf.int32, call_2)
    comp = fn_3

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(
        comp.tff_repr,
        '(e -> (d -> d)((c -> c)((let b=data in <intrinsic(a)>[0]))))')
    self.assertEqual(
        transformed_comp.tff_repr,
        '(let b=data,_var1=intrinsic(a) in (e -> (d -> d)((c -> c)(<_var1>[0]))))'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_multiple_intrinsics(self):
    data = computation_building_blocks.Data('data', tf.int32)
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    tup = computation_building_blocks.Tuple(
        (called_intrinsic, called_intrinsic))
    sel = computation_building_blocks.Selection(tup, index=0)
    block = computation_building_blocks.Block((
        ('b', data),
        ('c', called_intrinsic),
    ), sel)
    fn_1 = _create_lambda_to_identity('d', tf.int32)
    call_1 = computation_building_blocks.Call(fn_1, block)
    fn_2 = _create_lambda_to_identity('e', tf.int32)
    call_2 = computation_building_blocks.Call(fn_2, call_1)
    fn_3 = computation_building_blocks.Lambda('f', tf.int32, call_2)
    comp = fn_3

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(
        comp.tff_repr,
        '(f -> (e -> e)((d -> d)((let b=data,c=intrinsic(a) in <intrinsic(a),intrinsic(a)>[0]))))'
    )
    self.assertEqual(
        transformed_comp.tff_repr,
        '(let b=data,c=intrinsic(a),_var1=intrinsic(a),_var2=intrinsic(a) in (f -> (e -> e)((d -> d)(<_var1,_var2>[0]))))'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_extracts_multiple_intrinsics_dependent_bindings(self):
    called_intrinsic_1 = _create_dummy_called_intrinsic(parameter_name='a')
    fn_1 = computation_building_blocks.Lambda('a', tf.int32, called_intrinsic_1)
    data = computation_building_blocks.Data('data', tf.int32)
    call_1 = computation_building_blocks.Call(fn_1, data)
    intrinsic_type = computation_types.FunctionType(tf.int32, tf.int32)
    intrinsic = computation_building_blocks.Intrinsic('intrinsic',
                                                      intrinsic_type)
    called_intrinsic_2 = computation_building_blocks.Call(intrinsic, call_1)
    fn_2 = computation_building_blocks.Lambda('b', tf.int32, called_intrinsic_2)
    comp = fn_2

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(comp.tff_repr,
                     '(b -> intrinsic((a -> intrinsic(a))(data)))')
    self.assertEqual(
        transformed_comp.tff_repr,
        '(let _var2=intrinsic((a -> (let _var1=intrinsic(a) in _var1))(data)) in (b -> _var2))'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_does_not_extract_from_block_variables_intrinsic(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    ref = computation_building_blocks.Reference('b',
                                                called_intrinsic.type_signature)
    block = computation_building_blocks.Block((('b', called_intrinsic),), ref)
    comp = block

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(transformed_comp.tff_repr, '(let b=intrinsic(a) in b)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_does_not_extract_from_lambda_block_one_var_bound_by_lambda(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    ref = computation_building_blocks.Reference('b',
                                                called_intrinsic.type_signature)
    block = computation_building_blocks.Block((('b', called_intrinsic),), ref)
    fn = computation_building_blocks.Lambda('a', tf.int32, block)
    comp = fn

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(transformed_comp.tff_repr,
                     '(a -> (let b=intrinsic(a) in b))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_does_not_extract_from_lambda_block_multiple_vars_bound_by_lambda(
      self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    ref = computation_building_blocks.Reference('b',
                                                called_intrinsic.type_signature)
    block = computation_building_blocks.Block((
        ('b', called_intrinsic),
        ('c', called_intrinsic),
    ), ref)
    fn = computation_building_blocks.Lambda('a', tf.int32, block)
    comp = fn

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(transformed_comp.tff_repr,
                     '(a -> (let b=intrinsic(a),c=intrinsic(a) in b))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_does_not_extract_called_lambda(self):
    fn = _create_lambda_to_identity('a', tf.int32)
    arg = computation_building_blocks.Data('data', tf.int32)
    call = computation_building_blocks.Call(fn, arg)
    comp = call

    transformed_comp, modified = transformations.extract_intrinsics(comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(transformed_comp.tff_repr, '(a -> a)(data)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)


class InlineBlockLocalsTest(absltest.TestCase):

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      transformations.inline_block_locals(None)

  def test_raises_value_error_with_non_unique_variable_names(self):
    data = computation_building_blocks.Data('data', tf.int32)
    block = computation_building_blocks.Block([('a', data), ('a', data)], data)
    with self.assertRaises(ValueError):
      transformations.inline_block_locals(block)

  def test_inlines_one_variable(self):
    block = _create_block_wrapping_data(variable_name='a')
    comp = block

    transformed_comp, modified = transformations.inline_block_locals(comp)

    self.assertEqual(comp.tff_repr, '(let a=data in a)')
    self.assertEqual(transformed_comp.tff_repr, 'data')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_inlines_two_variables(self):
    data = computation_building_blocks.Data('data', tf.int32)
    ref = computation_building_blocks.Reference('a', tf.int32)
    tup = computation_building_blocks.Tuple((ref, ref))
    block = computation_building_blocks.Block((('a', data),), tup)
    comp = block

    transformed_comp, modified = transformations.inline_block_locals(comp)

    self.assertEqual(comp.tff_repr, '(let a=data in <a,a>)')
    self.assertEqual(transformed_comp.tff_repr, '<data,data>')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_inlines_variables_in_block_variables(self):
    block_1 = _create_block_wrapping_data(variable_name='a')
    ref = computation_building_blocks.Reference('b', block_1.type_signature)
    block_2 = computation_building_blocks.Block((('b', block_1),), ref)
    comp = block_2

    transformed_comp, modified = transformations.inline_block_locals(comp)

    self.assertEqual(comp.tff_repr, '(let b=(let a=data in a) in b)')
    self.assertEqual(transformed_comp.tff_repr, 'data')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_inlines_variables_in_block_results(self):
    ref_1 = computation_building_blocks.Reference('a', tf.int32)
    data = computation_building_blocks.Data('data', tf.int32)
    ref_2 = computation_building_blocks.Reference('b', tf.int32)
    block_1 = computation_building_blocks.Block([('b', ref_1)], ref_2)
    block_2 = computation_building_blocks.Block([('a', data)], block_1)
    comp = block_2

    transformed_comp, modified = transformations.inline_block_locals(comp)

    self.assertEqual(comp.tff_repr, '(let a=data in (let b=a in b))')
    self.assertEqual(transformed_comp.tff_repr, 'data')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_inlines_variables_bound_sequentially(self):
    data = computation_building_blocks.Data('data', tf.int32)
    ref_1 = computation_building_blocks.Reference('a', tf.int32)
    ref_2 = computation_building_blocks.Reference('b', tf.int32)
    ref_3 = computation_building_blocks.Reference('c', tf.int32)
    block = computation_building_blocks.Block(
        (('b', data), ('c', ref_2), ('a', ref_3)), ref_1)
    comp = block

    transformed_comp, modified = transformations.inline_block_locals(comp)

    self.assertEqual(comp.tff_repr, '(let b=data,c=b,a=c in a)')
    self.assertEqual(transformed_comp.tff_repr, 'data')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_does_not_inline_lambda(self):
    fn = _create_lambda_to_identity('a', tf.int32)
    comp = fn

    transformed_comp, modified = transformations.inline_block_locals(comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(transformed_comp.tff_repr, '(a -> a)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)


class MergeChainedBlocksTest(absltest.TestCase):

  def test_fails_on_none(self):
    with self.assertRaises(TypeError):
      transformations.merge_chained_blocks(None)

  def test_single_level_of_nesting(self):
    input1 = computation_building_blocks.Reference('input1', tf.int32)
    result = computation_building_blocks.Reference('result', tf.int32)
    block1 = computation_building_blocks.Block([('result', input1)], result)
    input2 = computation_building_blocks.Data('input2', tf.int32)
    block2 = computation_building_blocks.Block([('input1', input2)], block1)
    self.assertEqual(block2.tff_repr,
                     '(let input1=input2 in (let result=input1 in result))')
    merged_blocks, modified = transformations.merge_chained_blocks(block2)
    self.assertEqual(merged_blocks.tff_repr,
                     '(let input1=input2,result=input1 in result)')
    self.assertTrue(modified)

  def test_leaves_names(self):
    input1 = computation_building_blocks.Data('input1', tf.int32)
    result_tuple = computation_building_blocks.Tuple([
        ('a', computation_building_blocks.Data('result_a', tf.int32)),
        ('b', computation_building_blocks.Data('result_b', tf.int32))
    ])
    block1 = computation_building_blocks.Block([('x', input1)], result_tuple)
    result_block = block1
    input2 = computation_building_blocks.Data('input2', tf.int32)
    block2 = computation_building_blocks.Block([('y', input2)], result_block)
    self.assertEqual(
        block2.tff_repr,
        '(let y=input2 in (let x=input1 in <a=result_a,b=result_b>))')
    merged, modified = transformations.merge_chained_blocks(block2)
    self.assertEqual(merged.tff_repr,
                     '(let y=input2,x=input1 in <a=result_a,b=result_b>)')
    self.assertTrue(modified)

  def test_leaves_separated_chained_blocks_alone(self):
    input1 = computation_building_blocks.Data('input1', tf.int32)
    result = computation_building_blocks.Data('result', tf.int32)
    block1 = computation_building_blocks.Block([('x', input1)], result)
    result_block = block1
    result_tuple = computation_building_blocks.Tuple([result_block])
    input2 = computation_building_blocks.Data('input2', tf.int32)
    block2 = computation_building_blocks.Block([('y', input2)], result_tuple)
    self.assertEqual(block2.tff_repr,
                     '(let y=input2 in <(let x=input1 in result)>)')
    merged, modified = transformations.merge_chained_blocks(block2)
    self.assertEqual(merged.tff_repr,
                     '(let y=input2 in <(let x=input1 in result)>)')
    self.assertFalse(modified)

  def test_two_levels_of_nesting(self):
    input1 = computation_building_blocks.Reference('input1', tf.int32)
    result = computation_building_blocks.Reference('result', tf.int32)
    block1 = computation_building_blocks.Block([('result', input1)], result)
    input2 = computation_building_blocks.Reference('input2', tf.int32)
    block2 = computation_building_blocks.Block([('input1', input2)], block1)
    input3 = computation_building_blocks.Data('input3', tf.int32)
    block3 = computation_building_blocks.Block([('input2', input3)], block2)
    self.assertEqual(
        block3.tff_repr,
        '(let input2=input3 in (let input1=input2 in (let result=input1 in result)))'
    )
    merged_blocks, modified = transformations.merge_chained_blocks(block3)
    self.assertEqual(
        merged_blocks.tff_repr,
        '(let input2=input3,input1=input2,result=input1 in result)')
    self.assertTrue(modified)


class MergeChainedFederatedMapOrApplysTest(parameterized.TestCase):

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      transformations.merge_chained_federated_maps_or_applys(None)

  def test_merges_federated_applys(self):
    fn = _create_lambda_to_identity('a', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.SERVER)
    arg = computation_building_blocks.Data('data', arg_type)
    call = _create_chained_dummy_federated_applys([fn, fn], arg)
    comp = call

    transformed_comp, modified = transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(
        comp.tff_repr,
        'federated_apply(<(a -> a),federated_apply(<(a -> a),data>)>)')
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_apply(<(let fn=<(a -> a),(a -> a)> in (arg -> fn[1](fn[0](arg)))),data>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), 'int32@SERVER')
    self.assertTrue(modified)

  def test_merges_federated_maps(self):
    fn = _create_lambda_to_identity('a', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = computation_building_blocks.Data('data', arg_type)
    call = _create_chained_dummy_federated_maps([fn, fn], arg)
    comp = call

    transformed_comp, modified = transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(
        comp.tff_repr,
        'federated_map(<(a -> a),federated_map(<(a -> a),data>)>)')
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(let fn=<(a -> a),(a -> a)> in (arg -> fn[1](fn[0](arg)))),data>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), '{int32}@CLIENTS')
    self.assertTrue(modified)

  def test_merges_federated_maps_with_different_names(self):
    fn_1 = _create_lambda_to_identity('a', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = computation_building_blocks.Data('data', arg_type)
    fn_2 = _create_lambda_to_identity('b', tf.int32)
    call = _create_chained_dummy_federated_maps([fn_1, fn_2], arg)
    comp = call

    transformed_comp, modified = transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(
        comp.tff_repr,
        'federated_map(<(b -> b),federated_map(<(a -> a),data>)>)')
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(let fn=<(a -> a),(b -> b)> in (arg -> fn[1](fn[0](arg)))),data>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), '{int32}@CLIENTS')
    self.assertTrue(modified)

  def test_merges_federated_maps_with_different_types(self):
    fn_1 = _create_lambda_to_dummy_cast('a', tf.int32, tf.float32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = computation_building_blocks.Data('data', arg_type)
    fn_2 = _create_lambda_to_identity('b', tf.float32)
    call = _create_chained_dummy_federated_maps([fn_1, fn_2], arg)
    comp = call

    transformed_comp, modified = transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(
        comp.tff_repr,
        'federated_map(<(b -> b),federated_map(<(a -> data),data>)>)')
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(let fn=<(a -> data),(b -> b)> in (arg -> fn[1](fn[0](arg)))),data>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), '{float32}@CLIENTS')
    self.assertTrue(modified)

  def test_merges_federated_maps_with_named_parameter_type(self):
    parameter_type = [('b', tf.int32), ('c', tf.int32)]
    fn = _create_lambda_to_identity('a', parameter_type)
    arg_type = computation_types.FederatedType(parameter_type,
                                               placements.CLIENTS)
    arg = computation_building_blocks.Data('data', arg_type)
    call = _create_chained_dummy_federated_maps([fn, fn], arg)
    comp = call

    transformed_comp, modified = transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(
        comp.tff_repr,
        'federated_map(<(a -> a),federated_map(<(a -> a),data>)>)')
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(let fn=<(a -> a),(a -> a)> in (arg -> fn[1](fn[0](arg)))),data>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature), '{<b=int32,c=int32>}@CLIENTS')
    self.assertTrue(modified)

  def test_merges_federated_maps_with_unbound_references(self):
    ref = computation_building_blocks.Reference('a', tf.int32)
    fn = computation_building_blocks.Lambda('b', tf.int32, ref)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = computation_building_blocks.Data('data', arg_type)
    call = _create_chained_dummy_federated_maps([fn, fn], arg)
    comp = call

    transformed_comp, modified = transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(
        comp.tff_repr,
        'federated_map(<(b -> a),federated_map(<(b -> a),data>)>)')
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(let fn=<(b -> a),(b -> a)> in (arg -> fn[1](fn[0](arg)))),data>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), '{int32}@CLIENTS')
    self.assertTrue(modified)

  def test_merges_nested_federated_maps(self):
    fn = _create_lambda_to_identity('a', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = computation_building_blocks.Data('data', arg_type)
    call = _create_chained_dummy_federated_maps([fn, fn], arg)
    block = _create_dummy_block(call, variable_name='b')
    comp = block

    transformed_comp, modified = transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(
        comp.tff_repr,
        '(let b=data in federated_map(<(a -> a),federated_map(<(a -> a),data>)>))'
    )
    self.assertEqual(
        transformed_comp.tff_repr,
        '(let b=data in federated_map(<(let fn=<(a -> a),(a -> a)> in (arg -> fn[1](fn[0](arg)))),data>))'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), '{int32}@CLIENTS')
    self.assertTrue(modified)

  def test_merges_multiple_federated_maps(self):
    fn = _create_lambda_to_identity('a', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = computation_building_blocks.Data('data', arg_type)
    call = _create_chained_dummy_federated_maps([fn, fn, fn], arg)
    comp = call

    transformed_comp, modified = transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(
        comp.tff_repr,
        'federated_map(<(a -> a),federated_map(<(a -> a),federated_map(<(a -> a),data>)>)>)'
    )
    # pyformat: disable
    # pylint: disable=bad-continuation
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<'
            '(let fn=<'
                '(let fn=<(a -> a),(a -> a)> in (arg -> fn[1](fn[0](arg)))),'
                '(a -> a)'
            '> in (arg -> fn[1](fn[0](arg)))),'
            'data'
        '>)'
    )
    # pylint: enable=bad-continuation
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), '{int32}@CLIENTS')
    self.assertTrue(modified)

  def test_does_not_merge_one_federated_map(self):
    fn = _create_lambda_to_identity('a', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = computation_building_blocks.Data('data', arg_type)
    call = computation_constructing_utils.create_federated_map(fn, arg)
    comp = call

    transformed_comp, modified = transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(transformed_comp.tff_repr,
                     'federated_map(<(a -> a),data>)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), '{int32}@CLIENTS')
    self.assertFalse(modified)

  def test_does_not_merge_separated_federated_maps(self):
    fn = _create_lambda_to_identity('a', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = computation_building_blocks.Data('data', arg_type)
    call_1 = computation_constructing_utils.create_federated_map(fn, arg)
    block = _create_dummy_block(call_1, variable_name='b')
    call_2 = computation_constructing_utils.create_federated_map(fn, block)
    comp = call_2

    transformed_comp, modified = transformations.merge_chained_federated_maps_or_applys(
        comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(
        transformed_comp.tff_repr,
        'federated_map(<(a -> a),(let b=data in federated_map(<(a -> a),data>))>)'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), '{int32}@CLIENTS')
    self.assertFalse(modified)


class MergeTupleIntrinsicsTest(absltest.TestCase):

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      transformations.merge_tuple_intrinsics(None)

  def test_merges_federated_aggregates(self):
    called_intrinsic = _create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    calls = computation_building_blocks.Tuple(
        (called_intrinsic, called_intrinsic))
    comp = calls

    transformed_comp, modified = transformations.merge_tuple_intrinsics(comp)

    self.assertEqual(
        comp.tff_repr,
        '<federated_aggregate(<data,data,(a -> data),(b -> data),(c -> c)>),federated_aggregate(<data,data,(a -> data),(b -> data),(c -> c)>)>'
    )
    # pyformat: disable
    # pylint: disable=bad-continuation
    self.assertEqual(
        transformed_comp.tff_repr,
        '(x -> <x[0],x[1]>)((let value=federated_aggregate(<'
            'federated_map(<(x -> <x[0],x[1]>),'
                'federated_map(<'
                    '(arg -> arg),'
                    '(let value=<data,data> in federated_zip_at_clients(<value[0],value[1]>))'
                '>)'
            '>),'
            '<data,data>,'
            '(let fn=<(a -> data),(a -> data)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>)),'
            '(let fn=<(b -> data),(b -> data)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>)),'
            '(let fn=<(c -> c),(c -> c)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>))'
        '>) in <'
            'federated_apply(<(arg -> arg[0]),value>),'
            'federated_apply(<(arg -> arg[1]),value>)'
        '>))'
    )
    # pylint: enable=bad-continuation
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature), '<int32@SERVER,int32@SERVER>')
    self.assertTrue(modified)

  def test_merges_federated_maps(self):
    called_intrinsic = _create_dummy_called_federated_map(parameter_name='a')
    calls = computation_building_blocks.Tuple(
        (called_intrinsic, called_intrinsic))
    comp = calls

    transformed_comp, modified = transformations.merge_tuple_intrinsics(comp)

    self.assertEqual(
        comp.tff_repr,
        '<federated_map(<(a -> a),data>),federated_map(<(a -> a),data>)>')
    # pyformat: disable
    # pylint: disable=bad-continuation
    self.assertEqual(
        transformed_comp.tff_repr,
        '(x -> <x[0],x[1]>)((let value=federated_map(<'
            '(let fn=<(a -> a),(a -> a)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>)),'
            'federated_map(<'
                '(x -> <x[0],x[1]>),'
                'federated_map(<'
                    '(arg -> arg),'
                    '(let value=<data,data> in federated_zip_at_clients(<value[0],value[1]>))'
                '>)'
            '>)'
        '>) in <'
            'federated_map(<(arg -> arg[0]),value>),'
            'federated_map(<(arg -> arg[1]),value>)'
        '>))'
    )
    # pylint: enable=bad-continuation
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature),
        '<{int32}@CLIENTS,{int32}@CLIENTS>')
    self.assertTrue(modified)

  def test_merges_federated_maps_with_different_names(self):
    called_intrinsic_1 = _create_dummy_called_federated_map(parameter_name='a')
    called_intrinsic_2 = _create_dummy_called_federated_map(parameter_name='b')
    calls = computation_building_blocks.Tuple(
        (called_intrinsic_1, called_intrinsic_2))
    comp = calls

    transformed_comp, modified = transformations.merge_tuple_intrinsics(comp)

    self.assertEqual(
        comp.tff_repr,
        '<federated_map(<(a -> a),data>),federated_map(<(b -> b),data>)>')
    # pyformat: disable
    # pylint: disable=bad-continuation
    self.assertEqual(
        transformed_comp.tff_repr,
        '(x -> <x[0],x[1]>)((let value=federated_map(<'
            '(let fn=<(a -> a),(b -> b)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>)),'
            'federated_map(<'
                '(x -> <x[0],x[1]>),'
                'federated_map(<'
                    '(arg -> arg),'
                    '(let value=<data,data> in federated_zip_at_clients(<value[0],value[1]>))'
                '>)'
            '>)'
        '>) in <'
            'federated_map(<(arg -> arg[0]),value>),'
            'federated_map(<(arg -> arg[1]),value>)'
        '>))'
    )
    # pylint: enable=bad-continuation
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature),
        '<{int32}@CLIENTS,{int32}@CLIENTS>')
    self.assertTrue(modified)

  def test_merges_federated_maps_with_different_types(self):
    called_intrinsic_1 = _create_dummy_called_federated_map(
        parameter_name='a', parameter_type=tf.int32)
    called_intrinsic_2 = _create_dummy_called_federated_map(
        parameter_name='b', parameter_type=tf.float32)
    calls = computation_building_blocks.Tuple(
        (called_intrinsic_1, called_intrinsic_2))
    comp = calls

    transformed_comp, modified = transformations.merge_tuple_intrinsics(comp)

    self.assertEqual(
        comp.tff_repr,
        '<federated_map(<(a -> a),data>),federated_map(<(b -> b),data>)>')
    # pyformat: disable
    # pylint: disable=bad-continuation
    self.assertEqual(
        transformed_comp.tff_repr,
        '(x -> <x[0],x[1]>)((let value=federated_map(<'
            '(let fn=<(a -> a),(b -> b)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>)),'
            'federated_map(<'
                '(x -> <x[0],x[1]>),'
                'federated_map(<'
                    '(arg -> arg),'
                    '(let value=<data,data> in federated_zip_at_clients(<value[0],value[1]>))'
                '>)'
            '>)'
        '>) in <'
            'federated_map(<(arg -> arg[0]),value>),'
            'federated_map(<(arg -> arg[1]),value>)'
        '>))'
    )
    # pylint: enable=bad-continuation
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature),
        '<{int32}@CLIENTS,{float32}@CLIENTS>')
    self.assertTrue(modified)

  def test_merges_federated_maps_with_named_parameter_type(self):
    parameter_type = [('b', tf.int32), ('c', tf.int32)]
    called_intrinsic = _create_dummy_called_federated_map(
        parameter_name='a', parameter_type=parameter_type)
    calls = computation_building_blocks.Tuple(
        (called_intrinsic, called_intrinsic))
    comp = calls

    transformed_comp, modified = transformations.merge_tuple_intrinsics(comp)

    self.assertEqual(
        comp.tff_repr,
        '<federated_map(<(a -> a),data>),federated_map(<(a -> a),data>)>')
    # pyformat: disable
    # pylint: disable=bad-continuation
    self.assertEqual(
        transformed_comp.tff_repr,
        '(x -> <x[0],x[1]>)((let value=federated_map(<'
            '(let fn=<(a -> a),(a -> a)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>)),'
            'federated_map(<'
                '(x -> <x[0],x[1]>),'
                'federated_map(<'
                    '(arg -> arg),'
                    '(let value=<data,data> in federated_zip_at_clients(<value[0],value[1]>))'
                '>)'
            '>)'
        '>) in <'
            'federated_map(<(arg -> arg[0]),value>),'
            'federated_map(<(arg -> arg[1]),value>)'
        '>))'
    )
    # pylint: enable=bad-continuation
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature),
        '<{<b=int32,c=int32>}@CLIENTS,{<b=int32,c=int32>}@CLIENTS>')
    self.assertTrue(modified)

  def test_merges_federated_maps_with_unbound_reference(self):
    ref = computation_building_blocks.Reference('a', tf.int32)
    fn = computation_building_blocks.Lambda('b', tf.int32, ref)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = computation_building_blocks.Data('data', arg_type)
    called_intrinsic = computation_constructing_utils.create_federated_map(
        fn, arg)
    calls = computation_building_blocks.Tuple(
        (called_intrinsic, called_intrinsic))
    comp = calls

    transformed_comp, modified = transformations.merge_tuple_intrinsics(comp)

    self.assertEqual(
        comp.tff_repr,
        '<federated_map(<(b -> a),data>),federated_map(<(b -> a),data>)>')
    # pyformat: disable
    # pylint: disable=bad-continuation
    self.assertEqual(
        transformed_comp.tff_repr,
        '(x -> <x[0],x[1]>)((let value=federated_map(<'
            '(let fn=<(b -> a),(b -> a)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>)),'
            'federated_map(<'
                '(x -> <x[0],x[1]>),'
                'federated_map(<'
                    '(arg -> arg),'
                    '(let value=<data,data> in federated_zip_at_clients(<value[0],value[1]>))'
                '>)'
            '>)'
        '>) in <'
            'federated_map(<(arg -> arg[0]),value>),'
            'federated_map(<(arg -> arg[1]),value>)'
        '>))'
    )
    # pylint: enable=bad-continuation
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature),
        '<{int32}@CLIENTS,{int32}@CLIENTS>')
    self.assertTrue(modified)

  def test_merges_named_federated_maps(self):
    called_intrinsic = _create_dummy_called_federated_map(parameter_name='a')
    calls = computation_building_blocks.Tuple(
        (('b', called_intrinsic), ('c', called_intrinsic)))
    comp = calls

    transformed_comp, modified = transformations.merge_tuple_intrinsics(comp)

    self.assertEqual(
        comp.tff_repr,
        '<b=federated_map(<(a -> a),data>),c=federated_map(<(a -> a),data>)>')
    # pyformat: disable
    # pylint: disable=bad-continuation
    self.assertEqual(
        transformed_comp.tff_repr,
        '(x -> <b=x[0],c=x[1]>)((let value=federated_map(<'
            '(let fn=<(a -> a),(a -> a)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>)),'
            'federated_map(<'
                '(x -> <x[0],x[1]>),'
                'federated_map(<'
                    '(arg -> arg),'
                    '(let value=<data,data> in federated_zip_at_clients(<value[0],value[1]>))'
                '>)'
            '>)'
        '>) in <'
            'federated_map(<(arg -> arg[0]),value>),'
            'federated_map(<(arg -> arg[1]),value>)'
        '>))'
    )
    # pylint: enable=bad-continuation
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature),
        '<b={int32}@CLIENTS,c={int32}@CLIENTS>')
    self.assertTrue(modified)

  def test_merges_nested_federated_maps(self):
    called_intrinsic = _create_dummy_called_federated_map(parameter_name='a')
    calls = computation_building_blocks.Tuple(
        (called_intrinsic, called_intrinsic))
    block = _create_dummy_block(calls, variable_name='a')
    comp = block

    transformed_comp, modified = transformations.merge_tuple_intrinsics(comp)

    self.assertEqual(
        comp.tff_repr,
        '(let a=data in <federated_map(<(a -> a),data>),federated_map(<(a -> a),data>)>)'
    )
    # pyformat: disable
    # pylint: disable=bad-continuation
    self.assertEqual(
        transformed_comp.tff_repr,
        '(let a=data in (x -> <x[0],x[1]>)((let value=federated_map(<'
            '(let fn=<(a -> a),(a -> a)> in (arg -> <fn[0](arg[0]),fn[1](arg[1])>)),'
            'federated_map(<'
                '(x -> <x[0],x[1]>),'
                'federated_map(<'
                    '(arg -> arg),'
                    '(let value=<data,data> in federated_zip_at_clients(<value[0],value[1]>))'
                '>)'
            '>)'
        '>) in <'
            'federated_map(<(arg -> arg[0]),value>),'
            'federated_map(<(arg -> arg[1]),value>)'
        '>)))'
    )
    # pylint: enable=bad-continuation
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature),
        '<{int32}@CLIENTS,{int32}@CLIENTS>')
    self.assertTrue(modified)

  def test_merges_multiple_federated_maps(self):
    called_intrinsic = _create_dummy_called_federated_map(parameter_name='a')
    calls = computation_building_blocks.Tuple(
        (called_intrinsic, called_intrinsic, called_intrinsic))
    comp = calls

    transformed_comp, modified = transformations.merge_tuple_intrinsics(comp)

    self.assertEqual(
        comp.tff_repr,
        '<federated_map(<(a -> a),data>),federated_map(<(a -> a),data>),federated_map(<(a -> a),data>)>'
    )
    # pyformat: disable
    # pylint: disable=bad-continuation
    self.assertEqual(
        transformed_comp.tff_repr,
        '(x -> <x[0],x[1],x[2]>)((let value=federated_map(<'
            '(let fn=<(a -> a),(a -> a),(a -> a)> in (arg -> <fn[0](arg[0]),fn[1](arg[1]),fn[2](arg[2])>)),'
            'federated_map(<'
                '(x -> <x[0],x[1],x[2]>),'
                'federated_map(<'
                    '(arg -> (let comps=<(arg -> arg)(arg[0]),arg[1]> in <comps[0][0],comps[0][1],comps[1]>)),'
                    '(let value=<data,data,data> in federated_zip_at_clients(<federated_zip_at_clients(<value[0],value[1]>),value[2]>))'
                '>)'
            '>)'
        '>) in <'
            'federated_map(<(arg -> arg[0]),value>),'
            'federated_map(<(arg -> arg[1]),value>),'
            'federated_map(<(arg -> arg[2]),value>)'
        '>))'
    )
    # pylint: enable=bad-continuation
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature),
        '<{int32}@CLIENTS,{int32}@CLIENTS,{int32}@CLIENTS>')
    self.assertTrue(modified)

  def test_merges_one_federated_map(self):
    called_intrinsic = _create_dummy_called_federated_map(parameter_name='a')
    calls = computation_building_blocks.Tuple((called_intrinsic,))
    comp = calls

    transformed_comp, modified = transformations.merge_tuple_intrinsics(comp)

    self.assertEqual(comp.tff_repr, '<federated_map(<(a -> a),data>)>')
    # pyformat: disable
    # pylint: disable=bad-continuation
    self.assertEqual(
        transformed_comp.tff_repr,
        '(x -> <x[0]>)((let value=federated_map(<'
            '(let fn=<(a -> a)> in (arg -> <fn[0](arg[0])>)),'
            'federated_map(<(arg -> <arg>),<data>[0]>)'
        '>) in <federated_map(<(arg -> arg[0]),value>)>))'
    )
    # pylint: enable=bad-continuation
    # pyformat: enable
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), '<{int32}@CLIENTS>')
    self.assertTrue(modified)

  def test_does_not_merge_different_federated_maps(self):
    called_intrinsic_1 = _create_dummy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    called_intrinsic_2 = _create_dummy_called_federated_map(parameter_name='a')
    calls = computation_building_blocks.Tuple(
        (called_intrinsic_1, called_intrinsic_2))
    comp = calls

    transformed_comp, modified = transformations.merge_tuple_intrinsics(comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(
        transformed_comp.tff_repr,
        '<federated_aggregate(<data,data,(a -> data),(b -> data),(c -> c)>),federated_map(<(a -> a),data>)>'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(
        str(transformed_comp.type_signature), '<int32@SERVER,{int32}@CLIENTS>')
    self.assertFalse(modified)

  def test_does_not_merge_dummy_intrinsics(self):
    called_intrinsic = _create_dummy_called_intrinsic(parameter_name='a')
    calls = computation_building_blocks.Tuple(
        (called_intrinsic, called_intrinsic))
    comp = calls

    transformed_comp, modified = transformations.merge_tuple_intrinsics(comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(transformed_comp.tff_repr, '<intrinsic(a),intrinsic(a)>')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertEqual(str(transformed_comp.type_signature), '<int32,int32>')
    self.assertFalse(modified)


class RemoveMappedOrAppliedIdentityTest(parameterized.TestCase):

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      transformations.remove_mapped_or_applied_identity(None)

  # pyformat: disable
  @parameterized.named_parameters(
      ('federated_apply',
       intrinsic_defs.FEDERATED_APPLY.uri,
       _create_dummy_called_federated_apply),
      ('federated_map',
       intrinsic_defs.FEDERATED_MAP.uri,
       _create_dummy_called_federated_map),
      ('sequence_map',
       intrinsic_defs.SEQUENCE_MAP.uri,
       _create_dummy_called_sequence_map))
  # pyformat: enable
  def test_removes_intrinsic(self, uri, factory):
    call = factory(parameter_name='a')
    comp = call

    transformed_comp, modified = transformations.remove_mapped_or_applied_identity(
        comp)

    self.assertEqual(comp.tff_repr, '{}(<(a -> a),data>)'.format(uri))
    self.assertEqual(transformed_comp.tff_repr, 'data')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_removes_federated_map_with_named_result(self):
    parameter_type = [('a', tf.int32), ('b', tf.int32)]
    fn = _create_lambda_to_identity('c', parameter_type)
    arg_type = computation_types.FederatedType(parameter_type,
                                               placements.CLIENTS)
    arg = computation_building_blocks.Data('data', arg_type)
    call = computation_constructing_utils.create_federated_map(fn, arg)
    comp = call

    transformed_comp, modified = transformations.remove_mapped_or_applied_identity(
        comp)

    self.assertEqual(comp.tff_repr, 'federated_map(<(c -> c),data>)')
    self.assertEqual(transformed_comp.tff_repr, 'data')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_removes_nested_federated_map(self):
    called_intrinsic = _create_dummy_called_federated_map(parameter_name='a')
    block = _create_dummy_block(called_intrinsic, variable_name='b')
    comp = block

    transformed_comp, modified = transformations.remove_mapped_or_applied_identity(
        comp)

    self.assertEqual(comp.tff_repr,
                     '(let b=data in federated_map(<(a -> a),data>))')
    self.assertEqual(transformed_comp.tff_repr, '(let b=data in data)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_removes_chained_federated_maps(self):
    fn = _create_lambda_to_identity('a', tf.int32)
    arg_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    arg = computation_building_blocks.Data('data', arg_type)
    call = _create_chained_dummy_federated_maps([fn, fn], arg)
    comp = call

    transformed_comp, modified = transformations.remove_mapped_or_applied_identity(
        comp)

    self.assertEqual(
        comp.tff_repr,
        'federated_map(<(a -> a),federated_map(<(a -> a),data>)>)')
    self.assertEqual(transformed_comp.tff_repr, 'data')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_does_not_remove_dummy_intrinsic(self):
    comp = _create_dummy_called_intrinsic(parameter_name='a')

    transformed_comp, modified = transformations.remove_mapped_or_applied_identity(
        comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(transformed_comp.tff_repr, 'intrinsic(a)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_does_not_remove_called_lambda(self):
    fn = _create_lambda_to_identity('a', tf.int32)
    arg = computation_building_blocks.Data('data', tf.int32)
    call = computation_building_blocks.Call(fn, arg)
    comp = call

    transformed_comp, modified = transformations.remove_mapped_or_applied_identity(
        comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(transformed_comp.tff_repr, '(a -> a)(data)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)


class ReplaceCalledLambdaWithBlockTest(absltest.TestCase):

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      transformations.replace_called_lambda_with_block(None)

  def test_replaces_called_lambda(self):
    fn = _create_lambda_to_identity('a', tf.int32)
    arg = computation_building_blocks.Data('data', tf.int32)
    call = computation_building_blocks.Call(fn, arg)
    comp = call

    transformed_comp, modified = transformations.replace_called_lambda_with_block(
        comp)

    self.assertEqual(comp.tff_repr, '(a -> a)(data)')
    self.assertEqual(transformed_comp.tff_repr, '(let a=data in a)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replaces_nested_called_lambda(self):
    fn = _create_lambda_to_identity('a', tf.int32)
    arg = computation_building_blocks.Data('data', tf.int32)
    call = computation_building_blocks.Call(fn, arg)
    block = _create_dummy_block(call, variable_name='b')
    comp = block

    transformed_comp, modified = transformations.replace_called_lambda_with_block(
        comp)

    self.assertEqual(comp.tff_repr, '(let b=data in (a -> a)(data))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let b=data in (let a=data in a))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replaces_chained_called_lambdas(self):
    fn = _create_lambda_to_identity('a', tf.int32)
    arg = computation_building_blocks.Data('data', tf.int32)
    call = _create_chained_calls([fn, fn], arg)
    comp = call

    transformed_comp, modified = transformations.replace_called_lambda_with_block(
        comp)

    self.assertEqual(comp.tff_repr, '(a -> a)((a -> a)(data))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let a=(let a=data in a) in a)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_does_not_replace_uncalled_lambda(self):
    fn = _create_lambda_to_identity('a', tf.int32)
    comp = fn

    transformed_comp, modified = transformations.replace_called_lambda_with_block(
        comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(transformed_comp.tff_repr, '(a -> a)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)

  def test_does_not_replace_separated_called_lambda(self):
    fn = _create_lambda_to_identity('a', tf.int32)
    block = _create_dummy_block(fn, variable_name='b')
    arg = computation_building_blocks.Data('data', tf.int32)
    call = computation_building_blocks.Call(block, arg)
    comp = call

    transformed_comp, modified = transformations.replace_called_lambda_with_block(
        comp)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(transformed_comp.tff_repr,
                     '(let b=data in (a -> a))(data)')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)


class ReplaceIntrinsicWithCallableTest(absltest.TestCase):

  def test_raises_type_error_none_comp(self):
    uri = 'intrinsic'
    body = lambda x: x

    with self.assertRaises(TypeError):
      transformations.replace_intrinsic_with_callable(
          None, uri, body, context_stack_impl.context_stack)

  def test_raises_type_error_none_uri(self):
    comp = _create_lambda_to_dummy_intrinsic(parameter_name='a')
    body = lambda x: x

    with self.assertRaises(TypeError):
      transformations.replace_intrinsic_with_callable(
          comp, None, body, context_stack_impl.context_stack)

  def test_raises_type_error_none_body(self):
    comp = _create_lambda_to_dummy_intrinsic(parameter_name='a')
    uri = 'intrinsic'

    with self.assertRaises(TypeError):
      transformations.replace_intrinsic_with_callable(
          comp, uri, None, context_stack_impl.context_stack)

  def test_raises_type_error_none_context_stack(self):
    comp = _create_lambda_to_dummy_intrinsic(parameter_name='a')
    uri = 'intrinsic'
    body = lambda x: x

    with self.assertRaises(TypeError):
      transformations.replace_intrinsic_with_callable(comp, uri, body, None)

  def test_replaces_intrinsic(self):
    comp = _create_lambda_to_dummy_intrinsic(parameter_name='a')
    uri = 'intrinsic'
    body = lambda x: x

    transformed_comp, modified = transformations.replace_intrinsic_with_callable(
        comp, uri, body, context_stack_impl.context_stack)

    self.assertEqual(comp.tff_repr, '(a -> intrinsic(a))')
    self.assertEqual(transformed_comp.tff_repr,
                     '(a -> (intrinsic_arg -> intrinsic_arg)(a))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replaces_nested_intrinsic(self):
    fn = _create_lambda_to_dummy_intrinsic(parameter_name='a')
    block = _create_dummy_block(fn, variable_name='b')
    comp = block
    uri = 'intrinsic'
    body = lambda x: x

    transformed_comp, modified = transformations.replace_intrinsic_with_callable(
        comp, uri, body, context_stack_impl.context_stack)

    self.assertEqual(comp.tff_repr, '(let b=data in (a -> intrinsic(a)))')
    self.assertEqual(
        transformed_comp.tff_repr,
        '(let b=data in (a -> (intrinsic_arg -> intrinsic_arg)(a)))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replaces_chained_intrinsics(self):
    fn = _create_lambda_to_dummy_intrinsic(parameter_name='a')
    arg = computation_building_blocks.Data('data', tf.int32)
    call = _create_chained_calls([fn, fn], arg)
    comp = call
    uri = 'intrinsic'
    body = lambda x: x

    transformed_comp, modified = transformations.replace_intrinsic_with_callable(
        comp, uri, body, context_stack_impl.context_stack)

    self.assertEqual(comp.tff_repr,
                     '(a -> intrinsic(a))((a -> intrinsic(a))(data))')
    self.assertEqual(
        transformed_comp.tff_repr,
        '(a -> (intrinsic_arg -> intrinsic_arg)(a))((a -> (intrinsic_arg -> intrinsic_arg)(a))(data))'
    )
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_does_not_replace_other_intrinsic(self):
    comp = _create_lambda_to_dummy_intrinsic(parameter_name='a')
    uri = 'other'
    body = lambda x: x

    transformed_comp, modified = transformations.replace_intrinsic_with_callable(
        comp, uri, body, context_stack_impl.context_stack)

    self.assertEqual(transformed_comp.tff_repr, comp.tff_repr)
    self.assertEqual(transformed_comp.tff_repr, '(a -> intrinsic(a))')
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)


class ReplaceSelectionFromTupleWithElementTest(absltest.TestCase):

  def test_fails_on_none_comp(self):
    with self.assertRaises(TypeError):
      transformations.replace_selection_from_tuple_with_element(None)

  def test_leaves_selection_from_ref_by_index_alone(self):
    ref_to_tuple = computation_building_blocks.Reference(
        'tup', [('a', tf.int32), ('b', tf.float32)])
    a_selected = computation_building_blocks.Selection(ref_to_tuple, index=0)
    b_selected = computation_building_blocks.Selection(ref_to_tuple, index=1)

    a_returned, a_transformed = transformations.replace_selection_from_tuple_with_element(
        a_selected)
    b_returned, b_transformed = transformations.replace_selection_from_tuple_with_element(
        b_selected)

    self.assertFalse(a_transformed)
    self.assertEqual(a_returned.proto, a_selected.proto)
    self.assertFalse(b_transformed)
    self.assertEqual(b_returned.proto, b_selected.proto)

  def test_leaves_selection_from_ref_by_name_alone(self):
    ref_to_tuple = computation_building_blocks.Reference(
        'tup', [('a', tf.int32), ('b', tf.float32)])
    a_selected = computation_building_blocks.Selection(ref_to_tuple, name='a')
    b_selected = computation_building_blocks.Selection(ref_to_tuple, name='b')

    a_returned, a_transformed = transformations.replace_selection_from_tuple_with_element(
        a_selected)
    b_returned, b_transformed = transformations.replace_selection_from_tuple_with_element(
        b_selected)

    self.assertFalse(a_transformed)
    self.assertEqual(a_returned.proto, a_selected.proto)
    self.assertFalse(b_transformed)
    self.assertEqual(b_returned.proto, b_selected.proto)

  def test_by_index_grabs_correct_element(self):
    x_data = computation_building_blocks.Data('x', tf.int32)
    y_data = computation_building_blocks.Data('y', [('a', tf.float32)])
    tup = computation_building_blocks.Tuple([x_data, y_data])
    x_selected = computation_building_blocks.Selection(tup, index=0)
    y_selected = computation_building_blocks.Selection(tup, index=1)

    collapsed_selection_x, x_transformed = transformations.replace_selection_from_tuple_with_element(
        x_selected)
    collapsed_selection_y, y_transformed = transformations.replace_selection_from_tuple_with_element(
        y_selected)

    self.assertTrue(x_transformed)
    self.assertTrue(y_transformed)
    self.assertEqual(collapsed_selection_x.proto, x_data.proto)
    self.assertEqual(collapsed_selection_y.proto, y_data.proto)

  def test_by_name_grabs_correct_element(self):
    x_data = computation_building_blocks.Data('x', tf.int32)
    y_data = computation_building_blocks.Data('y', [('a', tf.float32)])
    tup = computation_building_blocks.Tuple([('a', x_data), ('b', y_data)])
    x_selected = computation_building_blocks.Selection(tup, name='a')
    y_selected = computation_building_blocks.Selection(tup, name='b')

    collapsed_selection_x, x_transformed = transformations.replace_selection_from_tuple_with_element(
        x_selected)
    collapsed_selection_y, y_transformed = transformations.replace_selection_from_tuple_with_element(
        y_selected)

    self.assertTrue(x_transformed)
    self.assertTrue(y_transformed)
    self.assertEqual(collapsed_selection_x.proto, x_data.proto)
    self.assertEqual(collapsed_selection_y.proto, y_data.proto)


class UniquifyCompiledComputationNamesTest(parameterized.TestCase):

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      transformations.uniquify_compiled_computation_names(None)

  def test_replaces_name(self):
    fn = lambda: tf.constant(1)
    tf_comp, _ = tensorflow_serialization.serialize_py_fn_as_tf_computation(
        fn, None, context_stack_impl.context_stack)
    compiled_comp = computation_building_blocks.CompiledComputation(tf_comp)
    comp = compiled_comp

    transformed_comp, modified = transformations.uniquify_compiled_computation_names(
        comp)

    self.assertNotEqual(transformed_comp._name, comp._name)
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertTrue(modified)

  def test_replaces_multiple_names(self):
    elements = []
    for _ in range(10):
      fn = lambda: tf.constant(1)
      tf_comp, _ = tensorflow_serialization.serialize_py_fn_as_tf_computation(
          fn, None, context_stack_impl.context_stack)
      compiled_comp = computation_building_blocks.CompiledComputation(tf_comp)
      elements.append(compiled_comp)
    compiled_comps = computation_building_blocks.Tuple(elements)
    comp = compiled_comps

    transformed_comp, modified = transformations.uniquify_compiled_computation_names(
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
    comp = computation_building_blocks.Reference('name', tf.int32)

    transformed_comp, modified = transformations.uniquify_compiled_computation_names(
        comp)

    self.assertEqual(transformed_comp._name, comp._name)
    self.assertEqual(transformed_comp.type_signature, comp.type_signature)
    self.assertFalse(modified)


class UniquifyReferenceNamesTest(absltest.TestCase):

  def test_raises_type_error(self):
    with self.assertRaises(TypeError):
      transformations.uniquify_reference_names(None)

  def test_single_level_block(self):
    ref = computation_building_blocks.Reference('a', tf.int32)
    data = computation_building_blocks.Data('data', tf.int32)
    block = computation_building_blocks.Block(
        (('a', data), ('a', ref), ('a', ref)), ref)

    transformed_comp, modified = transformations.uniquify_reference_names(block)

    self.assertEqual(block.tff_repr, '(let a=data,a=a,a=a in a)')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let _var1=data,_var2=_var1,_var3=_var2 in _var3)')
    self.assertTrue(transformation_utils.has_unique_names(transformed_comp))
    self.assertTrue(modified)

  def test_nested_blocks(self):
    x_ref = computation_building_blocks.Reference('a', tf.int32)
    data = computation_building_blocks.Data('data', tf.int32)
    block1 = computation_building_blocks.Block([('a', data), ('a', x_ref)],
                                               x_ref)
    block2 = computation_building_blocks.Block([('a', data), ('a', x_ref)],
                                               block1)

    transformed_comp, modified = transformations.uniquify_reference_names(
        block2)

    self.assertEqual(block2.tff_repr,
                     '(let a=data,a=a in (let a=data,a=a in a))')
    self.assertEqual(
        transformed_comp.tff_repr,
        '(let _var1=data,_var2=_var1 in (let _var3=data,_var4=_var3 in _var4))')
    self.assertTrue(transformation_utils.has_unique_names(transformed_comp))
    self.assertTrue(modified)

  def test_nested_lambdas(self):
    data = computation_building_blocks.Data('data', tf.int32)
    input1 = computation_building_blocks.Reference('a', data.type_signature)
    first_level_call = computation_building_blocks.Call(
        computation_building_blocks.Lambda('a', input1.type_signature, input1),
        data)
    input2 = computation_building_blocks.Reference(
        'b', first_level_call.type_signature)
    second_level_call = computation_building_blocks.Call(
        computation_building_blocks.Lambda('b', input2.type_signature, input2),
        first_level_call)

    transformed_comp, modified = transformations.uniquify_reference_names(
        second_level_call)

    self.assertEqual(transformed_comp.tff_repr,
                     '(_var1 -> _var1)((_var2 -> _var2)(data))')
    self.assertTrue(transformation_utils.has_unique_names(transformed_comp))
    self.assertTrue(modified)

  def test_block_lambda_block_lambda(self):
    x_ref = computation_building_blocks.Reference('a', tf.int32)
    inner_lambda = computation_building_blocks.Lambda('a', tf.int32, x_ref)
    called_lambda = computation_building_blocks.Call(inner_lambda, x_ref)
    lower_block = computation_building_blocks.Block([('a', x_ref),
                                                     ('a', x_ref)],
                                                    called_lambda)
    second_lambda = computation_building_blocks.Lambda('a', tf.int32,
                                                       lower_block)
    second_call = computation_building_blocks.Call(second_lambda, x_ref)
    data = computation_building_blocks.Data('data', tf.int32)
    last_block = computation_building_blocks.Block([('a', data), ('a', x_ref)],
                                                   second_call)

    transformed_comp, modified = transformations.uniquify_reference_names(
        last_block)

    self.assertEqual(
        last_block.tff_repr,
        '(let a=data,a=a in (a -> (let a=a,a=a in (a -> a)(a)))(a))')
    self.assertEqual(
        transformed_comp.tff_repr,
        '(let _var1=data,_var2=_var1 in (_var3 -> (let _var4=_var3,_var5=_var4 in (_var6 -> _var6)(_var5)))(_var2))'
    )
    self.assertTrue(transformation_utils.has_unique_names(transformed_comp))
    self.assertTrue(modified)

  def test_blocks_nested_inside_of_locals(self):
    data = computation_building_blocks.Data('data', tf.int32)
    lower_block = computation_building_blocks.Block([('a', data)], data)
    middle_block = computation_building_blocks.Block([('a', lower_block)], data)
    higher_block = computation_building_blocks.Block([('a', middle_block)],
                                                     data)
    y_ref = computation_building_blocks.Reference('a', tf.int32)
    lower_block_with_y_ref = computation_building_blocks.Block([('a', y_ref)],
                                                               data)
    middle_block_with_y_ref = computation_building_blocks.Block(
        [('a', lower_block_with_y_ref)], data)
    higher_block_with_y_ref = computation_building_blocks.Block(
        [('a', middle_block_with_y_ref)], data)
    multiple_bindings_highest_block = computation_building_blocks.Block(
        [('a', higher_block),
         ('a', higher_block_with_y_ref)], higher_block_with_y_ref)

    transformed_comp, modified = transformations.uniquify_reference_names(
        multiple_bindings_highest_block)

    self.assertEqual(higher_block.tff_repr,
                     '(let a=(let a=(let a=data in data) in data) in data)')
    self.assertEqual(higher_block_with_y_ref.tff_repr,
                     '(let a=(let a=(let a=a in data) in data) in data)')
    self.assertEqual(transformed_comp.locals[0][0], '_var4')
    self.assertEqual(
        transformed_comp.locals[0][1].tff_repr,
        '(let _var3=(let _var2=(let _var1=data in data) in data) in data)')
    self.assertEqual(transformed_comp.locals[1][0], '_var8')
    self.assertEqual(
        transformed_comp.locals[1][1].tff_repr,
        '(let _var7=(let _var6=(let _var5=_var4 in data) in data) in data)')
    self.assertEqual(
        transformed_comp.result.tff_repr,
        '(let _var11=(let _var10=(let _var9=_var8 in data) in data) in data)')
    self.assertTrue(transformation_utils.has_unique_names(transformed_comp))
    self.assertTrue(modified)

  def test_renames_names_ignores_existing_names(self):
    data = computation_building_blocks.Data('data', tf.int32)
    block = computation_building_blocks.Block([('a', data), ('b', data)], data)
    comp = block

    transformed_comp, modified = transformations.uniquify_reference_names(comp)

    self.assertEqual(block.tff_repr, '(let a=data,b=data in data)')
    self.assertEqual(transformed_comp.tff_repr,
                     '(let _var1=data,_var2=data in data)')
    self.assertTrue(modified)

    transformed_comp, modified = transformations.uniquify_reference_names(comp)

    self.assertEqual(transformed_comp.tff_repr,
                     '(let _var1=data,_var2=data in data)')
    self.assertTrue(modified)


if __name__ == '__main__':
  absltest.main()
