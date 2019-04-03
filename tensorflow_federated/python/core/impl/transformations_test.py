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
"""Tests for transformations.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
import six
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl import computation_building_blocks
from tensorflow_federated.python.core.impl import computation_impl
from tensorflow_federated.python.core.impl import context_stack_impl
from tensorflow_federated.python.core.impl import intrinsic_bodies
from tensorflow_federated.python.core.impl import intrinsic_defs
from tensorflow_federated.python.core.impl import transformations


def _to_building_block(comp):
  """Deserializes `comp` into a computation building block.

  Args:
    comp: An instance of `ComputationImpl` to deserialize.

  Returns:
    A corresponding instance of `ComputationBuildingBlock`.
  """
  assert isinstance(comp, computation_impl.ComputationImpl)
  proto = computation_impl.ComputationImpl.get_proto(comp)
  return computation_building_blocks.ComputationBuildingBlock.from_proto(proto)


class TransformationsTest(parameterized.TestCase):

  def test_transform_postorder_fails_on_none(self):

    def transform(comp):
      return comp

    with self.assertRaises(TypeError):
      transformations.transform_postorder(None, transform)

  def test_transform_postorder_with_lambda_call_selection_and_reference(self):

    @computations.federated_computation(
        [computation_types.FunctionType(tf.int32, tf.int32), tf.int32])
    def foo(f, x):
      return f(x)

    comp = _to_building_block(foo)
    self.assertEqual(str(comp), '(foo_arg -> foo_arg[0](foo_arg[1]))')

    def _transformation_fn_generator():
      n = 0
      while True:
        n = n + 1

        def _fn(x):
          return computation_building_blocks.Call(
              computation_building_blocks.Intrinsic(
                  'F{}'.format(n),
                  computation_types.FunctionType(x.type_signature,
                                                 x.type_signature)), x)

        yield _fn

    transformation_fn_sequence = _transformation_fn_generator()
    # pylint: disable=unnecessary-lambda
    tx_fn = lambda x: six.next(transformation_fn_sequence)(x)
    # pylint: enable=unnecessary-lambda
    transfomed_comp = transformations.transform_postorder(comp, tx_fn)
    self.assertEqual(
        str(transfomed_comp),
        'F6((foo_arg -> F5(F2(F1(foo_arg)[0])(F4(F3(foo_arg)[1])))))')

  # TODO(b/113123410): Add more tests for corner cases of `transform_preorder`.

  def test_name_compiled_computations(self):
    plus = computations.tf_computation(lambda x, y: x + y, [tf.int32, tf.int32])

    @computations.federated_computation(tf.int32)
    def add_one(x):
      return plus(x, 1)

    comp = _to_building_block(add_one)
    transformed_comp = transformations.name_compiled_computations(comp)
    self.assertEqual(
        str(transformed_comp),
        '(add_one_arg -> comp#1(<add_one_arg,comp#2()>))')

  def test_replace_intrinsic(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.SERVER, True))
    def foo(x):
      return intrinsics.federated_sum(intrinsics.federated_broadcast(x))

    comp = _to_building_block(foo)
    self.assertEqual(
        str(comp), '(foo_arg -> federated_sum(federated_broadcast(foo_arg)))')

    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    transformed_comp = transformations.replace_intrinsic(
        comp, intrinsic_defs.FEDERATED_SUM.uri, bodies['federated_sum'],
        context_stack_impl.context_stack)

    # TODO(b/120793862): Add a transform to eliminate unnecessary lambdas, then
    # simplify this test.

    self.assertEqual(
        str(transformed_comp),
        '(foo_arg -> (federated_sum_arg -> federated_reduce('
        '<federated_sum_arg,generic_zero,generic_plus>))'
        '(federated_broadcast(foo_arg)))')

  def test_replace_intrinsic_plus_reduce_lambdas(self):

    @computations.federated_computation(
        computation_types.FederatedType(tf.int32, placements.SERVER, True))
    def foo(x):
      return intrinsics.federated_sum(intrinsics.federated_broadcast(x))

    comp = _to_building_block(foo)

    self.assertEqual(
        str(comp), '(foo_arg -> federated_sum(federated_broadcast(foo_arg)))')

    bodies = intrinsic_bodies.get_intrinsic_bodies(
        context_stack_impl.context_stack)

    transformed_comp = transformations.replace_intrinsic(
        comp, intrinsic_defs.FEDERATED_SUM.uri, bodies['federated_sum'],
        context_stack_impl.context_stack)

    self.assertEqual(
        str(transformed_comp),
        '(foo_arg -> (federated_sum_arg -> federated_reduce('
        '<federated_sum_arg,generic_zero,generic_plus>))'
        '(federated_broadcast(foo_arg)))')

    reduced_lambda_comp = transformations.replace_called_lambdas_with_block(
        transformed_comp)

    self.assertEqual(
        str(reduced_lambda_comp),
        '(foo_arg -> (let federated_sum_arg=federated_broadcast(foo_arg) in '
        'federated_reduce(<federated_sum_arg,generic_zero,generic_plus>)))')

  def test_simple_reduce_lambda(self):
    x = computation_building_blocks.Reference('x', [tf.int32])
    l = computation_building_blocks.Lambda('x', [tf.int32], x)
    input_val = computation_building_blocks.Tuple(
        [computation_building_blocks.Data('test', tf.int32)])
    called = computation_building_blocks.Call(l, input_val)
    self.assertEqual(str(called), '(x -> x)(<test>)')
    reduced = transformations.replace_called_lambdas_with_block(called)
    self.assertEqual(str(reduced), '(let x=<test> in x)')

  def test_nested_reduce_lambda(self):
    comp = computation_building_blocks.Tuple(
        [computation_building_blocks.Data('test', tf.int32)])
    input1 = computation_building_blocks.Reference('input1',
                                                   comp.type_signature)
    first_level_call = computation_building_blocks.Call(
        computation_building_blocks.Lambda('input1', input1.type_signature,
                                           input1), comp)
    input2 = computation_building_blocks.Reference(
        'input2', first_level_call.type_signature)
    second_level_call = computation_building_blocks.Call(
        computation_building_blocks.Lambda('input2', input2.type_signature,
                                           input2), first_level_call)

    lambda_reduced_comp = transformations.replace_called_lambdas_with_block(
        second_level_call)
    self.assertEqual(
        str(second_level_call),
        '(input2 -> input2)((input1 -> input1)(<test>))')
    self.assertEqual(
        str(lambda_reduced_comp),
        '(let input2=(let input1=<test> in input1) in input2)')

  def test_no_reduce_lambda_without_call(self):

    @computations.federated_computation(tf.int32)
    def foo(x):
      return x

    comp = _to_building_block(foo)
    py_typecheck.check_type(comp, computation_building_blocks.Lambda)
    lambda_reduced_comp = transformations.replace_called_lambdas_with_block(
        comp)
    self.assertEqual(str(comp), '(foo_arg -> foo_arg)')
    self.assertEqual(str(comp), str(lambda_reduced_comp))

  def test_remove_mapped_or_applied_identity_fails_on_none(self):
    with self.assertRaises(TypeError):
      transformations.remove_mapped_or_applied_identity(None)

  # pyformat: disable
  @parameterized.named_parameters(
      ('federated_map', 'federated_map',
       computation_types.FederatedType(tf.float32, placements.CLIENTS)),
      ('federated_apply', 'federated_apply',
       computation_types.FederatedType(tf.float32, placements.SERVER)),
      ('sequence_map', 'sequence_map', computation_types.SequenceType(
          tf.float32)))
  # pyformat: enable
  def test_remove_identity_at_root(self, uri_string, data_type):
    data = computation_building_blocks.Data('x', data_type)
    identity_arg = computation_building_blocks.Reference('arg', tf.float32)
    identity_lam = computation_building_blocks.Lambda('arg', tf.float32,
                                                      identity_arg)
    arg_tuple = computation_building_blocks.Tuple([identity_lam, data])
    intrinsic = computation_building_blocks.Intrinsic(
        uri_string,
        computation_types.FunctionType(
            [arg_tuple.type_signature[0], arg_tuple.type_signature[1]],
            arg_tuple.type_signature[1]))
    call = computation_building_blocks.Call(intrinsic, arg_tuple)
    self.assertEqual(str(call), uri_string + '(<(arg -> arg),x>)')
    reduced = transformations.remove_mapped_or_applied_identity(call)
    self.assertEqual(str(reduced), 'x')

  @parameterized.named_parameters(
      ('federated_map', 'federated_map',
       computation_types.FederatedType(tf.float32, placements.CLIENTS)),
      ('federated_apply', 'federated_apply',
       computation_types.FederatedType(tf.float32, placements.SERVER)),
      ('sequence_map', 'sequence_map', computation_types.SequenceType(
          tf.float32)))
  def test_identity_removed_deep_in_tree(self, uri_string, data_type):
    data = computation_building_blocks.Data('x', data_type)
    identity_arg = computation_building_blocks.Reference('arg', tf.float32)
    identity_lam = computation_building_blocks.Lambda('arg', tf.float32,
                                                      identity_arg)
    arg_tuple = computation_building_blocks.Tuple([identity_lam, data])
    seq_apply = computation_building_blocks.Intrinsic(
        uri_string,
        computation_types.FunctionType(
            [arg_tuple.type_signature[0], arg_tuple.type_signature[1]],
            arg_tuple.type_signature[1]))
    call = computation_building_blocks.Call(seq_apply, arg_tuple)
    tuple_wrapped_call = computation_building_blocks.Tuple([call])
    lambda_wrapped_tuple = computation_building_blocks.Lambda(
        'y', tf.int32, tuple_wrapped_call)
    self.assertEqual(
        str(lambda_wrapped_tuple),
        '(y -> <' + uri_string + '(<(arg -> arg),x>)>)')
    reduced = transformations.remove_mapped_or_applied_identity(
        lambda_wrapped_tuple)
    self.assertEqual(str(reduced), '(y -> <x>)')

  def test_remove_identity_does_not_remove_dummy_intrinsic(self):
    dummy_intrinsic = computation_building_blocks.Intrinsic('dummy', [])
    new_dummy = transformations.remove_mapped_or_applied_identity(
        dummy_intrinsic)
    self.assertEqual(str(new_dummy), str(dummy_intrinsic))

  def test_remove_identity_does_not_remove_unmapped_lambda(self):
    x = computation_building_blocks.Reference('x', tf.int32)
    dummy_lambda = computation_building_blocks.Lambda('x', tf.int32, x)
    test_arg = computation_building_blocks.Data('test', tf.int32)
    called = computation_building_blocks.Call(dummy_lambda, test_arg)
    self.assertEqual(str(called), '(x -> x)(test)')
    self.assertEqual(
        str(transformations.remove_mapped_or_applied_identity(called)),
        '(x -> x)(test)')

  def test_no_reduce_separated_lambda_and_call(self):

    @computations.federated_computation(tf.int32)
    def foo(x):
      return x

    comp = _to_building_block(foo)
    block_wrapped_comp = computation_building_blocks.Block([], comp)
    test_arg = computation_building_blocks.Data('test', tf.int32)
    called_block = computation_building_blocks.Call(block_wrapped_comp,
                                                    test_arg)
    lambda_reduced_comp = transformations.replace_called_lambdas_with_block(
        called_block)
    self.assertEqual(str(called_block), '(let  in (foo_arg -> foo_arg))(test)')
    self.assertEqual(str(called_block), str(lambda_reduced_comp))

  def test_simple_block_snapshot(self):
    used1 = computation_building_blocks.Reference('used1', tf.int32)
    used2 = computation_building_blocks.Reference('used2', tf.int32)
    ref = computation_building_blocks.Reference('x', used1.type_signature)
    lower_block = computation_building_blocks.Block([('x', used1)], ref)
    higher_block = computation_building_blocks.Block([('used1', used2)],
                                                     lower_block)
    self.assertEqual(
        str(higher_block), '(let used1=used2 in (let x=used1 in x))')
    snapshot = transformations.scope_count_snapshot(higher_block)
    self.assertEqual(snapshot[str(lower_block)]['x'], 1)
    self.assertEqual(snapshot[str(higher_block)]['used1'], 1)
    self.assertIsNone(snapshot[str(higher_block)].get('x'))

  def test_scope_snapshot_block_overwrite(self):
    innermost = computation_building_blocks.Reference('x', tf.int32)
    intermediate_arg = computation_building_blocks.Reference('y', tf.int32)
    item2 = computation_building_blocks.Block([('x', intermediate_arg)],
                                              innermost)
    item1 = computation_building_blocks.Reference('x', tf.int32)
    mediate_tuple = computation_building_blocks.Tuple([item1, item2])
    used = computation_building_blocks.Reference('used', tf.int32)
    used1 = computation_building_blocks.Reference('used1', tf.int32)
    outer_block = computation_building_blocks.Block([('x', used), ('y', used1)],
                                                    mediate_tuple)
    self.assertEqual(
        str(outer_block), '(let x=used,y=used1 in <x,(let x=y in x)>)')
    snapshot = transformations.scope_count_snapshot(outer_block)
    self.assertEqual(snapshot[str(item2)], {'x': 1})
    self.assertEqual(snapshot[str(outer_block)], {'x': 1, 'y': 1})
    self.assertIsNone(snapshot.get(str(mediate_tuple)))

  def test_scope_snapshot_lambda_overwrite(self):
    inner_x = computation_building_blocks.Reference('x', tf.int32)
    inner_lambda = computation_building_blocks.Lambda('x', tf.int32, inner_x)
    outer_x = computation_building_blocks.Reference('x', tf.int32)
    call = computation_building_blocks.Call(inner_lambda, outer_x)
    outer_lambda = computation_building_blocks.Lambda('x', tf.int32, call)
    snapshot = transformations.scope_count_snapshot(outer_lambda)
    self.assertEqual(snapshot[str(inner_lambda)], {'x': 1})
    self.assertEqual(snapshot[str(outer_lambda)], {'x': 1})
    outer_call = computation_building_blocks.Call(inner_lambda, outer_x)
    third_lambda = computation_building_blocks.Lambda('x', tf.int32, outer_call)
    second_snapshot = transformations.scope_count_snapshot(third_lambda)
    self.assertEqual(str(third_lambda), '(x -> (x -> x)(x))')
    self.assertEqual(second_snapshot[str(inner_lambda)], {'x': 1})
    self.assertEqual(second_snapshot[str(outer_lambda)], {'x': 1})
    self.assertEqual(second_snapshot[str(third_lambda)], {'x': 1})

  def test_nested_lambda_block_overwrite_scope_snapshot(self):
    innermost_x = computation_building_blocks.Reference('x', tf.int32)
    inner_lambda = computation_building_blocks.Lambda('x', tf.int32,
                                                      innermost_x)
    second_x = computation_building_blocks.Reference('x', tf.int32)
    called_lambda = computation_building_blocks.Call(inner_lambda, second_x)
    block_input = computation_building_blocks.Reference('block_in', tf.int32)
    lower_block = computation_building_blocks.Block([('x', block_input)],
                                                    called_lambda)
    second_lambda = computation_building_blocks.Lambda('block_in', tf.int32,
                                                       lower_block)
    third_x = computation_building_blocks.Reference('x', tf.int32)
    second_call = computation_building_blocks.Call(second_lambda, third_x)
    final_input = computation_building_blocks.Data('test_data', tf.int32)
    last_block = computation_building_blocks.Block([('x', final_input)],
                                                   second_call)
    global_snapshot = transformations.scope_count_snapshot(last_block)
    self.assertEqual(
        str(last_block),
        '(let x=test_data in (block_in -> (let x=block_in in (x -> x)(x)))(x))')
    self.assertLen(global_snapshot, 4)
    self.assertEqual(global_snapshot[str(inner_lambda)], {'x': 1})
    self.assertEqual(global_snapshot[str(lower_block)], {'x': 1})
    self.assertEqual(global_snapshot[str(second_lambda)], {'block_in': 1})
    self.assertEqual(global_snapshot[str(last_block)], {'x': 1})

  def test_scope_snapshot_called_lambdas(self):
    comp = computation_building_blocks.Tuple(
        [computation_building_blocks.Data('test', tf.int32)])
    input1 = computation_building_blocks.Reference('input1',
                                                   comp.type_signature)
    first_level_call = computation_building_blocks.Call(
        computation_building_blocks.Lambda('input1', input1.type_signature,
                                           input1), comp)
    input2 = computation_building_blocks.Reference(
        'input2', first_level_call.type_signature)
    second_level_call = computation_building_blocks.Call(
        computation_building_blocks.Lambda('input2', input2.type_signature,
                                           input2), first_level_call)
    self.assertEqual(
        str(second_level_call),
        '(input2 -> input2)((input1 -> input1)(<test>))')
    global_snapshot = transformations.scope_count_snapshot(second_level_call)
    self.assertEqual(global_snapshot, {
        '(input2 -> input2)': {
            'input2': 1
        },
        '(input1 -> input1)': {
            'input1': 1
        }
    })

  def test_inline_conflicting_lambdas(self):
    comp = computation_building_blocks.Tuple(
        [computation_building_blocks.Data('test', tf.int32)])
    input1 = computation_building_blocks.Reference('input2',
                                                   comp.type_signature)
    first_level_call = computation_building_blocks.Call(
        computation_building_blocks.Lambda('input2', input1.type_signature,
                                           input1), comp)
    input2 = computation_building_blocks.Reference(
        'input2', first_level_call.type_signature)
    second_level_call = computation_building_blocks.Call(
        computation_building_blocks.Lambda('input2', input2.type_signature,
                                           input2), first_level_call)
    self.assertEqual(
        str(second_level_call),
        '(input2 -> input2)((input2 -> input2)(<test>))')
    lambda_reduced_comp = transformations.replace_called_lambdas_with_block(
        second_level_call)
    self.assertEqual(
        str(lambda_reduced_comp),
        '(let input2=(let input2=<test> in input2) in input2)')
    inlined = transformations.inline_blocks_with_n_referenced_locals(
        lambda_reduced_comp)
    self.assertEqual(str(inlined), '(let  in (let  in <test>))')

  def test_inline_conflicting_locals(self):
    arg_comp = computation_building_blocks.Reference('arg',
                                                     [tf.int32, tf.int32])
    selected = computation_building_blocks.Selection(arg_comp, index=0)
    internal_arg = computation_building_blocks.Reference('arg', tf.int32)
    block = computation_building_blocks.Block([('arg', selected)], internal_arg)
    lam = computation_building_blocks.Lambda('arg', arg_comp.type_signature,
                                             block)
    self.assertEqual(str(lam), '(arg -> (let arg=arg[0] in arg))')
    inlined = transformations.inline_blocks_with_n_referenced_locals(lam)
    self.assertEqual(str(inlined), '(arg -> (let  in arg[0]))')

  def test_simple_block_inlining(self):
    test_arg = computation_building_blocks.Data('test_data', tf.int32)
    result = computation_building_blocks.Reference('test_x',
                                                   test_arg.type_signature)
    simple_block = computation_building_blocks.Block([('test_x', test_arg)],
                                                     result)
    self.assertEqual(str(simple_block), '(let test_x=test_data in test_x)')
    inlined = transformations.inline_blocks_with_n_referenced_locals(
        simple_block)
    self.assertEqual(str(inlined), '(let  in test_data)')

  def test_no_inlining_if_referenced_twice(self):
    test_arg = computation_building_blocks.Data('test_data', tf.int32)
    ref1 = computation_building_blocks.Reference('test_x',
                                                 test_arg.type_signature)
    ref2 = computation_building_blocks.Reference('test_x',
                                                 test_arg.type_signature)
    result = computation_building_blocks.Tuple([ref1, ref2])
    simple_block = computation_building_blocks.Block([('test_x', test_arg)],
                                                     result)
    self.assertEqual(
        str(simple_block), '(let test_x=test_data in <test_x,test_x>)')
    inlined = transformations.inline_blocks_with_n_referenced_locals(
        simple_block)
    self.assertEqual(str(inlined), str(simple_block))

  def test_inlining_n_2(self):
    test_arg = computation_building_blocks.Data('test_data', tf.int32)
    ref1 = computation_building_blocks.Reference('test_x',
                                                 test_arg.type_signature)
    ref2 = computation_building_blocks.Reference('test_x',
                                                 test_arg.type_signature)
    result = computation_building_blocks.Tuple([ref1, ref2])
    simple_block = computation_building_blocks.Block([('test_x', test_arg)],
                                                     result)
    self.assertEqual(
        str(simple_block), '(let test_x=test_data in <test_x,test_x>)')
    inlined = transformations.inline_blocks_with_n_referenced_locals(
        simple_block, 2)
    self.assertEqual(str(inlined), '(let  in <test_data,test_data>)')

  def test_conflicting_name_resolved_inlining(self):
    red_herring_arg = computation_building_blocks.Reference(
        'redherring', tf.int32)
    used_arg = computation_building_blocks.Reference('used', tf.int32)
    ref = computation_building_blocks.Reference('x', used_arg.type_signature)
    lower_block = computation_building_blocks.Block([('x', used_arg)], ref)
    higher_block = computation_building_blocks.Block([('x', red_herring_arg)],
                                                     lower_block)
    self.assertEqual(
        str(higher_block), '(let x=redherring in (let x=used in x))')
    inlined = transformations.inline_blocks_with_n_referenced_locals(
        higher_block)
    self.assertEqual(str(inlined), '(let  in (let  in used))')

  def test_multiple_inline_for_nested_block(self):
    used1 = computation_building_blocks.Reference('used1', tf.int32)
    used2 = computation_building_blocks.Reference('used2', tf.int32)
    ref = computation_building_blocks.Reference('x', used1.type_signature)
    lower_block = computation_building_blocks.Block([('x', used1)], ref)
    higher_block = computation_building_blocks.Block([('used1', used2)],
                                                     lower_block)
    inlined = transformations.inline_blocks_with_n_referenced_locals(
        higher_block)
    self.assertEqual(
        str(higher_block), '(let used1=used2 in (let x=used1 in x))')
    self.assertEqual(str(inlined), '(let  in (let  in used2))')
    user_inlined_lower_block = computation_building_blocks.Block([('x', used1)],
                                                                 used1)
    user_inlined_higher_block = computation_building_blocks.Block(
        [('used1', used2)], user_inlined_lower_block)
    self.assertEqual(
        str(user_inlined_higher_block),
        '(let used1=used2 in (let x=used1 in used1))')
    inlined_noop = transformations.inline_blocks_with_n_referenced_locals(
        user_inlined_higher_block)
    self.assertEqual(str(inlined_noop), '(let used1=used2 in (let  in used1))')

  def test_conflicting_nested_name_inlining(self):
    innermost = computation_building_blocks.Reference('x', tf.int32)
    intermediate_arg = computation_building_blocks.Reference('y', tf.int32)
    item2 = computation_building_blocks.Block([('x', intermediate_arg)],
                                              innermost)
    item1 = computation_building_blocks.Reference('x', tf.int32)
    mediate_tuple = computation_building_blocks.Tuple([item1, item2])
    used = computation_building_blocks.Reference('used', tf.int32)
    used1 = computation_building_blocks.Reference('used1', tf.int32)
    outer_block = computation_building_blocks.Block([('x', used), ('y', used1)],
                                                    mediate_tuple)
    self.assertEqual(
        str(outer_block), '(let x=used,y=used1 in <x,(let x=y in x)>)')
    inlined = transformations.inline_blocks_with_n_referenced_locals(
        outer_block)
    self.assertEqual(str(inlined), '(let  in <used,(let  in used1)>)')


if __name__ == '__main__':
  absltest.main()
