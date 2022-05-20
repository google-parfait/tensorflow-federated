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
import tensorflow as tf

from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_block_test_utils
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import transformations
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_test_utils


class ToCallDominantTest(absltest.TestCase):

  def assert_compact_representations_equal(
      self, actual: building_blocks.ComputationBuildingBlock,
      expected: building_blocks.ComputationBuildingBlock):
    """Asserts that two building blocks have the same compact representation."""
    self.assertEqual(actual.compact_representation(),
                     expected.compact_representation())

  def test_inlines_references(self):
    int_type = computation_types.to_type(tf.int32)
    int_ref = lambda name: building_blocks.Reference(name, int_type)
    int_fn = lambda name, result: building_blocks.Lambda(name, int_type, result)
    before = int_fn(
        'x',
        building_blocks.Block([
            ('y', int_ref('x')),
            ('z', int_ref('y')),
        ], int_ref('z')))
    after = transformations.to_call_dominant(before)
    expected = int_fn('x', int_ref('x'))
    self.assert_compact_representations_equal(after, expected)

  def test_inlines_selections(self):
    int_type = computation_types.to_type(tf.int32)
    structed = computation_types.StructType([int_type])
    double = computation_types.StructType([structed])
    bb = building_blocks
    before = bb.Lambda(
        'x', double,
        bb.Block([
            ('y', bb.Selection(bb.Reference('x', double), index=0)),
            ('z', bb.Selection(bb.Reference('y', structed), index=0)),
        ], bb.Reference('z', int_type)))
    after = transformations.to_call_dominant(before)
    expected = bb.Lambda(
        'x', double,
        bb.Selection(bb.Selection(bb.Reference('x', double), index=0), index=0))
    self.assert_compact_representations_equal(after, expected)

  def test_inlines_structs(self):
    int_type = computation_types.to_type(tf.int32)
    structed = computation_types.StructType([int_type])
    double = computation_types.StructType([structed])
    bb = building_blocks
    before = bb.Lambda(
        'x', int_type,
        bb.Block([
            ('y', bb.Struct([building_blocks.Reference('x', int_type)])),
            ('z', bb.Struct([building_blocks.Reference('y', structed)])),
        ], bb.Reference('z', double)))
    after = transformations.to_call_dominant(before)
    expected = bb.Lambda('x', int_type,
                         bb.Struct([bb.Struct([bb.Reference('x', int_type)])]))
    self.assert_compact_representations_equal(after, expected)

  def test_inlines_selection_from_struct(self):
    int_type = computation_types.to_type(tf.int32)
    bb = building_blocks
    before = bb.Lambda(
        'x', int_type,
        bb.Selection(bb.Struct([bb.Reference('x', int_type)]), index=0))
    after = transformations.to_call_dominant(before)
    expected = bb.Lambda('x', int_type, bb.Reference('x', int_type))
    self.assert_compact_representations_equal(after, expected)

  def test_creates_binding_for_each_call(self):
    int_type = computation_types.to_type(tf.int32)
    int_to_int_type = computation_types.FunctionType(int_type, int_type)
    bb = building_blocks
    int_to_int_fn = bb.Data('ext', int_to_int_type)
    before = bb.Lambda(
        'x', int_type,
        bb.Call(int_to_int_fn,
                bb.Call(int_to_int_fn, bb.Reference('x', int_type))))
    after = transformations.to_call_dominant(before)
    expected = bb.Lambda(
        'x', int_type,
        bb.Block([
            ('_var1', bb.Call(int_to_int_fn, bb.Reference('x', int_type))),
            ('_var2', bb.Call(int_to_int_fn, bb.Reference('_var1', int_type)))
        ], bb.Reference('_var2', int_type)))
    self.assert_compact_representations_equal(after, expected)

  def test_evaluates_called_lambdas(self):
    int_type = computation_types.to_type(tf.int32)
    int_to_int_type = computation_types.FunctionType(int_type, int_type)
    int_thunk_type = computation_types.FunctionType(None, int_type)
    bb = building_blocks
    int_to_int_fn = bb.Data('ext', int_to_int_type)

    # -> (let result = ext(x) in (-> result))
    # Each call of the outer lambda should create a single binding, with
    # calls to the inner lambda repeatedly returning references to the binding.
    higher_fn = bb.Lambda(
        None, None,
        bb.Block(
            [('result', bb.Call(int_to_int_fn, bb.Reference('x', int_type)))],
            bb.Lambda(None, None, bb.Reference('result', int_type))))
    block_locals = [
        ('fn', higher_fn),
        # fn = -> (let result = ext(x) in (-> result))
        ('get_val1', bb.Call(bb.Reference('fn', higher_fn.type_signature))),
        # _var2 = ext(x)
        # get_val1 = -> _var2
        ('get_val2', bb.Call(bb.Reference('fn', higher_fn.type_signature))),
        # _var3 = ext(x)
        # get_val2 = -> _var3
        ('val11', bb.Call(bb.Reference('get_val1', int_thunk_type))),
        # val11 = _var2
        ('val12', bb.Call(bb.Reference('get_val1', int_thunk_type))),
        # val12 = _var2
        ('val2', bb.Call(bb.Reference('get_val2', int_thunk_type))),
        # val2 = _var3
    ]
    before = bb.Lambda(
        'x',
        int_type,
        bb.Block(
            block_locals,
            # <_var2, _var2, _var3>
            bb.Struct([
                bb.Reference('val11', int_type),
                bb.Reference('val12', int_type),
                bb.Reference('val2', int_type)
            ])))
    after = transformations.to_call_dominant(before)
    expected = bb.Lambda(
        'x', int_type,
        bb.Block([
            ('_var2', bb.Call(int_to_int_fn, bb.Reference('x', int_type))),
            ('_var3', bb.Call(int_to_int_fn, bb.Reference('x', int_type))),
        ],
                 bb.Struct([
                     bb.Reference('_var2', int_type),
                     bb.Reference('_var2', int_type),
                     bb.Reference('_var3', int_type)
                 ])))
    self.assert_compact_representations_equal(after, expected)

  def test_creates_block_for_non_lambda(self):
    bb = building_blocks
    int_type = computation_types.TensorType(tf.int32)
    two_int_type = computation_types.StructType([(None, int_type),
                                                 (None, int_type)])
    get_two_int_type = computation_types.FunctionType(None, two_int_type)
    call_ext = bb.Call(bb.Data('ext', get_two_int_type))
    before = bb.Selection(call_ext, index=0)
    after = transformations.to_call_dominant(before)
    expected = bb.Block([
        ('_var1', call_ext),
    ], bb.Selection(bb.Reference('_var1', two_int_type), index=0))
    self.assert_compact_representations_equal(after, expected)

  def test_call_to_higher_order_external_allowed(self):
    bb = building_blocks
    types = computation_types
    int_type = types.TensorType(tf.int32)
    int_to_int_type = types.FunctionType(int_type, int_type)
    int_to_int_to_int_type = types.FunctionType(int_to_int_type, int_type)
    call_ext = bb.Call(
        bb.Data('call_with_one', int_to_int_to_int_type),
        bb.Lambda('x', int_type, bb.Data('num', int_type)))
    after = transformations.to_call_dominant(call_ext)
    after.check_block()
    self.assertLen(after.locals, 1)
    (ref_name, bound_call) = after.locals[0]
    self.assertEqual(bound_call.compact_representation(),
                     call_ext.compact_representation())
    expected_result = bb.Reference(ref_name, call_ext.type_signature)
    self.assert_compact_representations_equal(after.result, expected_result)


class ForceAlignAndSplitByIntrinsicTest(absltest.TestCase):

  def assert_splits_on(self, comp, calls):
    """Asserts that `force_align_and_split_by_intrinsics` removes intrinsics."""
    if not isinstance(calls, list):
      calls = [calls]
    uris = [call.function.uri for call in calls]
    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, calls)

    # Ensure that the resulting computations no longer contain the split
    # intrinsics.
    self.assertFalse(tree_analysis.contains_called_intrinsic(before, uris))
    self.assertFalse(tree_analysis.contains_called_intrinsic(after, uris))
    # Removal isn't interesting to test for if it wasn't there to begin with.
    self.assertTrue(tree_analysis.contains_called_intrinsic(comp, uris))

    if comp.parameter_type is not None:
      type_test_utils.assert_types_equivalent(comp.parameter_type,
                                              before.parameter_type)
    else:
      self.assertIsNone(before.parameter_type)
    # THere must be one parameter for each intrinsic in `calls`.
    before.type_signature.result.check_struct()
    self.assertLen(before.type_signature.result, len(calls))

    # Check that `after`'s parameter is a structure like:
    # {
    #   'original_arg': comp.parameter_type, (if present)
    #   'intrinsic_results': [...],
    # }
    after.parameter_type.check_struct()
    if comp.parameter_type is not None:
      self.assertLen(after.parameter_type, 2)
      type_test_utils.assert_types_equivalent(comp.parameter_type,
                                              after.parameter_type.original_arg)
    else:
      self.assertLen(after.parameter_type, 1)
    # There must be one result for each intrinsic in `calls`.
    self.assertLen(after.parameter_type.intrinsic_results, len(calls))

    # Check that each pair of (param, result) is a valid type substitution
    # for the intrinsic in question.
    for i in range(len(calls)):
      concrete_signature = computation_types.FunctionType(
          before.type_signature.result[i],
          after.parameter_type.intrinsic_results[i])
      abstract_signature = calls[i].function.intrinsic_def().type_signature
      type_analysis.check_concrete_instance_of(concrete_signature,
                                               abstract_signature)

  def test_cannot_split_on_chained_intrinsic(self):
    int_type = computation_types.TensorType(tf.int32)
    client_int_type = computation_types.at_clients(int_type)
    int_ref = lambda name: building_blocks.Reference(name, int_type)
    client_int_ref = (
        lambda name: building_blocks.Reference(name, client_int_type))
    body = building_blocks.Block([
        ('a',
         building_block_factory.create_federated_map(
             building_blocks.Lambda('p1', int_type, int_ref('p1')),
             client_int_ref('param'))),
        ('b',
         building_block_factory.create_federated_map(
             building_blocks.Lambda('p2', int_type, int_ref('p2')),
             client_int_ref('a'))),
    ], client_int_ref('b'))
    comp = building_blocks.Lambda('param', client_int_type, body)
    with self.assertRaises(transformations._NonAlignableAlongIntrinsicError):
      transformations.force_align_and_split_by_intrinsics(
          comp, [building_block_factory.create_null_federated_map()])

  def test_splits_on_intrinsic_noarg_function(self):
    federated_broadcast = building_block_test_utils.create_whimsy_called_federated_broadcast(
    )
    called_intrinsics = building_blocks.Struct([federated_broadcast])
    comp = building_blocks.Lambda(None, None, called_intrinsics)
    call = building_block_factory.create_null_federated_broadcast()
    self.assert_splits_on(comp, call)

  def test_splits_on_selected_intrinsic_broadcast(self):
    federated_broadcast = building_block_test_utils.create_whimsy_called_federated_broadcast(
    )
    called_intrinsics = building_blocks.Struct([federated_broadcast])
    comp = building_blocks.Lambda('a', tf.int32, called_intrinsics)
    call = building_block_factory.create_null_federated_broadcast()
    self.assert_splits_on(comp, call)

  def test_splits_on_selected_intrinsic_nested_in_tuple_broadcast(self):
    first_broadcast = building_block_test_utils.create_whimsy_called_federated_broadcast(
    )
    packed_broadcast = building_blocks.Struct([
        building_blocks.Data('a', computation_types.at_server(tf.int32)),
        first_broadcast
    ])
    sel = building_blocks.Selection(packed_broadcast, index=0)
    second_broadcast = building_block_factory.create_federated_broadcast(sel)
    result = transformations.to_call_dominant(second_broadcast)
    comp = building_blocks.Lambda('a', tf.int32, result)
    call = building_block_factory.create_null_federated_broadcast()
    self.assert_splits_on(comp, call)

  def test_splits_on_multiple_of_selected_intrinsic_broadcast(self):
    federated_broadcast = building_block_test_utils.create_whimsy_called_federated_broadcast(
    )
    called_intrinsics = building_blocks.Struct([
        federated_broadcast,
        federated_broadcast,
    ])
    comp = building_blocks.Lambda('a', tf.int32, called_intrinsics)
    call = building_block_factory.create_null_federated_broadcast()
    self.assert_splits_on(comp, call)

  def test_splits_on_selected_intrinsic_aggregate(self):
    federated_aggregate = building_block_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    called_intrinsics = building_blocks.Struct([federated_aggregate])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    call = building_block_factory.create_null_federated_aggregate()
    self.assert_splits_on(comp, call)

  def test_splits_on_multiple_of_selected_intrinsic_aggregate(self):
    federated_aggregate = building_block_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    called_intrinsics = building_blocks.Struct([
        federated_aggregate,
        federated_aggregate,
    ])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    call = building_block_factory.create_null_federated_aggregate()
    self.assert_splits_on(comp, call)

  def test_splits_on_selected_intrinsic_secure_sum_bitwidth(self):
    federated_secure_sum_bitwidth = building_block_test_utils.create_whimsy_called_federated_secure_sum_bitwidth(
    )
    called_intrinsics = building_blocks.Struct([federated_secure_sum_bitwidth])
    comp = building_blocks.Lambda('a', tf.int32, called_intrinsics)
    call = building_block_factory.create_null_federated_secure_sum_bitwidth()
    self.assert_splits_on(comp, call)

  def test_splits_on_multiple_of_selected_intrinsic_secure_sum_bitwidths(self):
    federated_secure_sum_bitwidth = building_block_test_utils.create_whimsy_called_federated_secure_sum_bitwidth(
    )
    called_intrinsics = building_blocks.Struct([
        federated_secure_sum_bitwidth,
        federated_secure_sum_bitwidth,
    ])
    comp = building_blocks.Lambda('a', tf.int32, called_intrinsics)
    call = building_block_factory.create_null_federated_secure_sum_bitwidth()
    self.assert_splits_on(comp, call)

  def test_removes_selected_intrinsic_leaving_remaining_intrinsic(self):
    federated_aggregate = building_block_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    federated_secure_sum_bitwidth = building_block_test_utils.create_whimsy_called_federated_secure_sum_bitwidth(
    )
    called_intrinsics = building_blocks.Struct([
        federated_aggregate,
        federated_secure_sum_bitwidth,
    ])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    null_aggregate = building_block_factory.create_null_federated_aggregate()
    secure_sum_bitwidth_uri = federated_secure_sum_bitwidth.function.uri
    aggregate_uri = null_aggregate.function.uri
    before, after = transformations.force_align_and_split_by_intrinsics(
        comp, [null_aggregate])
    self.assertTrue(
        tree_analysis.contains_called_intrinsic(comp, secure_sum_bitwidth_uri))
    self.assertTrue(
        tree_analysis.contains_called_intrinsic(comp, aggregate_uri))
    self.assertFalse(
        tree_analysis.contains_called_intrinsic(before, aggregate_uri))
    self.assertFalse(
        tree_analysis.contains_called_intrinsic(after, aggregate_uri))
    self.assertTrue(
        tree_analysis.contains_called_intrinsic(before,
                                                secure_sum_bitwidth_uri) or
        tree_analysis.contains_called_intrinsic(after, secure_sum_bitwidth_uri))

  def test_splits_on_two_intrinsics(self):
    federated_aggregate = building_block_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    federated_secure_sum_bitwidth = building_block_test_utils.create_whimsy_called_federated_secure_sum_bitwidth(
    )
    called_intrinsics = building_blocks.Struct([
        federated_aggregate,
        federated_secure_sum_bitwidth,
    ])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    self.assert_splits_on(comp, [
        building_block_factory.create_null_federated_aggregate(),
        building_block_factory.create_null_federated_secure_sum_bitwidth()
    ])

  def test_splits_on_multiple_instances_of_two_intrinsics(self):
    federated_aggregate = building_block_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    federated_secure_sum_bitwidth = building_block_test_utils.create_whimsy_called_federated_secure_sum_bitwidth(
    )
    called_intrinsics = building_blocks.Struct([
        federated_aggregate,
        federated_aggregate,
        federated_secure_sum_bitwidth,
        federated_secure_sum_bitwidth,
    ])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    self.assert_splits_on(comp, [
        building_block_factory.create_null_federated_aggregate(),
        building_block_factory.create_null_federated_secure_sum_bitwidth()
    ])

  def test_splits_even_when_selected_intrinsic_is_not_present(self):
    federated_aggregate = building_block_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    called_intrinsics = building_blocks.Struct([federated_aggregate])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    transformations.force_align_and_split_by_intrinsics(comp, [
        building_block_factory.create_null_federated_aggregate(),
        building_block_factory.create_null_federated_secure_sum_bitwidth(),
    ])


if __name__ == '__main__':
  absltest.main()
