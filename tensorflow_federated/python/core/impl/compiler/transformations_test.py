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

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.core.api import test_case
from tensorflow_federated.python.core.impl.compiler import building_block_analysis
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import test_utils as compiler_test_utils
from tensorflow_federated.python.core.impl.compiler import transformations
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_analysis
from tensorflow_federated.python.core.impl.types import type_serialization


class ToCallDominantTest(test_case.TestCase):

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


class CompileLocalComputationToTensorFlow(test_case.TestCase):

  def assert_compiles_to_tensorflow(
      self, comp: building_blocks.ComputationBuildingBlock):
    result = transformations.compile_local_computation_to_tensorflow(comp)
    if comp.type_signature.is_function():
      result.check_compiled_computation()
    else:
      result.check_call()
      result.function.check_compiled_computation()
    self.assert_types_equivalent(comp.type_signature, result.type_signature)

  def test_returns_tf_computation_with_functional_type_lambda_no_block(self):
    param = building_blocks.Reference('x', [('a', tf.int32), ('b', tf.float32)])
    sel = building_blocks.Selection(source=param, index=0)
    tup = building_blocks.Struct([sel, sel, sel])
    lam = building_blocks.Lambda(param.name, param.type_signature, tup)
    self.assert_compiles_to_tensorflow(lam)

  def test_returns_tf_computation_with_functional_type_lambda_with_block(self):
    param = building_blocks.Reference('x', [('a', tf.int32), ('b', tf.float32)])
    block_to_param = building_blocks.Block([('x', param)], param)
    lam = building_blocks.Lambda(param.name, param.type_signature,
                                 block_to_param)
    self.assert_compiles_to_tensorflow(lam)

  def test_returns_tf_computation_with_functional_type_block_to_lambda_no_block(
      self):
    concrete_int_type = computation_types.TensorType(tf.int32)
    param = building_blocks.Reference('x', tf.float32)
    lam = building_blocks.Lambda(param.name, param.type_signature, param)
    unused_int = building_block_factory.create_tensorflow_constant(
        concrete_int_type, 1)
    blk_to_lam = building_blocks.Block([('y', unused_int)], lam)
    self.assert_compiles_to_tensorflow(blk_to_lam)

  def test_returns_tf_computation_with_functional_type_block_to_lambda_with_block(
      self):
    concrete_int_type = computation_types.TensorType(tf.int32)
    param = building_blocks.Reference('x', tf.float32)
    block_to_param = building_blocks.Block([('x', param)], param)
    lam = building_blocks.Lambda(param.name, param.type_signature,
                                 block_to_param)
    unused_int = building_block_factory.create_tensorflow_constant(
        concrete_int_type, 1)
    blk_to_lam = building_blocks.Block([('y', unused_int)], lam)
    self.assert_compiles_to_tensorflow(blk_to_lam)

  def test_returns_tf_computation_block_with_compiled_comp(self):
    concrete_int_type = computation_types.TensorType(tf.int32)
    tf_identity = building_block_factory.create_compiled_identity(
        concrete_int_type)
    unused_int = building_block_factory.create_tensorflow_constant(
        concrete_int_type, 1)
    block_to_id = building_blocks.Block([('x', unused_int)], tf_identity)
    self.assert_compiles_to_tensorflow(block_to_id)

  def test_returns_tf_computation_ompiled_comp(self):
    concrete_int_type = computation_types.TensorType(tf.int32)
    tf_identity = building_block_factory.create_compiled_identity(
        concrete_int_type)
    self.assert_compiles_to_tensorflow(tf_identity)

  def test_returns_called_tf_computation_with_truct(self):
    constant_tuple_type = computation_types.StructType([tf.int32, tf.float32])
    constant_tuple = building_block_factory.create_tensorflow_constant(
        constant_tuple_type, 1)
    sel = building_blocks.Selection(source=constant_tuple, index=0)
    tup = building_blocks.Struct([sel, sel, sel])
    self.assert_compiles_to_tensorflow(tup)

  def test_deduplicates_by_counting_ops(self):

    def _construct_inlined_tuple(k):
      constant_tuple_type = computation_types.TensorType(tf.int32)
      concrete_int = building_block_factory.create_tensorflow_constant(
          constant_tuple_type, 1)
      first_tf_fn = building_block_factory.create_tensorflow_binary_operator(
          tf.add, concrete_int.type_signature)
      call = building_blocks.Call(
          first_tf_fn, building_blocks.Struct([concrete_int, concrete_int]))
      for _ in range(k):
        # Simulating large TF computation
        call = building_blocks.Call(first_tf_fn,
                                    building_blocks.Struct([call, call]))
      return building_blocks.Struct([call, call])

    def _count_ops_parameterized_by_layers(k):
      inlined_tuple = _construct_inlined_tuple(k)
      with_deduping = transformations.compile_local_computation_to_tensorflow(
          inlined_tuple)
      num_ops_with_deduping = tree_analysis.count_tensorflow_ops_under(
          with_deduping)
      without_deduping = transformations.compile_local_computation_to_tensorflow(
          inlined_tuple, deduplicate=False)
      num_ops_without_deduping = tree_analysis.count_tensorflow_ops_under(
          without_deduping)
      return num_ops_with_deduping, num_ops_without_deduping

    num_ops_deduped_0_layers, num_ops_0_layers = _count_ops_parameterized_by_layers(
        0)
    num_ops_deduped_1_layers, num_ops_1_layers = _count_ops_parameterized_by_layers(
        1)
    num_ops_deduped_2_layers, num_ops_2_layers = _count_ops_parameterized_by_layers(
        2)
    num_ops_deduped_3_layers, num_ops_3_layers = _count_ops_parameterized_by_layers(
        3)

    # asserting that block ops are linear in k.
    self.assertEqual(num_ops_deduped_1_layers - num_ops_deduped_0_layers,
                     num_ops_deduped_2_layers - num_ops_deduped_1_layers)
    self.assertEqual(num_ops_deduped_3_layers - num_ops_deduped_2_layers,
                     num_ops_deduped_2_layers - num_ops_deduped_1_layers)

    # asserting that tuple ops are exponential in k.
    first_factor = (num_ops_2_layers - num_ops_1_layers) / (
        num_ops_1_layers - num_ops_0_layers)
    second_factor = (num_ops_3_layers - num_ops_2_layers) / (
        num_ops_2_layers - num_ops_1_layers)
    self.assertEqual(first_factor, second_factor)

  def test_passes_on_tf(self):
    tf_comp = building_block_factory.create_compiled_identity(
        computation_types.TensorType(tf.int32))
    transformed = transformations.compile_local_computation_to_tensorflow(
        tf_comp)
    self.assertEqual(tf_comp, transformed)

  def test_raises_on_xla(self):
    function_type = computation_types.FunctionType(
        computation_types.TensorType(tf.int32),
        computation_types.TensorType(tf.int32))
    empty_xla_computation_proto = pb.Computation(
        type=type_serialization.serialize_type(function_type), xla=pb.Xla())

    compiled_comp = building_blocks.CompiledComputation(
        proto=empty_xla_computation_proto)

    with self.assertRaises(transformations.XlaToTensorFlowError):
      transformations.compile_local_computation_to_tensorflow(compiled_comp)

  def test_generates_tf_with_lambda(self):
    ref_to_x = building_blocks.Reference(
        'x', computation_types.StructType([tf.int32, tf.float32]))
    identity_lambda = building_blocks.Lambda(ref_to_x.name,
                                             ref_to_x.type_signature, ref_to_x)
    self.assert_compiles_to_tensorflow(identity_lambda)

  def test_generates_tf_with_block(self):
    ref_to_x = building_blocks.Reference(
        'x', computation_types.StructType([tf.int32, tf.float32]))
    identity_lambda = building_blocks.Lambda(ref_to_x.name,
                                             ref_to_x.type_signature, ref_to_x)
    tf_zero = building_block_factory.create_tensorflow_constant(
        computation_types.StructType([tf.int32, tf.float32]), 0)
    ref_to_z = building_blocks.Reference('z', [tf.int32, tf.float32])
    called_lambda_on_z = building_blocks.Call(identity_lambda, ref_to_z)
    blk = building_blocks.Block([('z', tf_zero)], called_lambda_on_z)
    self.assert_compiles_to_tensorflow(blk)

  def test_generates_tf_with_sequence_type(self):
    ref_to_x = building_blocks.Reference(
        'x', computation_types.SequenceType([tf.int32, tf.float32]))
    identity_lambda = building_blocks.Lambda(ref_to_x.name,
                                             ref_to_x.type_signature, ref_to_x)
    self.assert_compiles_to_tensorflow(identity_lambda)


class CompileLocalSubcomputationsToTensorFlowTest(test_case.TestCase):

  def test_leaves_federated_comp_alone(self):
    ref_to_federated_x = building_blocks.Reference(
        'x', computation_types.FederatedType(tf.int32, placements.SERVER))
    identity_lambda = building_blocks.Lambda(ref_to_federated_x.name,
                                             ref_to_federated_x.type_signature,
                                             ref_to_federated_x)
    transformed = transformations.compile_local_subcomputations_to_tensorflow(
        identity_lambda)
    self.assertEqual(transformed, identity_lambda)

  def test_compiles_lambda_under_federated_comp_to_tf(self):
    ref_to_x = building_blocks.Reference(
        'x', computation_types.StructType([tf.int32, tf.float32]))
    identity_lambda = building_blocks.Lambda(ref_to_x.name,
                                             ref_to_x.type_signature, ref_to_x)
    federated_data = building_blocks.Data(
        'a',
        computation_types.FederatedType(
            computation_types.StructType([tf.int32, tf.float32]),
            placements.SERVER))
    applied = building_block_factory.create_federated_apply(
        identity_lambda, federated_data)

    transformed = transformations.compile_local_subcomputations_to_tensorflow(
        applied)

    self.assertIsInstance(transformed, building_blocks.Call)
    self.assertIsInstance(transformed.function, building_blocks.Intrinsic)
    self.assertIsInstance(transformed.argument[0],
                          building_blocks.CompiledComputation)
    self.assertEqual(transformed.argument[1], federated_data)
    self.assertEqual(transformed.argument[0].type_signature,
                     identity_lambda.type_signature)

  def test_leaves_local_comp_with_unbound_reference_alone(self):
    ref_to_x = building_blocks.Reference('x', [tf.int32, tf.float32])
    ref_to_z = building_blocks.Reference('z', [tf.int32, tf.float32])
    lambda_with_unbound_ref = building_blocks.Lambda(ref_to_x.name,
                                                     ref_to_x.type_signature,
                                                     ref_to_z)
    transformed = transformations.compile_local_subcomputations_to_tensorflow(
        lambda_with_unbound_ref)

    self.assertEqual(transformed, lambda_with_unbound_ref)


class ToDedupedCallDominantTest(test_case.TestCase):

  def test_handles_called_lambda_returning_function(self):
    lower_level_lambda = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    higher_level_lambda = building_blocks.Lambda('y', tf.int32,
                                                 lower_level_lambda)

    call_dominant_rep = transformations.to_deduped_call_dominant(
        higher_level_lambda)

    self.assertRegexMatch(call_dominant_rep.compact_representation(),
                          [r'\(_([a-z]{3})1 -> \(_(\1)2 -> _(\1)2\)\)'])

  def test_handles_block_returning_function(self):
    lower_level_lambda = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    blk = building_blocks.Block([], lower_level_lambda)

    call_dominant_rep = transformations.to_deduped_call_dominant(blk)
    self.assertRegexMatch(call_dominant_rep.compact_representation(),
                          [r'\(_([a-z]{3})1 -> _(\1)1\)'])

  def test_merges_nested_blocks(self):
    data = building_blocks.Data('a', tf.int32)
    ref_to_x = building_blocks.Reference('x', tf.int32)
    blk1 = building_blocks.Block([('x', data)], ref_to_x)
    blk2 = building_blocks.Block([('x', blk1)], ref_to_x)

    call_dominant_rep = transformations.to_deduped_call_dominant(blk2)

    self.assertEqual(call_dominant_rep.formatted_representation(),
                     data.formatted_representation())

  def test_extracts_called_intrinsics_to_block(self):
    called_aggregate = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    tuple_holding_aggregate = building_blocks.Struct([called_aggregate])
    sel_from_tuple = building_blocks.Selection(
        source=tuple_holding_aggregate, index=0)
    lambda_to_sel = building_blocks.Lambda('x', tf.int32, sel_from_tuple)

    call_dominant_rep = transformations.to_deduped_call_dominant(lambda_to_sel)

    call_dominant_rep.check_lambda()
    call_dominant_rep.result.check_block()
    self.assertLen(call_dominant_rep.result.locals, 1)
    self.assertTrue(
        building_block_analysis.is_called_intrinsic(
            call_dominant_rep.result.locals[0][1],
            intrinsic_defs.FEDERATED_AGGREGATE.uri))

  def test_deduplicates_called_intrinsics(self):
    called_aggregate1 = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    called_aggregate2 = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    tuple_holding_aggregates = building_blocks.Struct(
        [called_aggregate1, called_aggregate2])
    lambda_to_tup = building_blocks.Lambda('x', tf.int32,
                                           tuple_holding_aggregates)

    call_dominant_rep = transformations.to_deduped_call_dominant(lambda_to_tup)

    call_dominant_rep.check_lambda()
    call_dominant_rep.result.check_block()
    self.assertLen(call_dominant_rep.result.locals, 1)
    self.assertTrue(
        building_block_analysis.is_called_intrinsic(
            call_dominant_rep.result.locals[0][1],
            intrinsic_defs.FEDERATED_AGGREGATE.uri))

  def test_hoists_aggregations_packed_in_tuple(self):
    called_aggregate1 = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c',
        value_type=tf.int32)
    called_aggregate2 = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c',
        value_type=tf.float32)
    tuple_holding_aggregates = building_blocks.Struct(
        [called_aggregate1, called_aggregate2])
    lambda_to_tuple = building_blocks.Lambda('x', tf.int32,
                                             tuple_holding_aggregates)

    call_dominant_rep = transformations.to_deduped_call_dominant(
        lambda_to_tuple)

    call_dominant_rep.check_lambda()
    call_dominant_rep.result.check_block()
    self.assertLen(call_dominant_rep.result.locals, 2)
    self.assertTrue(
        building_block_analysis.is_called_intrinsic(
            call_dominant_rep.result.locals[0][1],
            intrinsic_defs.FEDERATED_AGGREGATE.uri))
    self.assertTrue(
        building_block_analysis.is_called_intrinsic(
            call_dominant_rep.result.locals[1][1],
            intrinsic_defs.FEDERATED_AGGREGATE.uri))

  def test_handles_lambda_with_lambda_parameter(self):
    int_identity_lambda = building_blocks.Lambda(
        'x', tf.int32, building_blocks.Reference('x', tf.int32))
    ref_to_fn_and_int = building_blocks.Reference(
        'y',
        computation_types.StructType([
            int_identity_lambda.type_signature,
            computation_types.TensorType(tf.int32)
        ]))
    fn = building_blocks.Selection(ref_to_fn_and_int, index=0)
    arg = building_blocks.Selection(ref_to_fn_and_int, index=1)
    called_fn = building_blocks.Call(fn, arg)
    lambda_accepting_fn = building_blocks.Lambda(
        ref_to_fn_and_int.name, ref_to_fn_and_int.type_signature, called_fn)
    ref_to_int = building_blocks.Reference('z', tf.int32)
    arg_tuple = building_blocks.Struct([int_identity_lambda, ref_to_int])
    called_lambda_with_fn = building_blocks.Call(lambda_accepting_fn, arg_tuple)
    lambda_accepting_int = building_blocks.Lambda(ref_to_int.name,
                                                  ref_to_int.type_signature,
                                                  called_lambda_with_fn)

    call_dominant_rep = transformations.to_deduped_call_dominant(
        lambda_accepting_int)

    call_dominant_rep.check_lambda()
    param_name = call_dominant_rep.parameter_name
    expected = building_blocks.Lambda(
        param_name, tf.int32, building_blocks.Reference(param_name, tf.int32))
    self.assertEqual(call_dominant_rep.formatted_representation(),
                     expected.formatted_representation())


class ForceAlignAndSplitByIntrinsicTest(test_case.TestCase):

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
      self.assert_types_equivalent(comp.parameter_type, before.parameter_type)
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
      self.assert_types_equivalent(comp.parameter_type,
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
    self.skipTest('b/203780753')
    federated_broadcast = compiler_test_utils.create_whimsy_called_federated_broadcast(
    )
    called_intrinsics = building_blocks.Struct([federated_broadcast])
    comp = building_blocks.Lambda(None, None, called_intrinsics)
    call = building_block_factory.create_null_federated_broadcast()
    self.assert_splits_on(comp, call)

  def test_splits_on_selected_intrinsic_broadcast(self):
    federated_broadcast = compiler_test_utils.create_whimsy_called_federated_broadcast(
    )
    called_intrinsics = building_blocks.Struct([federated_broadcast])
    comp = building_blocks.Lambda('a', tf.int32, called_intrinsics)
    call = building_block_factory.create_null_federated_broadcast()
    self.assert_splits_on(comp, call)

  def test_splits_on_selected_intrinsic_nested_in_tuple_broadcast(self):
    first_broadcast = compiler_test_utils.create_whimsy_called_federated_broadcast(
    )
    packed_broadcast = building_blocks.Struct([
        building_blocks.Data('a', computation_types.at_server(tf.int32)),
        first_broadcast
    ])
    sel = building_blocks.Selection(packed_broadcast, index=0)
    second_broadcast = building_block_factory.create_federated_broadcast(sel)
    result = transformations.to_deduped_call_dominant(second_broadcast)
    comp = building_blocks.Lambda('a', tf.int32, result)
    call = building_block_factory.create_null_federated_broadcast()
    self.assert_splits_on(comp, call)

  def test_splits_on_multiple_of_selected_intrinsic_broadcast(self):
    federated_broadcast = compiler_test_utils.create_whimsy_called_federated_broadcast(
    )
    called_intrinsics = building_blocks.Struct([
        federated_broadcast,
        federated_broadcast,
    ])
    comp = building_blocks.Lambda('a', tf.int32, called_intrinsics)
    call = building_block_factory.create_null_federated_broadcast()
    self.assert_splits_on(comp, call)

  def test_splits_on_selected_intrinsic_aggregate(self):
    federated_aggregate = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    called_intrinsics = building_blocks.Struct([federated_aggregate])
    comp = building_blocks.Lambda('d', tf.int32, called_intrinsics)
    call = building_block_factory.create_null_federated_aggregate()
    self.assert_splits_on(comp, call)

  def test_splits_on_multiple_of_selected_intrinsic_aggregate(self):
    federated_aggregate = compiler_test_utils.create_whimsy_called_federated_aggregate(
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
    federated_secure_sum_bitwidth = compiler_test_utils.create_whimsy_called_federated_secure_sum_bitwidth(
    )
    called_intrinsics = building_blocks.Struct([federated_secure_sum_bitwidth])
    comp = building_blocks.Lambda('a', tf.int32, called_intrinsics)
    call = building_block_factory.create_null_federated_secure_sum_bitwidth()
    self.assert_splits_on(comp, call)

  def test_splits_on_multiple_of_selected_intrinsic_secure_sum_bitwidths(self):
    federated_secure_sum_bitwidth = compiler_test_utils.create_whimsy_called_federated_secure_sum_bitwidth(
    )
    called_intrinsics = building_blocks.Struct([
        federated_secure_sum_bitwidth,
        federated_secure_sum_bitwidth,
    ])
    comp = building_blocks.Lambda('a', tf.int32, called_intrinsics)
    call = building_block_factory.create_null_federated_secure_sum_bitwidth()
    self.assert_splits_on(comp, call)

  def test_removes_selected_intrinsic_leaving_remaining_intrinsic(self):
    federated_aggregate = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    federated_secure_sum_bitwidth = compiler_test_utils.create_whimsy_called_federated_secure_sum_bitwidth(
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
    federated_aggregate = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    federated_secure_sum_bitwidth = compiler_test_utils.create_whimsy_called_federated_secure_sum_bitwidth(
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
    federated_aggregate = compiler_test_utils.create_whimsy_called_federated_aggregate(
        accumulate_parameter_name='a',
        merge_parameter_name='b',
        report_parameter_name='c')
    federated_secure_sum_bitwidth = compiler_test_utils.create_whimsy_called_federated_secure_sum_bitwidth(
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
    federated_aggregate = compiler_test_utils.create_whimsy_called_federated_aggregate(
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
  test_case.main()
