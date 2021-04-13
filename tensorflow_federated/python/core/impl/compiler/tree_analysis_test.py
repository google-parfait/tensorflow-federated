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

import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import building_block_analysis
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import test_utils
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


class TestCheckContainsOnlyReducibleIntrinsics(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      tree_analysis.check_contains_only_reducible_intrinsics(None)

  def test_passes_with_federated_map(self):
    intrinsic = building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_MAP.uri,
        computation_types.FunctionType([
            computation_types.FunctionType(tf.int32, tf.float32),
            computation_types.FederatedType(tf.int32, placements.CLIENTS)
        ], computation_types.FederatedType(tf.float32, placements.CLIENTS)))
    tree_analysis.check_contains_only_reducible_intrinsics(intrinsic)

  def test_raises_with_federated_mean(self):
    intrinsic = building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_MEAN.uri,
        computation_types.FunctionType(
            computation_types.FederatedType(tf.int32, placements.CLIENTS),
            computation_types.FederatedType(tf.int32, placements.SERVER)))

    with self.assertRaisesRegex(ValueError, intrinsic.compact_representation()):
      tree_analysis.check_contains_only_reducible_intrinsics(intrinsic)


def whimsy_intrinsic_predicate(x):
  return x.is_intrinsic() and x.uri == 'whimsy_intrinsic'


class NodesDependentOnPredicateTest(absltest.TestCase):

  def test_raises_on_none_comp(self):
    with self.assertRaises(TypeError):
      tree_analysis.extract_nodes_consuming(None, lambda x: True)

  def test_raises_on_none_predicate(self):
    data_type = computation_types.StructType([])
    data = building_blocks.Data('whimsy', data_type)
    with self.assertRaises(TypeError):
      tree_analysis.extract_nodes_consuming(data, None)

  def test_adds_all_nodes_to_set_with_constant_true_predicate(self):
    nested_tree = test_utils.create_nested_syntax_tree()
    all_nodes = tree_analysis.extract_nodes_consuming(nested_tree,
                                                      lambda x: True)
    node_count = tree_analysis.count(nested_tree)
    self.assertLen(all_nodes, node_count)

  def test_adds_no_nodes_to_set_with_constant_false_predicate(self):
    nested_tree = test_utils.create_nested_syntax_tree()
    all_nodes = tree_analysis.extract_nodes_consuming(nested_tree,
                                                      lambda x: False)
    self.assertEmpty(all_nodes)

  def test_propogates_dependence_up_through_lambda(self):
    type_signature = computation_types.TensorType(tf.int32)
    whimsy_intrinsic = building_blocks.Intrinsic('whimsy_intrinsic',
                                                 type_signature)
    lam = building_blocks.Lambda('x', tf.int32, whimsy_intrinsic)
    dependent_nodes = tree_analysis.extract_nodes_consuming(
        lam, whimsy_intrinsic_predicate)
    self.assertIn(lam, dependent_nodes)

  def test_propogates_dependence_up_through_block_result(self):
    type_signature = computation_types.TensorType(tf.int32)
    whimsy_intrinsic = building_blocks.Intrinsic('whimsy_intrinsic',
                                                 type_signature)
    integer_reference = building_blocks.Reference('int', tf.int32)
    block = building_blocks.Block([('x', integer_reference)], whimsy_intrinsic)
    dependent_nodes = tree_analysis.extract_nodes_consuming(
        block, whimsy_intrinsic_predicate)
    self.assertIn(block, dependent_nodes)

  def test_propogates_dependence_up_through_block_locals(self):
    type_signature = computation_types.TensorType(tf.int32)
    whimsy_intrinsic = building_blocks.Intrinsic('whimsy_intrinsic',
                                                 type_signature)
    integer_reference = building_blocks.Reference('int', tf.int32)
    block = building_blocks.Block([('x', whimsy_intrinsic)], integer_reference)
    dependent_nodes = tree_analysis.extract_nodes_consuming(
        block, whimsy_intrinsic_predicate)
    self.assertIn(block, dependent_nodes)

  def test_propogates_dependence_up_through_tuple(self):
    type_signature = computation_types.TensorType(tf.int32)
    whimsy_intrinsic = building_blocks.Intrinsic('whimsy_intrinsic',
                                                 type_signature)
    integer_reference = building_blocks.Reference('int', tf.int32)
    tup = building_blocks.Struct([integer_reference, whimsy_intrinsic])
    dependent_nodes = tree_analysis.extract_nodes_consuming(
        tup, whimsy_intrinsic_predicate)
    self.assertIn(tup, dependent_nodes)

  def test_propogates_dependence_up_through_selection(self):
    type_signature = computation_types.StructType([tf.int32])
    whimsy_intrinsic = building_blocks.Intrinsic('whimsy_intrinsic',
                                                 type_signature)
    selection = building_blocks.Selection(whimsy_intrinsic, index=0)
    dependent_nodes = tree_analysis.extract_nodes_consuming(
        selection, whimsy_intrinsic_predicate)
    self.assertIn(selection, dependent_nodes)

  def test_propogates_dependence_up_through_call(self):
    type_signature = computation_types.TensorType(tf.int32)
    whimsy_intrinsic = building_blocks.Intrinsic('whimsy_intrinsic',
                                                 type_signature)
    ref_to_x = building_blocks.Reference('x', tf.int32)
    identity_lambda = building_blocks.Lambda('x', tf.int32, ref_to_x)
    called_lambda = building_blocks.Call(identity_lambda, whimsy_intrinsic)
    dependent_nodes = tree_analysis.extract_nodes_consuming(
        called_lambda, whimsy_intrinsic_predicate)
    self.assertIn(called_lambda, dependent_nodes)

  def test_propogates_dependence_into_binding_to_reference(self):
    fed_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    ref_to_x = building_blocks.Reference('x', fed_type)
    federated_zero = building_blocks.Intrinsic(intrinsic_defs.GENERIC_ZERO.uri,
                                               fed_type)

    def federated_zero_predicate(x):
      return x.is_intrinsic() and x.uri == intrinsic_defs.GENERIC_ZERO.uri

    block = building_blocks.Block([('x', federated_zero)], ref_to_x)
    dependent_nodes = tree_analysis.extract_nodes_consuming(
        block, federated_zero_predicate)
    self.assertIn(ref_to_x, dependent_nodes)


class BroadcastDependentOnAggregateTest(absltest.TestCase):

  def test_raises_on_none_comp(self):
    with self.assertRaises(TypeError):
      tree_analysis.check_broadcast_not_dependent_on_aggregate(None)

  def test_does_not_find_aggregate_dependent_on_broadcast(self):
    broadcast = test_utils.create_whimsy_called_federated_broadcast()
    value_type = broadcast.type_signature
    zero = building_blocks.Data('zero', value_type.member)
    accumulate_result = building_blocks.Data('accumulate_result',
                                             value_type.member)
    accumulate = building_blocks.Lambda('accumulate_parameter',
                                        [value_type.member, value_type.member],
                                        accumulate_result)
    merge_result = building_blocks.Data('merge_result', value_type.member)
    merge = building_blocks.Lambda('merge_parameter',
                                   [value_type.member, value_type.member],
                                   merge_result)
    report_result = building_blocks.Data('report_result', value_type.member)
    report = building_blocks.Lambda('report_parameter', value_type.member,
                                    report_result)
    aggregate_dependent_on_broadcast = building_block_factory.create_federated_aggregate(
        broadcast, zero, accumulate, merge, report)
    tree_analysis.check_broadcast_not_dependent_on_aggregate(
        aggregate_dependent_on_broadcast)

  def test_finds_broadcast_dependent_on_aggregate(self):
    aggregate = test_utils.create_whimsy_called_federated_aggregate()
    broadcasted_aggregate = building_block_factory.create_federated_broadcast(
        aggregate)
    with self.assertRaises(ValueError):
      tree_analysis.check_broadcast_not_dependent_on_aggregate(
          broadcasted_aggregate)

  def test_returns_correct_example_of_broadcast_dependent_on_aggregate(self):
    aggregate = test_utils.create_whimsy_called_federated_aggregate()
    broadcasted_aggregate = building_block_factory.create_federated_broadcast(
        aggregate)
    with self.assertRaisesRegex(ValueError, 'acc_param'):
      tree_analysis.check_broadcast_not_dependent_on_aggregate(
          broadcasted_aggregate)


class CountTensorFlowOpsTest(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      tree_analysis.count_tensorflow_ops_under(None)

  def test_returns_zero_no_tensorflow(self):
    no_tensorflow_comp = test_utils.create_nested_syntax_tree()
    tf_count = tree_analysis.count_tensorflow_ops_under(no_tensorflow_comp)
    self.assertEqual(tf_count, 0)

  def test_single_tensorflow_node_count_agrees_with_node_count(self):
    tensor_type = computation_types.TensorType(tf.int32)
    integer_identity = building_block_factory.create_compiled_identity(
        tensor_type)
    node_tf_op_count = building_block_analysis.count_tensorflow_ops_in(
        integer_identity)
    tree_tf_op_count = tree_analysis.count_tensorflow_ops_under(
        integer_identity)
    self.assertEqual(node_tf_op_count, tree_tf_op_count)

  def test_tensorflow_op_count_doubles_number_of_ops_in_two_tuple(self):
    tensor_type = computation_types.TensorType(tf.int32)
    integer_identity = building_block_factory.create_compiled_identity(
        tensor_type)
    node_tf_op_count = building_block_analysis.count_tensorflow_ops_in(
        integer_identity)
    tf_tuple = building_blocks.Struct([integer_identity, integer_identity])
    tree_tf_op_count = tree_analysis.count_tensorflow_ops_under(tf_tuple)
    self.assertEqual(tree_tf_op_count, 2 * node_tf_op_count)


def _pack_noarg_graph(graph_def, return_type, result_binding):
  packed_graph_def = serialization_utils.pack_graph_def(graph_def)
  function_type = computation_types.FunctionType(None, return_type)
  proto = pb.Computation(
      type=type_serialization.serialize_type(function_type),
      tensorflow=pb.TensorFlow(
          graph_def=packed_graph_def, parameter=None, result=result_binding))
  building_block = building_blocks.ComputationBuildingBlock.from_proto(proto)
  return building_block


def _create_no_variable_tensorflow():
  with tf.Graph().as_default() as g:
    a = tf.constant(0, name='variable1')
    b = tf.constant(1, name='variable2')
    c = a + b

  result_type, result_binding = tensorflow_utils.capture_result_from_graph(c, g)

  return _pack_noarg_graph(g.as_graph_def(), result_type, result_binding)


def _create_two_variable_tensorflow():
  with tf.Graph().as_default() as g:
    a = tf.Variable(0, name='variable1')
    b = tf.Variable(1, name='variable2')
    c = a + b

  result_type, result_binding = tensorflow_utils.capture_result_from_graph(c, g)

  return _pack_noarg_graph(g.as_graph_def(), result_type, result_binding)


def _create_tensorflow_graph_with_nan():
  with tf.Graph().as_default() as g:
    a = tf.constant(float('NaN'))

  result_type, result_binding = tensorflow_utils.capture_result_from_graph(a, g)

  return _pack_noarg_graph(g.as_graph_def(), result_type, result_binding)


class CountTensorFlowVariablesTest(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      tree_analysis.count_tensorflow_variables_under(None)

  def test_returns_zero_no_tensorflow(self):
    no_tensorflow_comp = test_utils.create_nested_syntax_tree()
    variable_count = tree_analysis.count_tensorflow_variables_under(
        no_tensorflow_comp)
    self.assertEqual(variable_count, 0)

  def test_returns_zero_tensorflow_with_no_variables(self):
    no_variable_comp = _create_no_variable_tensorflow()
    variable_count = tree_analysis.count_tensorflow_variables_under(
        no_variable_comp)
    self.assertEqual(variable_count, 0)

  def test_tensorflow_op_count_doubles_number_of_ops_in_two_tuple(self):
    two_variable_comp = _create_two_variable_tensorflow()
    node_tf_variable_count = building_block_analysis.count_tensorflow_variables_in(
        two_variable_comp)
    tf_tuple = building_blocks.Struct([two_variable_comp, two_variable_comp])
    tree_tf_variable_count = tree_analysis.count_tensorflow_variables_under(
        tf_tuple)
    self.assertEqual(tree_tf_variable_count, 2 * node_tf_variable_count)


class ContainsCalledIntrinsic(absltest.TestCase):

  def test_raises_type_error_with_none_tree(self):
    with self.assertRaises(TypeError):
      tree_analysis.contains_called_intrinsic(None)

  def test_returns_true_with_none_uri(self):
    comp = test_utils.create_whimsy_called_federated_broadcast()
    self.assertTrue(tree_analysis.contains_called_intrinsic(comp))

  def test_returns_true_with_matching_uri(self):
    comp = test_utils.create_whimsy_called_federated_broadcast()
    uri = intrinsic_defs.FEDERATED_BROADCAST.uri
    self.assertTrue(tree_analysis.contains_called_intrinsic(comp, uri))

  def test_returns_false_with_no_called_intrinsic(self):
    comp = test_utils.create_identity_function('a')
    self.assertFalse(tree_analysis.contains_called_intrinsic(comp))

  def test_returns_false_with_unmatched_called_intrinsic(self):
    comp = test_utils.create_whimsy_called_federated_broadcast()
    uri = intrinsic_defs.FEDERATED_MAP.uri
    self.assertFalse(tree_analysis.contains_called_intrinsic(comp, uri))


class ContainsNoUnboundReferencesTest(absltest.TestCase):

  def test_raises_type_error_with_none_tree(self):
    with self.assertRaises(TypeError):
      tree_analysis.contains_no_unbound_references(None)

  def test_raises_type_error_with_int_excluding(self):
    ref = building_blocks.Reference('a', tf.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    with self.assertRaises(TypeError):
      tree_analysis.contains_no_unbound_references(fn, 1)

  def test_returns_true(self):
    ref = building_blocks.Reference('a', tf.int32)
    fn = building_blocks.Lambda(ref.name, ref.type_signature, ref)
    self.assertTrue(tree_analysis.contains_no_unbound_references(fn))

  def test_returns_true_with_excluded_reference(self):
    ref = building_blocks.Reference('a', tf.int32)
    fn = building_blocks.Lambda('b', tf.int32, ref)
    self.assertTrue(
        tree_analysis.contains_no_unbound_references(fn, excluding='a'))

  def test_returns_false(self):
    ref = building_blocks.Reference('a', tf.int32)
    fn = building_blocks.Lambda('b', tf.int32, ref)
    self.assertFalse(tree_analysis.contains_no_unbound_references(fn))


class TreesEqualTest(absltest.TestCase):

  def test_raises_type_error(self):
    data = building_blocks.Data('data', tf.int32)
    with self.assertRaises(TypeError):
      tree_analysis.trees_equal(data, 0)
    with self.assertRaises(TypeError):
      tree_analysis.trees_equal(0, data)

  def test_returns_false_for_block_and_none(self):
    data = building_blocks.Data('data', tf.int32)
    self.assertFalse(tree_analysis.trees_equal(data, None))
    self.assertFalse(tree_analysis.trees_equal(None, data))

  def test_returns_true_for_none_and_none(self):
    self.assertTrue(tree_analysis.trees_equal(None, None))

  def test_returns_true_for_the_same_comp(self):
    data = building_blocks.Data('data', tf.int32)
    self.assertTrue(tree_analysis.trees_equal(data, data))

  def test_returns_false_for_comps_with_different_types(self):
    data = building_blocks.Data('data', tf.int32)
    ref = building_blocks.Reference('a', tf.int32)
    self.assertFalse(tree_analysis.trees_equal(data, ref))
    self.assertFalse(tree_analysis.trees_equal(ref, data))

  def test_returns_false_for_blocks_with_different_results(self):
    data_1 = building_blocks.Data('data', tf.int32)
    comp_1 = building_blocks.Block([], data_1)
    data_2 = building_blocks.Data('data', tf.float32)
    comp_2 = building_blocks.Block([], data_2)
    self.assertFalse(tree_analysis.trees_equal(comp_1, comp_2))

  def test_returns_false_for_blocks_with_different_variable_lengths(self):
    data = building_blocks.Data('data', tf.int32)
    comp_1 = building_blocks.Block([('a', data)], data)
    comp_2 = building_blocks.Block([('a', data), ('b', data)], data)
    self.assertFalse(tree_analysis.trees_equal(comp_1, comp_2))

  def test_returns_true_for_blocks_with_different_variable_names(self):
    data = building_blocks.Data('data', tf.int32)
    comp_1 = building_blocks.Block([('a', data)], data)
    comp_2 = building_blocks.Block([('b', data)], data)
    self.assertTrue(tree_analysis.trees_equal(comp_1, comp_2))

  def test_returns_true_for_blocks_resulting_reference_to_same_local(self):
    data = building_blocks.Data('data', tf.int32)
    ref_to_a = building_blocks.Reference('a', data.type_signature)
    ref_to_b = building_blocks.Reference('b', data.type_signature)
    comp_1 = building_blocks.Block([('a', data)], ref_to_a)
    comp_2 = building_blocks.Block([('b', data)], ref_to_b)
    self.assertTrue(tree_analysis.trees_equal(comp_1, comp_2))

  def test_returns_true_for_blocks_referring_to_same_comp_in_local(self):
    data = building_blocks.Data('data', tf.int32)
    ref_to_a = building_blocks.Reference('a', data.type_signature)
    ref_to_b = building_blocks.Reference('b', data.type_signature)
    comp_1 = building_blocks.Block([('a', data), ('b', ref_to_a)], data)
    comp_2 = building_blocks.Block([('b', data), ('a', ref_to_b)], data)
    self.assertTrue(tree_analysis.trees_equal(comp_1, comp_2))

  def test_returns_true_for_blocks_referring_same_local(self):
    data = building_blocks.Data('data', tf.int32)
    ref_to_a = building_blocks.Reference('a', data.type_signature)
    ref_to_b = building_blocks.Reference('b', data.type_signature)
    comp_1 = building_blocks.Block([('a', data), ('b', ref_to_a)], ref_to_b)
    comp_2 = building_blocks.Block([('b', data), ('a', ref_to_b)], ref_to_a)
    self.assertTrue(tree_analysis.trees_equal(comp_1, comp_2))

  def test_returns_false_for_blocks_referring_to_different_local(self):
    data = building_blocks.Data('data', tf.int32)
    ref_to_a = building_blocks.Reference('a', data.type_signature)
    ref_to_b = building_blocks.Reference('b', data.type_signature)
    comp_1 = building_blocks.Block([('a', data), ('b', ref_to_a)], ref_to_a)
    comp_2 = building_blocks.Block([('b', data), ('a', ref_to_b)], ref_to_a)
    self.assertFalse(tree_analysis.trees_equal(comp_1, comp_2))
    self.assertFalse(tree_analysis.trees_equal(comp_2, comp_1))

  def test_returns_false_for_blocks_with_different_variable_values(self):
    data = building_blocks.Data('data', tf.int32)
    data_1 = building_blocks.Data('data', tf.float32)
    comp_1 = building_blocks.Block([('a', data_1)], data)
    data_2 = building_blocks.Data('data', tf.bool)
    comp_2 = building_blocks.Block([('a', data_2)], data)
    self.assertFalse(tree_analysis.trees_equal(comp_1, comp_2))

  def test_returns_true_for_blocks(self):
    data_1 = building_blocks.Data('data', tf.int32)
    comp_1 = building_blocks.Block([('a', data_1)], data_1)
    data_2 = building_blocks.Data('data', tf.int32)
    comp_2 = building_blocks.Block([('a', data_2)], data_2)
    self.assertTrue(tree_analysis.trees_equal(comp_1, comp_2))

  def test_returns_false_for_calls_with_different_functions(self):
    function_type_1 = computation_types.FunctionType(tf.int32, tf.int32)
    fn_1 = building_blocks.Reference('a', function_type_1)
    arg_1 = building_blocks.Data('data', tf.int32)
    comp_1 = building_blocks.Call(fn_1, arg_1)
    function_type_2 = computation_types.FunctionType(tf.int32, tf.int32)
    fn_2 = building_blocks.Reference('b', function_type_2)
    arg_2 = building_blocks.Data('data', tf.int32)
    comp_2 = building_blocks.Call(fn_2, arg_2)
    self.assertFalse(tree_analysis.trees_equal(comp_1, comp_2))

  def test_returns_false_for_calls_with_different_arguments(self):
    function_type_1 = computation_types.FunctionType(tf.int32, tf.int32)
    fn_1 = building_blocks.Reference('a', function_type_1)
    arg_1 = building_blocks.Data('a', tf.int32)
    comp_1 = building_blocks.Call(fn_1, arg_1)
    function_type_2 = computation_types.FunctionType(tf.int32, tf.int32)
    fn_2 = building_blocks.Reference('a', function_type_2)
    arg_2 = building_blocks.Data('b', tf.int32)
    comp_2 = building_blocks.Call(fn_2, arg_2)
    self.assertFalse(tree_analysis.trees_equal(comp_1, comp_2))

  def test_returns_true_for_calls(self):
    function_type_1 = computation_types.FunctionType(tf.int32, tf.int32)
    fn_1 = building_blocks.Reference('a', function_type_1)
    arg_1 = building_blocks.Data('data', tf.int32)
    comp_1 = building_blocks.Call(fn_1, arg_1)
    function_type_2 = computation_types.FunctionType(tf.int32, tf.int32)
    fn_2 = building_blocks.Reference('a', function_type_2)
    arg_2 = building_blocks.Data('data', tf.int32)
    comp_2 = building_blocks.Call(fn_2, arg_2)
    self.assertTrue(tree_analysis.trees_equal(comp_1, comp_2))

  def test_returns_true_for_calls_with_no_arguments(self):
    function_type_1 = computation_types.FunctionType(None, tf.int32)
    fn_1 = building_blocks.Reference('a', function_type_1)
    comp_1 = building_blocks.Call(fn_1)
    function_type_2 = computation_types.FunctionType(None, tf.int32)
    fn_2 = building_blocks.Reference('a', function_type_2)
    comp_2 = building_blocks.Call(fn_2)
    self.assertTrue(tree_analysis.trees_equal(comp_1, comp_2))

  def test_returns_false_for_compiled_computations_with_different_types(self):
    tensor_type_1 = computation_types.TensorType(tf.int32)
    compiled_1 = building_block_factory.create_compiled_identity(
        tensor_type_1, 'a')
    tensor_type_2 = computation_types.TensorType(tf.float32)
    compiled_2 = building_block_factory.create_compiled_identity(
        tensor_type_2, 'a')
    self.assertFalse(tree_analysis.trees_equal(compiled_1, compiled_2))

  def test_returns_true_for_compiled_computations(self):
    tensor_type = computation_types.TensorType(tf.int32)
    compiled_1 = building_block_factory.create_compiled_identity(
        tensor_type, 'a')
    compiled_2 = building_block_factory.create_compiled_identity(
        tensor_type, 'a')
    self.assertTrue(tree_analysis.trees_equal(compiled_1, compiled_2))

  def test_returns_true_for_compiled_computations_with_different_names(self):
    tensor_type = computation_types.TensorType(tf.int32)
    compiled_1 = building_block_factory.create_compiled_identity(
        tensor_type, 'a')
    compiled_2 = building_block_factory.create_compiled_identity(
        tensor_type, 'b')
    self.assertTrue(tree_analysis.trees_equal(compiled_1, compiled_2))

  def test_returns_false_for_data_with_different_types(self):
    data_1 = building_blocks.Data('data', tf.int32)
    data_2 = building_blocks.Data('data', tf.float32)
    self.assertFalse(tree_analysis.trees_equal(data_1, data_2))

  def test_returns_false_for_data_with_different_names(self):
    data_1 = building_blocks.Data('a', tf.int32)
    data_2 = building_blocks.Data('b', tf.int32)
    self.assertFalse(tree_analysis.trees_equal(data_1, data_2))

  def test_returns_true_for_data(self):
    data_1 = building_blocks.Data('data', tf.int32)
    data_2 = building_blocks.Data('data', tf.int32)
    self.assertTrue(tree_analysis.trees_equal(data_1, data_2))

  def test_returns_false_for_intrinsics_with_different_types(self):
    type_signature_1 = computation_types.TensorType(tf.int32)
    intrinsic_1 = building_blocks.Intrinsic('intrinsic', type_signature_1)
    type_signature_2 = computation_types.TensorType(tf.float32)
    intrinsic_2 = building_blocks.Intrinsic('intrinsic', type_signature_2)
    self.assertFalse(tree_analysis.trees_equal(intrinsic_1, intrinsic_2))

  def test_returns_false_for_intrinsics_with_different_names(self):
    type_signature_1 = computation_types.TensorType(tf.int32)
    intrinsic_1 = building_blocks.Intrinsic('a', type_signature_1)
    type_signature_2 = computation_types.TensorType(tf.int32)
    intrinsic_2 = building_blocks.Intrinsic('b', type_signature_2)
    self.assertFalse(tree_analysis.trees_equal(intrinsic_1, intrinsic_2))

  def test_returns_true_for_intrinsics(self):
    type_signature_1 = computation_types.TensorType(tf.int32)
    intrinsic_1 = building_blocks.Intrinsic('intrinsic', type_signature_1)
    type_signature_2 = computation_types.TensorType(tf.int32)
    intrinsic_2 = building_blocks.Intrinsic('intrinsic', type_signature_2)
    self.assertTrue(tree_analysis.trees_equal(intrinsic_1, intrinsic_2))

  def test_returns_true_for_lambdas_representing_identical_functions(self):
    ref_1 = building_blocks.Reference('a', tf.int32)
    fn_1 = building_blocks.Lambda('a', ref_1.type_signature, ref_1)
    ref_2 = building_blocks.Reference('b', tf.int32)
    fn_2 = building_blocks.Lambda('b', ref_2.type_signature, ref_2)
    self.assertTrue(tree_analysis.trees_equal(fn_1, fn_2))

  def test_returns_false_for_lambdas_with_different_parameter_types(self):
    ref_1 = building_blocks.Reference('a', tf.int32)
    fn_1 = building_blocks.Lambda(ref_1.name, ref_1.type_signature, ref_1)
    ref_2 = building_blocks.Reference('a', tf.float32)
    fn_2 = building_blocks.Lambda(ref_2.name, ref_2.type_signature, ref_2)
    self.assertFalse(tree_analysis.trees_equal(fn_1, fn_2))

  def test_returns_false_for_lambdas_with_different_results(self):
    data_1 = building_blocks.Data('x', tf.int32)
    ref_1 = building_blocks.Reference('a', tf.int32)
    fn_1 = building_blocks.Lambda(ref_1.name, ref_1.type_signature, data_1)
    data_2 = building_blocks.Data('y', tf.int32)
    ref_2 = building_blocks.Reference('b', tf.int32)
    fn_2 = building_blocks.Lambda(ref_2.name, ref_2.type_signature, data_2)
    self.assertFalse(tree_analysis.trees_equal(fn_1, fn_2))

  def test_returns_true_for_lambdas_with_different_parameter_names_but_same_result(
      self):
    data_1 = building_blocks.Data('x', tf.int32)
    ref_1 = building_blocks.Reference('a', tf.int32)
    fn_1 = building_blocks.Lambda(ref_1.name, ref_1.type_signature, data_1)
    ref_2 = building_blocks.Reference('b', tf.int32)
    fn_2 = building_blocks.Lambda(ref_2.name, ref_2.type_signature, data_1)
    self.assertTrue(tree_analysis.trees_equal(fn_1, fn_2))

  def test_returns_false_for_lambdas_referring_to_different_unbound_variables(
      self):
    ref_to_x = building_blocks.Reference('x', tf.int32)
    ref_to_y = building_blocks.Reference('y', tf.int32)
    fn_1 = building_blocks.Lambda('a', tf.int32, ref_to_x)
    fn_2 = building_blocks.Lambda('a', tf.int32, ref_to_y)
    self.assertFalse(tree_analysis.trees_equal(fn_1, fn_2))

  def test_returns_true_for_lambdas_referring_to_same_unbound_variables(self):
    ref_to_x = building_blocks.Reference('x', tf.int32)
    fn_1 = building_blocks.Lambda('a', tf.int32, ref_to_x)
    fn_2 = building_blocks.Lambda('a', tf.int32, ref_to_x)
    self.assertTrue(tree_analysis.trees_equal(fn_1, fn_2))

  def test_returns_true_for_lambdas(self):
    ref_1 = building_blocks.Reference('a', tf.int32)
    fn_1 = building_blocks.Lambda(ref_1.name, ref_1.type_signature, ref_1)
    ref_2 = building_blocks.Reference('a', tf.int32)
    fn_2 = building_blocks.Lambda(ref_2.name, ref_2.type_signature, ref_2)
    self.assertTrue(tree_analysis.trees_equal(fn_1, fn_2))

  def test_returns_false_for_placements_with_literals(self):
    placement_1 = building_blocks.Placement(placements.CLIENTS)
    placement_2 = building_blocks.Placement(placements.SERVER)
    self.assertFalse(tree_analysis.trees_equal(placement_1, placement_2))

  def test_returns_true_for_placements(self):
    placement_1 = building_blocks.Placement(placements.CLIENTS)
    placement_2 = building_blocks.Placement(placements.CLIENTS)
    self.assertTrue(tree_analysis.trees_equal(placement_1, placement_2))

  def test_returns_false_for_references_with_different_types(self):
    reference_1 = building_blocks.Reference('a', tf.int32)
    reference_2 = building_blocks.Reference('a', tf.float32)
    self.assertFalse(tree_analysis.trees_equal(reference_1, reference_2))

  def test_returns_false_for_references_with_different_names(self):
    reference_1 = building_blocks.Reference('a', tf.int32)
    reference_2 = building_blocks.Reference('b', tf.int32)
    self.assertFalse(tree_analysis.trees_equal(reference_1, reference_2))

  def test_returns_true_for_references(self):
    reference_1 = building_blocks.Reference('a', tf.int32)
    reference_2 = building_blocks.Reference('a', tf.int32)
    self.assertTrue(tree_analysis.trees_equal(reference_1, reference_2))

  def test_returns_false_for_selections_with_differet_sources(self):
    ref_1 = building_blocks.Reference('a', [tf.int32, tf.int32])
    selection_1 = building_blocks.Selection(ref_1, index=0)
    ref_2 = building_blocks.Reference('b', [tf.int32, tf.int32])
    selection_2 = building_blocks.Selection(ref_2, index=1)
    self.assertFalse(tree_analysis.trees_equal(selection_1, selection_2))

  def test_returns_false_for_selections_with_different_indexes(self):
    ref_1 = building_blocks.Reference('a', [tf.int32, tf.int32])
    selection_1 = building_blocks.Selection(ref_1, index=0)
    ref_2 = building_blocks.Reference('a', [tf.int32, tf.int32])
    selection_2 = building_blocks.Selection(ref_2, index=1)
    self.assertFalse(tree_analysis.trees_equal(selection_1, selection_2))

  def test_returns_false_for_selections_with_differet_names(self):
    ref_1 = building_blocks.Reference('a', [('a', tf.int32), ('b', tf.int32)])
    selection_1 = building_blocks.Selection(ref_1, name='a')
    ref_2 = building_blocks.Reference('a', [('a', tf.int32), ('b', tf.int32)])
    selection_2 = building_blocks.Selection(ref_2, name='b')
    self.assertFalse(tree_analysis.trees_equal(selection_1, selection_2))

  def test_returns_true_for_selections_with_indexes(self):
    ref_1 = building_blocks.Reference('a', [tf.int32, tf.int32])
    selection_1 = building_blocks.Selection(ref_1, index=0)
    ref_2 = building_blocks.Reference('a', [tf.int32, tf.int32])
    selection_2 = building_blocks.Selection(ref_2, index=0)
    self.assertTrue(tree_analysis.trees_equal(selection_1, selection_2))

  def test_returns_true_for_selections_with_names(self):
    ref_1 = building_blocks.Reference('a', [('a', tf.int32), ('b', tf.int32)])
    selection_1 = building_blocks.Selection(ref_1, name='a')
    ref_2 = building_blocks.Reference('a', [('a', tf.int32), ('b', tf.int32)])
    selection_2 = building_blocks.Selection(ref_2, name='a')
    self.assertTrue(tree_analysis.trees_equal(selection_1, selection_2))

  def test_returns_false_for_tuples_with_different_lengths(self):
    data_1 = building_blocks.Data('data', tf.int32)
    tuple_1 = building_blocks.Struct([data_1])
    data_2 = building_blocks.Data('data', tf.int32)
    tuple_2 = building_blocks.Struct([data_2, data_2])
    self.assertFalse(tree_analysis.trees_equal(tuple_1, tuple_2))

  def test_returns_false_for_tuples_with_different_names(self):
    data_1 = building_blocks.Data('data', tf.int32)
    tuple_1 = building_blocks.Struct([('a', data_1), ('b', data_1)])
    data_2 = building_blocks.Data('data', tf.float32)
    tuple_2 = building_blocks.Struct([('c', data_2), ('d', data_2)])
    self.assertFalse(tree_analysis.trees_equal(tuple_1, tuple_2))

  def test_returns_false_for_tuples_with_different_elements(self):
    data_1 = building_blocks.Data('data', tf.int32)
    tuple_1 = building_blocks.Struct([data_1, data_1])
    data_2 = building_blocks.Data('data', tf.float32)
    tuple_2 = building_blocks.Struct([data_2, data_2])
    self.assertFalse(tree_analysis.trees_equal(tuple_1, tuple_2))

  def test_returns_true_for_tuples(self):
    data_1 = building_blocks.Data('data', tf.int32)
    tuple_1 = building_blocks.Struct([data_1, data_1])
    data_2 = building_blocks.Data('data', tf.int32)
    tuple_2 = building_blocks.Struct([data_2, data_2])
    self.assertTrue(tree_analysis.trees_equal(tuple_1, tuple_2))

  def test_returns_true_for_identical_graphs_with_nans(self):
    tf_comp1 = _create_tensorflow_graph_with_nan()
    tf_comp2 = _create_tensorflow_graph_with_nan()
    self.assertTrue(tree_analysis.trees_equal(tf_comp1, tf_comp2))


non_aggregation_intrinsics = building_blocks.Struct([
    (None, test_utils.create_whimsy_called_federated_broadcast()),
    (None, test_utils.create_whimsy_called_federated_value(placements.CLIENTS))
])

unit = computation_types.StructType([])
trivial_aggregate = test_utils.create_whimsy_called_federated_aggregate(
    value_type=unit)
trivial_collect = test_utils.create_whimsy_called_federated_collect(unit)
trivial_mean = test_utils.create_whimsy_called_federated_mean(unit)
trivial_sum = test_utils.create_whimsy_called_federated_sum(unit)
# TODO(b/120439632) Enable once federated_mean accepts structured weights.
# trivial_weighted_mean = ...
trivial_secure_sum = test_utils.create_whimsy_called_federated_secure_sum(unit)


class ContainsAggregationShared(parameterized.TestCase):

  @parameterized.named_parameters([
      ('non_aggregation_intrinsics', non_aggregation_intrinsics),
      ('trivial_aggregate', trivial_aggregate),
      ('trivial_collect', trivial_collect),
      ('trivial_mean', trivial_mean),
      ('trivial_sum', trivial_sum),
      # TODO(b/120439632) Enable once federated_mean accepts structured weight.
      # ('trivial_weighted_mean', trivial_weighted_mean),
      ('trivial_secure_sum', trivial_secure_sum),
  ])
  def test_returns_none(self, comp):
    self.assertEmpty(tree_analysis.find_unsecure_aggregation_in_tree(comp))
    self.assertEmpty(tree_analysis.find_secure_aggregation_in_tree(comp))

  def test_throws_on_unresolvable_function_call(self):
    comp = building_blocks.Call(
        building_blocks.Data(
            'unknown_func',
            computation_types.FunctionType(
                None, computation_types.at_clients(tf.int32))))
    with self.assertRaises(ValueError):
      tree_analysis.find_unsecure_aggregation_in_tree(comp)
    with self.assertRaises(ValueError):
      tree_analysis.find_secure_aggregation_in_tree(comp)

  # functions without a federated output can't aggregate
  def test_returns_none_on_unresolvable_function_call_with_non_federated_output(
      self):
    input_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    output_type = tf.int32
    comp = building_blocks.Call(
        building_blocks.Data(
            'unknown_func',
            computation_types.FunctionType(input_type, output_type)),
        building_blocks.Data('client_data', input_type))

    self.assertEmpty(tree_analysis.find_unsecure_aggregation_in_tree(comp))
    self.assertEmpty(tree_analysis.find_secure_aggregation_in_tree(comp))


simple_aggregate = test_utils.create_whimsy_called_federated_aggregate()
simple_collect = test_utils.create_whimsy_called_federated_collect()
simple_mean = test_utils.create_whimsy_called_federated_mean()
simple_sum = test_utils.create_whimsy_called_federated_sum()
simple_weighted_mean = test_utils.create_whimsy_called_federated_mean(
    tf.float32, tf.float32)
simple_secure_sum = test_utils.create_whimsy_called_federated_secure_sum()


class ContainsSecureAggregation(parameterized.TestCase):

  @parameterized.named_parameters([
      ('simple_aggregate', simple_aggregate),
      ('simple_collect', simple_collect),
      ('simple_mean', simple_mean),
      ('simple_sum', simple_sum),
      ('simple_weighted_mean', simple_weighted_mean),
  ])
  def test_returns_none_on_unsecure_aggregation(self, comp):
    self.assertEmpty(tree_analysis.find_secure_aggregation_in_tree(comp))

  def assert_one_aggregation(self, comp):
    self.assertLen(tree_analysis.find_secure_aggregation_in_tree(comp), 1)

  def test_returns_str_on_simple_secure_aggregation(self):
    self.assert_one_aggregation(simple_secure_sum)

  def test_returns_str_on_nested_secure_aggregation(self):
    comp = test_utils.create_whimsy_called_federated_secure_sum(
        (tf.int32, tf.int32))
    self.assert_one_aggregation(comp)


class ContainsUnsecureAggregation(parameterized.TestCase):

  def test_returns_none_on_secure_aggregation(self):
    self.assertEmpty(
        tree_analysis.find_unsecure_aggregation_in_tree(simple_secure_sum))

  @parameterized.named_parameters([
      ('simple_aggregate', simple_aggregate),
      ('simple_collect', simple_collect),
      ('simple_mean', simple_mean),
      ('simple_sum', simple_sum),
      ('simple_weighted_mean', simple_weighted_mean),
  ])
  def test_returns_one_on_unsecure_aggregation(self, comp):
    self.assertLen(tree_analysis.find_unsecure_aggregation_in_tree(comp), 1)


if __name__ == '__main__':
  absltest.main()
