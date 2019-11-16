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

from absl.testing import absltest

import tensorflow.compat.v2 as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl.compiler import building_block_analysis
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import intrinsic_defs
from tensorflow_federated.python.core.impl.compiler import test_utils
from tensorflow_federated.python.core.impl.compiler import tree_analysis
from tensorflow_federated.python.core.impl.compiler import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


class IntrinsicsWhitelistedTest(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      tree_analysis.check_intrinsics_whitelisted_for_reduction(None)

  def test_passes_with_federated_map(self):
    intrinsic = building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_MAP.uri,
        computation_types.FunctionType([
            computation_types.FunctionType(tf.int32, tf.float32),
            computation_types.FederatedType(tf.int32, placements.CLIENTS)
        ], computation_types.FederatedType(tf.float32, placements.CLIENTS)))
    tree_analysis.check_intrinsics_whitelisted_for_reduction(intrinsic)

  def test_raises_with_federated_mean(self):
    intrinsic = building_blocks.Intrinsic(
        intrinsic_defs.FEDERATED_MEAN.uri,
        computation_types.FunctionType(
            computation_types.FederatedType(tf.int32, placements.CLIENTS),
            computation_types.FederatedType(tf.int32, placements.SERVER)))

    with self.assertRaisesRegex(ValueError, intrinsic.compact_representation()):
      tree_analysis.check_intrinsics_whitelisted_for_reduction(intrinsic)


def dummy_intrinsic_predicate(x):
  return isinstance(x, building_blocks.Intrinsic) and x.uri == 'dummy_intrinsic'


class NodesDependentOnPredicateTest(absltest.TestCase):

  def test_raises_on_none_comp(self):
    with self.assertRaises(TypeError):
      tree_analysis.extract_nodes_consuming(None, lambda x: True)

  def test_raises_on_none_predicate(self):
    data = building_blocks.Data('dummy', [])
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
    dummy_intrinsic = building_blocks.Intrinsic('dummy_intrinsic', tf.int32)
    lam = building_blocks.Lambda('x', tf.int32, dummy_intrinsic)
    dependent_nodes = tree_analysis.extract_nodes_consuming(
        lam, dummy_intrinsic_predicate)
    self.assertIn(lam, dependent_nodes)

  def test_propogates_dependence_up_through_block_result(self):
    dummy_intrinsic = building_blocks.Intrinsic('dummy_intrinsic', tf.int32)
    integer_reference = building_blocks.Reference('int', tf.int32)
    block = building_blocks.Block([('x', integer_reference)], dummy_intrinsic)
    dependent_nodes = tree_analysis.extract_nodes_consuming(
        block, dummy_intrinsic_predicate)
    self.assertIn(block, dependent_nodes)

  def test_propogates_dependence_up_through_block_locals(self):
    dummy_intrinsic = building_blocks.Intrinsic('dummy_intrinsic', tf.int32)
    integer_reference = building_blocks.Reference('int', tf.int32)
    block = building_blocks.Block([('x', dummy_intrinsic)], integer_reference)
    dependent_nodes = tree_analysis.extract_nodes_consuming(
        block, dummy_intrinsic_predicate)
    self.assertIn(block, dependent_nodes)

  def test_propogates_dependence_up_through_tuple(self):
    dummy_intrinsic = building_blocks.Intrinsic('dummy_intrinsic', tf.int32)
    integer_reference = building_blocks.Reference('int', tf.int32)
    tup = building_blocks.Tuple([integer_reference, dummy_intrinsic])
    dependent_nodes = tree_analysis.extract_nodes_consuming(
        tup, dummy_intrinsic_predicate)
    self.assertIn(tup, dependent_nodes)

  def test_propogates_dependence_up_through_selection(self):
    dummy_intrinsic = building_blocks.Intrinsic('dummy_intrinsic', [tf.int32])
    selection = building_blocks.Selection(dummy_intrinsic, index=0)
    dependent_nodes = tree_analysis.extract_nodes_consuming(
        selection, dummy_intrinsic_predicate)
    self.assertIn(selection, dependent_nodes)

  def test_propogates_dependence_up_through_call(self):
    dummy_intrinsic = building_blocks.Intrinsic('dummy_intrinsic', tf.int32)
    ref_to_x = building_blocks.Reference('x', tf.int32)
    identity_lambda = building_blocks.Lambda('x', tf.int32, ref_to_x)
    called_lambda = building_blocks.Call(identity_lambda, dummy_intrinsic)
    dependent_nodes = tree_analysis.extract_nodes_consuming(
        called_lambda, dummy_intrinsic_predicate)
    self.assertIn(called_lambda, dependent_nodes)

  def test_propogates_dependence_into_binding_to_reference(self):
    fed_type = computation_types.FederatedType(tf.int32, placements.CLIENTS)
    ref_to_x = building_blocks.Reference('x', fed_type)
    federated_zero = building_blocks.Intrinsic(intrinsic_defs.GENERIC_ZERO.uri,
                                               fed_type)

    def federated_zero_predicate(x):
      return isinstance(x, building_blocks.Intrinsic
                       ) and x.uri == intrinsic_defs.GENERIC_ZERO.uri

    block = building_blocks.Block([('x', federated_zero)], ref_to_x)
    dependent_nodes = tree_analysis.extract_nodes_consuming(
        block, federated_zero_predicate)
    self.assertIn(ref_to_x, dependent_nodes)


class BroadcastDependentOnAggregateTest(absltest.TestCase):

  def test_raises_on_none_comp(self):
    with self.assertRaises(TypeError):
      tree_analysis.check_broadcast_not_dependent_on_aggregate(None)

  def test_does_not_find_aggregate_dependent_on_broadcast(self):
    broadcast = test_utils.create_dummy_called_federated_broadcast()
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
    aggregate = test_utils.create_dummy_called_federated_aggregate(
        'accumulate_parameter', 'merge_parameter', 'report_parameter')
    broadcasted_aggregate = building_block_factory.create_federated_broadcast(
        aggregate)
    with self.assertRaises(ValueError):
      tree_analysis.check_broadcast_not_dependent_on_aggregate(
          broadcasted_aggregate)

  def test_returns_correct_example_of_broadcast_dependent_on_aggregate(self):
    aggregate = test_utils.create_dummy_called_federated_aggregate(
        'accumulate_parameter', 'merge_parameter', 'report_parameter')
    broadcasted_aggregate = building_block_factory.create_federated_broadcast(
        aggregate)
    with self.assertRaisesRegex(ValueError, 'accumulate_parameter'):
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
    integer_identity = building_block_factory.create_compiled_identity(tf.int32)
    node_tf_op_count = building_block_analysis.count_tensorflow_ops_in(
        integer_identity)
    tree_tf_op_count = tree_analysis.count_tensorflow_ops_under(
        integer_identity)
    self.assertEqual(node_tf_op_count, tree_tf_op_count)

  def test_tensorflow_op_count_doubles_number_of_ops_in_two_tuple(self):
    integer_identity = building_block_factory.create_compiled_identity(tf.int32)
    node_tf_op_count = building_block_analysis.count_tensorflow_ops_in(
        integer_identity)
    tf_tuple = building_blocks.Tuple([integer_identity, integer_identity])
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
    tf_tuple = building_blocks.Tuple([two_variable_comp, two_variable_comp])
    tree_tf_variable_count = tree_analysis.count_tensorflow_variables_under(
        tf_tuple)
    self.assertEqual(tree_tf_variable_count, 2 * node_tf_variable_count)


if __name__ == '__main__':
  absltest.main()
