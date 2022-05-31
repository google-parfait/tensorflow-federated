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

import itertools
from typing import Iterator

from absl.testing import absltest
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_transformations
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import placements
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.types import type_test_utils
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


def _extract_call_ops(
    comp: computation_pb2.Computation) -> Iterator[tf.compat.v1.NodeDef]:
  computation_oneof = comp.WhichOneof('computation')
  if computation_oneof != 'tensorflow':
    raise TypeError('`prune_tensorflow_proto` only accepts `Computation` '
                    'protos of the "tensorflow" variety; you have passed '
                    'one of variety {}.'.format(computation_oneof))
  graph_def = serialization_utils.unpack_graph_def(comp.tensorflow.graph_def)
  all_nodes = itertools.chain(graph_def.node,
                              *[f.node_def for f in graph_def.library.function])
  for node in all_nodes:
    if node.op in tensorflow_computation_transformations.CALL_OPS:
      yield node


def _is_grappler_disabled(node: tf.compat.v1.NodeDef) -> bool:
  serialied_config_proto = node.attr.get('config_proto')
  if serialied_config_proto is None:
    return False
  config_proto = tf.compat.v1.ConfigProto.FromString(serialied_config_proto.s)
  return config_proto.graph_options.rewrite_options.disable_meta_optimizer


class DisableGrapplerForPartitionedCalls(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      tensorflow_computation_transformations.disable_grappler_for_partitioned_calls(
          None)

  def test_raises_on_compiled_computation(self):
    tensor_type = computation_types.TensorType(tf.int32)
    comp = building_block_factory.create_compiled_identity(tensor_type)
    with self.assertRaises(TypeError):
      tensorflow_computation_transformations.disable_grappler_for_partitioned_calls(
          comp)

  def assertCallOpsGrapplerNotDisabled(self, comp: computation_pb2.Computation):
    call_ops = _extract_call_ops(comp)
    self.assertFalse(all(_is_grappler_disabled(op) for op in call_ops))

  def assertCallOpsGrapplerDisabled(self, comp: computation_pb2.Computation):
    call_ops = _extract_call_ops(comp)
    self.assertTrue(all(_is_grappler_disabled(op) for op in call_ops))

  def test_partitioned_call_nodes(self):

    @tf.function
    def test():
      return tf.constant(1)

    with tf.Graph().as_default() as graph:
      result_type, result_binding = tensorflow_utils.capture_result_from_graph(
          test(), graph)

    function_type = computation_types.FunctionType(None, result_type)
    serialized_function_type = type_serialization.serialize_type(function_type)
    proto = computation_pb2.Computation(
        type=serialized_function_type,
        tensorflow=computation_pb2.TensorFlow(
            graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
            parameter=None,
            result=result_binding))

    self.assertCallOpsGrapplerNotDisabled(proto)
    transformed_proto = tensorflow_computation_transformations.disable_grappler_for_partitioned_calls(
        proto)
    self.assertCallOpsGrapplerDisabled(transformed_proto)

  def test_stateful_partitioned_call_nodes(self):

    with tf.Graph().as_default() as graph:
      v = tf.Variable(0)

      @tf.function
      def test():
        return v.assign_add(1)

      result_type, result_binding = tensorflow_utils.capture_result_from_graph(
          test(), graph)

    function_type = computation_types.FunctionType(None, result_type)
    serialized_function_type = type_serialization.serialize_type(function_type)
    proto = computation_pb2.Computation(
        type=serialized_function_type,
        tensorflow=computation_pb2.TensorFlow(
            graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
            parameter=None,
            result=result_binding))

    self.assertCallOpsGrapplerNotDisabled(proto)
    transformed_proto = tensorflow_computation_transformations.disable_grappler_for_partitioned_calls(
        proto)
    self.assertCallOpsGrapplerDisabled(transformed_proto)


class CheckAllowedOps(absltest.TestCase):

  def test_valid_ops(self):

    @tf.function
    def test():
      return tf.constant(1)

    with tf.Graph().as_default() as graph:
      result_type, result_binding = tensorflow_utils.capture_result_from_graph(
          test(), graph)

    function_type = computation_types.FunctionType(None, result_type)
    serialized_function_type = type_serialization.serialize_type(function_type)
    proto = computation_pb2.Computation(
        type=serialized_function_type,
        tensorflow=computation_pb2.TensorFlow(
            graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
            parameter=None,
            result=result_binding))

    allowed_op_names = frozenset(['Const', 'PartitionedCall', 'Identity'])
    tensorflow_computation_transformations.check_allowed_ops(
        proto, allowed_op_names)

  def test_invalid_ops(self):

    @tf.function
    def test():
      return tf.constant(1)

    with tf.Graph().as_default() as graph:
      result_type, result_binding = tensorflow_utils.capture_result_from_graph(
          test(), graph)

    function_type = computation_types.FunctionType(None, result_type)
    serialized_function_type = type_serialization.serialize_type(function_type)
    proto = computation_pb2.Computation(
        type=serialized_function_type,
        tensorflow=computation_pb2.TensorFlow(
            graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
            parameter=None,
            result=result_binding))

    allowed_op_names = frozenset(['Const'])
    with self.assertRaises(tensorflow_computation_transformations
                           .DisallowedOpInTensorFlowComputationError):
      tensorflow_computation_transformations.check_allowed_ops(
          proto, allowed_op_names)


class CheckNoDisallowedOps(absltest.TestCase):

  def test_valid_ops(self):

    @tf.function
    def test():
      return tf.constant(1)

    with tf.Graph().as_default() as graph:
      result_type, result_binding = tensorflow_utils.capture_result_from_graph(
          test(), graph)

    function_type = computation_types.FunctionType(None, result_type)
    serialized_function_type = type_serialization.serialize_type(function_type)
    proto = computation_pb2.Computation(
        type=serialized_function_type,
        tensorflow=computation_pb2.TensorFlow(
            graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
            parameter=None,
            result=result_binding))

    disallowed_op_names = frozenset(['ShardedFilename'])
    tensorflow_computation_transformations.check_no_disallowed_ops(
        proto, disallowed_op_names)

  def test_invalid_ops(self):

    @tf.function
    def test():
      return tf.constant(1)

    with tf.Graph().as_default() as graph:
      result_type, result_binding = tensorflow_utils.capture_result_from_graph(
          test(), graph)

    function_type = computation_types.FunctionType(None, result_type)
    serialized_function_type = type_serialization.serialize_type(function_type)
    proto = computation_pb2.Computation(
        type=serialized_function_type,
        tensorflow=computation_pb2.TensorFlow(
            graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
            parameter=None,
            result=result_binding))

    disallowed_op_names = frozenset(['Const'])
    with self.assertRaises(tensorflow_computation_transformations
                           .DisallowedOpInTensorFlowComputationError):
      tensorflow_computation_transformations.check_no_disallowed_ops(
          proto, disallowed_op_names)


class CompileLocalComputationToTensorFlow(absltest.TestCase):

  def assert_compiles_to_tensorflow(
      self, comp: building_blocks.ComputationBuildingBlock):
    result = tensorflow_computation_transformations.compile_local_computation_to_tensorflow(
        comp)
    if comp.type_signature.is_function():
      result.check_compiled_computation()
    else:
      result.check_call()
      result.function.check_compiled_computation()
    type_test_utils.assert_types_equivalent(comp.type_signature,
                                            result.type_signature)

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

  def test_passes_on_tf(self):
    tf_comp = building_block_factory.create_compiled_identity(
        computation_types.TensorType(tf.int32))
    transformed = tensorflow_computation_transformations.compile_local_computation_to_tensorflow(
        tf_comp)
    self.assertEqual(tf_comp, transformed)

  def test_raises_on_xla(self):
    function_type = computation_types.FunctionType(
        computation_types.TensorType(tf.int32),
        computation_types.TensorType(tf.int32))
    empty_xla_computation_proto = computation_pb2.Computation(
        type=type_serialization.serialize_type(function_type),
        xla=computation_pb2.Xla())

    compiled_comp = building_blocks.CompiledComputation(
        proto=empty_xla_computation_proto)

    with self.assertRaises(
        tensorflow_computation_transformations.XlaToTensorFlowError):
      tensorflow_computation_transformations.compile_local_computation_to_tensorflow(
          compiled_comp)

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


class CompileLocalSubcomputationsToTensorFlowTest(absltest.TestCase):

  def test_leaves_federated_comp_alone(self):
    ref_to_federated_x = building_blocks.Reference(
        'x', computation_types.FederatedType(tf.int32, placements.SERVER))
    identity_lambda = building_blocks.Lambda(ref_to_federated_x.name,
                                             ref_to_federated_x.type_signature,
                                             ref_to_federated_x)
    transformed = tensorflow_computation_transformations.compile_local_subcomputations_to_tensorflow(
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

    transformed = tensorflow_computation_transformations.compile_local_subcomputations_to_tensorflow(
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
    transformed = tensorflow_computation_transformations.compile_local_subcomputations_to_tensorflow(
        lambda_with_unbound_ref)

    self.assertEqual(transformed, lambda_with_unbound_ref)


if __name__ == '__main__':
  absltest.main()
