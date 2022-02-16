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

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_transformations
from tensorflow_federated.python.core.impl.types import computation_types
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


def _extract_call_ops(comp: pb.Computation) -> Iterator[tf.compat.v1.NodeDef]:
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

  def assertCallOpsGrapplerNotDisabled(self, comp: pb.Computation):
    call_ops = _extract_call_ops(comp)
    self.assertFalse(all(_is_grappler_disabled(op) for op in call_ops))

  def assertCallOpsGrapplerDisabled(self, comp: pb.Computation):
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
    proto = pb.Computation(
        type=serialized_function_type,
        tensorflow=pb.TensorFlow(
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
    proto = pb.Computation(
        type=serialized_function_type,
        tensorflow=pb.TensorFlow(
            graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
            parameter=None,
            result=result_binding))

    self.assertCallOpsGrapplerNotDisabled(proto)
    transformed_proto = tensorflow_computation_transformations.disable_grappler_for_partitioned_calls(
        proto)
    self.assertCallOpsGrapplerDisabled(transformed_proto)


if __name__ == '__main__':
  absltest.main()
