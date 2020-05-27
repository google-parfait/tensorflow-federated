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

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import building_block_analysis
from tensorflow_federated.python.core.impl.compiler import building_block_factory
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import tensorflow_computation_transformations
from tensorflow_federated.python.core.impl.compiler import test_utils
from tensorflow_federated.python.core.impl.types import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


def _create_proto_with_unnecessary_op():
  parameter_type = tf.int32

  with tf.Graph().as_default() as graph:
    parameter_value, parameter_binding = tensorflow_utils.stamp_parameter_in_graph(
        'x', parameter_type, graph)
    result_type, result_binding = tensorflow_utils.capture_result_from_graph(
        parameter_value, graph)
    unnecessary_op = tf.constant(0)
    tensorflow_utils.capture_result_from_graph(unnecessary_op, graph)

  function_type = computation_types.FunctionType(parameter_type, result_type)
  serialized_function_type = type_serialization.serialize_type(function_type)
  return pb.Computation(
      type=serialized_function_type,
      tensorflow=pb.TensorFlow(
          graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
          parameter=parameter_binding,
          result=result_binding))


class PruneTensorFlowProtoTest(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      tensorflow_computation_transformations.prune_tensorflow_proto(None)

  def test_raises_on_compiled_computation(self):
    comp = building_block_factory.create_compiled_identity(tf.int32)
    with self.assertRaises(TypeError):
      tensorflow_computation_transformations.prune_tensorflow_proto(comp)

  def test_does_not_reduce_no_unnecessary_ops(self):
    comp = building_block_factory.create_compiled_identity(tf.int32)
    pruned = building_blocks.CompiledComputation(
        tensorflow_computation_transformations.prune_tensorflow_proto(
            comp.proto))
    ops_before = building_block_analysis.count_tensorflow_ops_in(comp)
    ops_after = building_block_analysis.count_tensorflow_ops_in(pruned)
    self.assertEqual(ops_before, ops_after)

  def test_reduces_unnecessary_ops(self):
    proto = _create_proto_with_unnecessary_op()
    comp = building_blocks.CompiledComputation(proto)
    reduced_proto = tensorflow_computation_transformations.prune_tensorflow_proto(
        proto)
    reduced_comp = building_blocks.CompiledComputation(reduced_proto)
    ops_before = building_block_analysis.count_tensorflow_ops_in(comp)
    ops_after = building_block_analysis.count_tensorflow_ops_in(reduced_comp)
    self.assertLess(ops_after, ops_before)

  def test_prune_does_not_change_exeuction(self):
    proto = _create_proto_with_unnecessary_op()
    reduced_proto = tensorflow_computation_transformations.prune_tensorflow_proto(
        proto)
    for k in range(5):
      self.assertEqual(
          test_utils.run_tensorflow(proto, k),
          test_utils.run_tensorflow(reduced_proto, k))


if __name__ == '__main__':
  absltest.main()
