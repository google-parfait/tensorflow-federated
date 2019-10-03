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
import tensorflow as tf

from tensorflow_federated.proto.v0 import computation_pb2 as pb
from tensorflow_federated.python.common_libs import serialization_utils
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.impl.compiler import building_block_analysis
from tensorflow_federated.python.core.impl.compiler import building_blocks
from tensorflow_federated.python.core.impl.compiler import type_serialization
from tensorflow_federated.python.core.impl.utils import tensorflow_utils


class CountTensorFlowOpsTest(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      building_block_analysis.count_tensorflow_ops_in(None)

  def test_raises_on_reference(self):
    ref = building_blocks.Reference('x', tf.int32)
    with self.assertRaises(ValueError):
      building_block_analysis.count_tensorflow_ops_in(ref)

  def test_counts_correct_number_of_ops_simple_case(self):

    with tf.Graph().as_default() as g:
      a = tf.constant(0)
      b = tf.constant(1)
      c = a + b

    _, result_binding = tensorflow_utils.capture_result_from_graph(c, g)

    packed_graph_def = serialization_utils.pack_graph_def(g.as_graph_def())
    function_type = computation_types.FunctionType(None, tf.int32)
    proto = pb.Computation(
        type=type_serialization.serialize_type(function_type),
        tensorflow=pb.TensorFlow(
            graph_def=packed_graph_def, parameter=None, result=result_binding))
    building_block = building_blocks.ComputationBuildingBlock.from_proto(proto)
    tf_ops_in_graph = building_block_analysis.count_tensorflow_ops_in(
        building_block)
    self.assertEqual(tf_ops_in_graph, 3)


class CountTensorFlowVariablesTest(absltest.TestCase):

  def test_raises_on_none(self):
    with self.assertRaises(TypeError):
      building_block_analysis.count_tensorflow_variables_in(None)

  def test_counts_no_variables(self):

    with tf.Graph().as_default() as g:
      a = tf.constant(0)
      b = tf.constant(1)
      c = a + b

    _, result_binding = tensorflow_utils.capture_result_from_graph(c, g)

    packed_graph_def = serialization_utils.pack_graph_def(g.as_graph_def())
    function_type = computation_types.FunctionType(None, tf.int32)
    proto = pb.Computation(
        type=type_serialization.serialize_type(function_type),
        tensorflow=pb.TensorFlow(
            graph_def=packed_graph_def, parameter=None, result=result_binding))
    building_block = building_blocks.ComputationBuildingBlock.from_proto(proto)
    tf_vars_in_graph = building_block_analysis.count_tensorflow_variables_in(
        building_block)
    self.assertEqual(tf_vars_in_graph, 0)

  def test_avoids_misdirection_with_name(self):

    with tf.Graph().as_default() as g:
      a = tf.constant(0, name='variable1')
      b = tf.constant(1, name='variable2')
      c = a + b

    _, result_binding = tensorflow_utils.capture_result_from_graph(c, g)

    packed_graph_def = serialization_utils.pack_graph_def(g.as_graph_def())
    function_type = computation_types.FunctionType(None, tf.int32)
    proto = pb.Computation(
        type=type_serialization.serialize_type(function_type),
        tensorflow=pb.TensorFlow(
            graph_def=packed_graph_def, parameter=None, result=result_binding))
    building_block = building_blocks.ComputationBuildingBlock.from_proto(proto)
    tf_vars_in_graph = building_block_analysis.count_tensorflow_variables_in(
        building_block)
    self.assertEqual(tf_vars_in_graph, 0)

  def test_counts_two_variables_correctly(self):

    with tf.Graph().as_default() as g:
      a = tf.Variable(0, name='variable1')
      b = tf.Variable(1, name='variable2')
      c = a + b

    _, result_binding = tensorflow_utils.capture_result_from_graph(c, g)

    packed_graph_def = serialization_utils.pack_graph_def(g.as_graph_def())
    function_type = computation_types.FunctionType(None, tf.int32)
    proto = pb.Computation(
        type=type_serialization.serialize_type(function_type),
        tensorflow=pb.TensorFlow(
            graph_def=packed_graph_def, parameter=None, result=result_binding))
    building_block = building_blocks.ComputationBuildingBlock.from_proto(proto)
    tf_vars_in_graph = building_block_analysis.count_tensorflow_variables_in(
        building_block)
    self.assertEqual(tf_vars_in_graph, 2)


if __name__ == '__main__':
  absltest.main()
